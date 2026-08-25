"""Tests for the grouped-rollout GRPO trainer utilities."""

from __future__ import annotations

import copy

import pytest

pytest.importorskip("torch")
import torch

from src.model.post_training.grpo.trainers import (
    Policy,
    compute_group_advantages,
    grpo_clipped_policy_loss,
    make_grpo_trainer,
    reverse_kl_penalty,
)


def test_compute_group_advantages_zero_mean_per_group():
    rewards = torch.tensor([[1.0, 0.5, 0.0]])
    mask = torch.ones(1, 3)
    adv = compute_group_advantages(rewards, mask)
    assert adv.shape == (1, 3)
    assert float(adv.sum()) == pytest.approx(0.0, abs=1e-5)


def test_compute_group_advantages_uniform_rewards_give_zero():
    rewards = torch.tensor([[0.8, 0.8, 0.8]])
    mask = torch.ones(1, 3)
    adv = compute_group_advantages(rewards, mask)
    assert torch.allclose(adv, torch.zeros(1, 3), atol=1e-6)


def test_compute_group_advantages_masked_positions_ignored():
    rewards = torch.tensor([[1.0, 0.0, 999.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    adv = compute_group_advantages(rewards, mask)
    # The 999.0 position is masked and should not affect other advantages.
    assert adv[0, 2].item() == pytest.approx(0.0)
    assert adv[0, 0].item() != pytest.approx(adv[0, 1].item())


def test_compute_group_advantages_detached():
    rewards = torch.tensor([[1.0, 0.5]], requires_grad=True)
    mask = torch.ones(1, 2)
    adv = compute_group_advantages(rewards, mask)
    assert not adv.requires_grad


def test_compute_group_advantages_rejects_mismatched_shapes():
    rewards = torch.ones(2, 3)
    mask = torch.ones(1, 3)

    with pytest.raises(ValueError, match="same shape"):
        compute_group_advantages(rewards, mask)


def test_compute_group_advantages_rejects_groups_without_valid_rollouts():
    rewards = torch.ones(2, 3)
    mask = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="at least one valid rollout"):
        compute_group_advantages(rewards, mask)


@pytest.mark.parametrize("shape", [(0, 3), (2, 0)])
def test_compute_group_advantages_rejects_empty_grouped_dimensions(shape):
    rewards = torch.empty(shape)
    mask = torch.empty(shape)

    with pytest.raises(ValueError, match="non-empty group dimensions"):
        compute_group_advantages(rewards, mask)


def test_grpo_clipped_policy_loss_no_clip_when_ratio_in_range():
    log_p_new = torch.tensor([0.1])
    log_p_old = torch.tensor([0.0])
    adv = torch.tensor([1.0])
    loss, ratios = grpo_clipped_policy_loss(log_p_new, log_p_old, adv, clip_epsilon=0.2)
    expected_ratio = float(torch.exp(torch.tensor(0.1)))
    assert float(ratios[0]) == pytest.approx(expected_ratio, rel=1e-5)
    assert float(loss[0]) == pytest.approx(-expected_ratio, rel=1e-5)


def test_grpo_clipped_policy_loss_clips_large_ratio():
    log_p_new = torch.tensor([1.0])
    log_p_old = torch.tensor([0.0])
    adv = torch.tensor([1.0])
    loss, ratios = grpo_clipped_policy_loss(log_p_new, log_p_old, adv, clip_epsilon=0.2)
    assert float(loss[0]) == pytest.approx(-1.2, rel=1e-4)


def test_reverse_kl_penalty_zero_when_log_probs_equal():
    lp = torch.tensor([0.5, -0.3])
    penalty = reverse_kl_penalty(lp, lp)
    assert torch.allclose(penalty, torch.zeros(2), atol=1e-6)


def test_reverse_kl_penalty_positive_when_policy_deviates():
    policy_lp = torch.tensor([0.0])
    ref_lp = torch.tensor([-1.0])
    penalty = reverse_kl_penalty(policy_lp, ref_lp)
    assert float(penalty[0]) > 0.0


def test_policy_output_shape():
    policy = Policy(state_dim=4, action_dim=3, hidden_dim=16)
    states = torch.randn(8, 4)
    logits = policy(states)
    assert logits.shape == (8, 3)


def test_policy_log_probs_shape():
    policy = Policy(state_dim=4, action_dim=3, hidden_dim=16)
    states = torch.randn(8, 4)
    actions = torch.randint(0, 3, (8,))
    lp = policy.get_log_probs(states, actions)
    assert lp.shape == (8,)
    assert (lp <= 0).all()


def test_grpo_trainer_update_returns_metrics():
    trainer = make_grpo_trainer(state_dim=4, action_dim=3, hidden_dim=16, lr=1e-2)
    states = torch.randn(4 * 3, 4)
    actions = torch.randint(0, 3, (12,))
    old_log_probs = torch.full((12,), -1.0)
    rewards = torch.rand(4, 3)
    mask = torch.ones(4, 3)
    metrics = trainer.update(states, actions, old_log_probs, rewards, mask)
    for key in ("loss", "mean_reward", "mean_advantage", "mean_ratio", "mean_kl"):
        assert key in metrics
        assert isinstance(metrics[key], float)


def test_grpo_trainer_rejects_inconsistent_flat_batch_size():
    trainer = make_grpo_trainer(state_dim=4, action_dim=3)

    with pytest.raises(ValueError, match=r"batch_size \* group_size"):
        trainer.compute_loss(
            states=torch.randn(5, 4),
            actions=torch.randint(0, 3, (5,)),
            old_log_probs=torch.full((5,), -1.0),
            rewards=torch.rand(2, 3),
            mask=torch.ones(2, 3),
        )


def test_grpo_trainer_rejects_an_empty_grouped_batch_before_policy_inference():
    trainer = make_grpo_trainer(state_dim=4, action_dim=3)

    with pytest.raises(ValueError, match="non-empty group dimensions"):
        trainer.compute_loss(
            states=torch.empty(0, 4),
            actions=torch.empty(0, dtype=torch.long),
            old_log_probs=torch.empty(0),
            rewards=torch.empty(0, 3),
            mask=torch.empty(0, 3),
        )


def test_grpo_trainer_skips_reference_policy_when_kl_is_disabled():
    trainer = make_grpo_trainer(state_dim=4, action_dim=3, beta=0.0)

    def fail_if_called(states, actions):
        raise AssertionError("reference policy should not run when beta is zero")

    trainer.reference_policy.get_log_probs = fail_if_called
    states = torch.randn(6, 4)
    actions = torch.randint(0, 3, (6,))
    with torch.no_grad():
        old_log_probs = trainer.policy.get_log_probs(states, actions)

    loss, metrics = trainer.compute_loss(
        states=states,
        actions=actions,
        old_log_probs=old_log_probs,
        rewards=torch.rand(2, 3),
        mask=torch.ones(2, 3),
    )

    assert torch.isfinite(loss)
    assert metrics["mean_kl"] == 0.0


def test_grpo_trainer_loss_decreases_over_steps():
    torch.manual_seed(0)
    trainer = make_grpo_trainer(
        state_dim=8, action_dim=4, hidden_dim=32, lr=5e-3, beta=0.0
    )
    reward_matrix = torch.randn(8, 4)

    def _batch():
        base = torch.randn(16, 8)
        flat = base.unsqueeze(1).repeat(1, 6, 1).reshape(96, 8)
        with torch.no_grad():
            old_p = copy.deepcopy(trainer.policy)
            old_p.eval()
            dist = old_p.get_distribution(flat)
            actions = dist.sample()
            old_lp = dist.log_prob(actions)
        target = (flat @ reward_matrix).argmax(dim=-1)
        r = (actions == target).float().reshape(16, 6)
        return flat, actions, old_lp, r, torch.ones(16, 6)

    first = trainer.update(*_batch())["loss"]
    for _ in range(30):
        trainer.update(*_batch())
    last = trainer.update(*_batch())["loss"]
    # After training, loss should generally be lower (not strictly guaranteed
    # but holds for this deterministic seed / setup).
    assert last < first + 0.5


def test_make_grpo_trainer_reference_frozen():
    trainer = make_grpo_trainer(state_dim=4, action_dim=3)
    for p in trainer.reference_policy.parameters():
        assert not p.requires_grad
