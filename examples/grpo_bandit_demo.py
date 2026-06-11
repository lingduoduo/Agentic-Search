"""Grouped-bandit GRPO training demo.

Runs a synthetic discrete-action bandit where the true preferred action
per state is determined by a fixed reward matrix. Demonstrates that the
GRPOTrainer converges to higher mean reward over ~40 steps.

Usage::

    python -m examples.grpo_bandit_demo
"""

from __future__ import annotations

import copy

import torch

from src.training.ppo.grpo_trainer import GRPOTrainer, Policy


def reward_function(
    states: torch.Tensor,
    actions: torch.Tensor,
    reward_matrix: torch.Tensor,
) -> torch.Tensor:
    """Synthetic reward: 1 if the policy picks the argmax action, else 0, plus shaping."""
    target_logits = states @ reward_matrix
    target_actions = target_logits.argmax(dim=-1)
    rewards = (actions == target_actions).float()
    rewards = rewards + 0.05 * target_logits.gather(
        dim=-1, index=actions.unsqueeze(-1)
    ).squeeze(-1)
    return rewards


def generate_dummy_group_data(
    old_policy: Policy,
    reward_matrix: torch.Tensor,
    batch_size: int = 16,
    group_size: int = 6,
    state_dim: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a grouped batch from old_policy and compute rewards."""
    base_states = torch.randn(batch_size, state_dim)
    flat_states = (
        base_states.unsqueeze(1)
        .repeat(1, group_size, 1)
        .reshape(batch_size * group_size, state_dim)
    )
    with torch.no_grad():
        dist = old_policy.get_distribution(flat_states)
        actions = dist.sample()
        old_log_probs = dist.log_prob(actions)

    rewards = reward_function(flat_states, actions, reward_matrix).reshape(
        batch_size, group_size
    )
    mask = torch.ones(batch_size, group_size)
    return flat_states, actions, old_log_probs, rewards, mask


def run_demo(
    steps: int = 40,
    batch_size: int = 32,
    group_size: int = 8,
    state_dim: int = 8,
    action_dim: int = 4,
    hidden_dim: int = 64,
    lr: float = 1e-2,
) -> dict[str, float]:
    torch.manual_seed(42)

    policy = Policy(
        state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, lr=lr
    )
    reference_policy = copy.deepcopy(policy)

    trainer = GRPOTrainer(
        policy=policy,
        reference_policy=reference_policy,
        clip_epsilon=0.2,
        beta=0.05,
        grad_clip=1.0,
    )

    reward_matrix = torch.randn(state_dim, action_dim)
    latest_metrics: dict[str, float] = {}

    for step in range(steps):
        old_policy = copy.deepcopy(policy)
        old_policy.eval()
        for param in old_policy.parameters():
            param.requires_grad = False

        states, actions, old_log_probs, rewards, mask = generate_dummy_group_data(
            old_policy=old_policy,
            reward_matrix=reward_matrix,
            batch_size=batch_size,
            group_size=group_size,
            state_dim=state_dim,
        )
        latest_metrics = trainer.update(
            states=states,
            actions=actions,
            old_log_probs=old_log_probs,
            rewards=rewards,
            mask=mask,
        )

        if (step + 1) % 5 == 0 or step == 0:
            print(
                f"step={step + 1:02d} "
                f"loss={latest_metrics['loss']:.4f} "
                f"reward={latest_metrics['mean_reward']:.4f} "
                f"ratio={latest_metrics['mean_ratio']:.4f} "
                f"kl={latest_metrics['mean_kl']:.4f}"
            )

    return latest_metrics


if __name__ == "__main__":
    metrics = run_demo()
    print("\nGRPO training finished!")
    print(f"Final loss:    {metrics['loss']:.4f}")
    print(f"Mean reward:   {metrics['mean_reward']:.4f}")
    print(f"Mean ratio:    {metrics['mean_ratio']:.4f}")
    print(f"Mean KL:       {metrics['mean_kl']:.4f}")
