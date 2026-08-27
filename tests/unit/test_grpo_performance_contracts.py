"""Contracts that make the online-path optimizations safe to keep.

Like the reward contracts, these pin work done rather than time taken: how many
model forwards a step costs, what grad mode the frozen reference runs in, and
that the tensors coming out of batch assembly are bit-for-bit what the previous
implementation produced.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.model.post_training.grpo.generation import _left_pad_rows  # noqa: E402
from src.model.post_training.grpo.training import (  # noqa: E402
    LLMGRPOConfig,
    LLMGRPOTrainer,
    LLMRolloutResult,
)
from src.model.post_training.log_probs import get_response_log_probs  # noqa: E402

VOCAB = 11
PROMPT_LEN = 3
RESPONSE_LEN = 4


class _CountingLM(nn.Module):
    """Records every forward call and the grad mode it ran under."""

    def __init__(self, bias: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((VOCAB,), bias))
        self.calls = 0
        self.inference_flags: list[bool] = []
        self.grad_enabled_flags: list[bool] = []

    def forward(self, input_ids: torch.Tensor):  # type: ignore[override]
        self.calls += 1
        self.inference_flags.append(torch.is_inference_mode_enabled())
        self.grad_enabled_flags.append(torch.is_grad_enabled())
        batch, length = input_ids.shape
        positions = torch.arange(length, dtype=torch.float32).view(1, length, 1)
        logits = positions + self.weight.view(1, 1, VOCAB)
        return type("Output", (), {"logits": logits.expand(batch, length, VOCAB)})()


def _rollout(num_prompts: int = 2, num_rollouts: int = 2) -> LLMRolloutResult:
    total = num_prompts * num_rollouts
    return LLMRolloutResult(
        prompt_ids=torch.randint(0, VOCAB, (num_prompts, PROMPT_LEN)),
        response_ids=torch.randint(0, VOCAB, (total, RESPONSE_LEN)),
        response_mask=torch.ones(total, RESPONSE_LEN, dtype=torch.long),
        rewards=torch.rand(total),
        advantages=torch.rand(total, RESPONSE_LEN),
        old_log_probs=torch.zeros(total, RESPONSE_LEN),
        group_ids=torch.arange(num_prompts).repeat_interleave(num_rollouts),
    )


def _trainer(**overrides) -> tuple[LLMGRPOTrainer, _CountingLM, _CountingLM]:
    policy = _CountingLM(0.0)
    reference = _CountingLM(0.1)
    config = LLMGRPOConfig(num_rollouts=2, **overrides)
    trainer = LLMGRPOTrainer(
        policy=policy,
        reference_policy=reference,
        tokenizer=None,
        optimizer=torch.optim.SGD(policy.parameters(), lr=0.0),
        judge_fn=lambda answer, gold: 0.0,
        config=config,
    )
    return trainer, policy, reference


# ---------------------------------------------------------------------------
# Model-forward and grad-mode contracts
# ---------------------------------------------------------------------------


def test_loss_runs_the_policy_once_and_the_reference_once():
    trainer, policy, reference = _trainer()

    trainer.compute_loss(_rollout())

    assert policy.calls == 1
    assert reference.calls == 1


def test_the_frozen_reference_runs_with_grad_disabled():
    """The reference must never build a graph — it is frozen and never stepped.

    This asserts grad is *off*, not that `inference_mode` is on: inference mode
    was benchmarked here and rejected for being slower at realistic model sizes.
    """
    trainer, _, reference = _trainer()

    trainer.compute_loss(_rollout())

    assert reference.grad_enabled_flags == [False]


def test_the_trained_policy_keeps_its_autograd_graph():
    trainer, policy, _ = _trainer()

    loss, _ = trainer.compute_loss(_rollout())

    assert policy.inference_flags == [False]
    assert policy.grad_enabled_flags == [True]
    assert loss.requires_grad


def test_the_loss_still_backpropagates_into_the_policy():
    trainer, policy, _ = _trainer()

    loss, _ = trainer.compute_loss(_rollout())
    loss.backward()

    assert policy.weight.grad is not None
    assert torch.isfinite(policy.weight.grad).all()


def test_a_zero_kl_beta_still_reports_the_reference_kl():
    trainer, _, reference = _trainer(kl_beta=0.0)

    _, metrics = trainer.compute_loss(_rollout())

    assert reference.calls == 1
    assert "mean_kl" in metrics


# ---------------------------------------------------------------------------
# Batch-assembly equivalence
# ---------------------------------------------------------------------------


def _reference_left_pad(rows: list[torch.Tensor], pad_id: int) -> torch.Tensor:
    """The pre-optimization implementation, kept as the equivalence oracle."""
    width = max(row.numel() for row in rows)
    return torch.tensor(
        [[pad_id] * (width - row.numel()) + row.tolist() for row in rows],
        dtype=torch.long,
    )


@pytest.mark.parametrize(
    "lengths",
    [[3], [1, 4, 2], [5, 5, 5], [2, 7, 1, 4, 6]],
)
@pytest.mark.parametrize("pad_id", [0, 7])
def test_left_padding_matches_the_previous_implementation(
    lengths: list[int], pad_id: int
):
    rows = [torch.arange(1, n + 1, dtype=torch.long) for n in lengths]

    actual = _left_pad_rows(rows, pad_id)

    torch.testing.assert_close(actual, _reference_left_pad(rows, pad_id))
    assert actual.dtype == torch.long


def test_left_padding_puts_the_padding_before_the_content():
    rows = [torch.tensor([9, 9]), torch.tensor([5])]

    padded = _left_pad_rows(rows, pad_id=0)

    assert padded.tolist() == [[9, 9], [0, 5]]


def test_left_padding_rejects_an_empty_row_list():
    with pytest.raises(ValueError, match="empty"):
        _left_pad_rows([], pad_id=0)


# ---------------------------------------------------------------------------
# Log-prob alignment is unchanged by any of the above
# ---------------------------------------------------------------------------


def test_response_log_probs_are_unaffected_by_grad_mode():
    model = _CountingLM(0.0)
    ids = torch.randint(0, VOCAB, (2, PROMPT_LEN + RESPONSE_LEN))
    mask = torch.ones(2, RESPONSE_LEN, dtype=torch.long)

    with_grad = get_response_log_probs(model, ids, PROMPT_LEN, mask)
    with torch.inference_mode():
        without_grad = get_response_log_probs(model, ids, PROMPT_LEN, mask)

    torch.testing.assert_close(with_grad, without_grad.clone())
