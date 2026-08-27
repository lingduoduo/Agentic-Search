"""Tests for shared causal-LM response log-probability arithmetic."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
import torch
import torch.nn as nn


class PositionLogitModel(nn.Module):
    """Deterministic logits that distinguish both position and token index."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        batch_size, sequence_len = input_ids.shape
        positions = torch.arange(
            sequence_len, dtype=torch.float32, device=input_ids.device
        ).view(1, sequence_len, 1)
        vocabulary = torch.arange(
            self.vocab_size, dtype=torch.float32, device=input_ids.device
        ).view(1, 1, self.vocab_size)
        logits = positions * 10 + vocabulary
        return SimpleNamespace(logits=logits.expand(batch_size, -1, -1))


def test_response_log_probs_align_logits_with_the_tokens_they_predict():
    torch = pytest.importorskip("torch", exc_type=ImportError)
    from src.model.post_training.log_probs import get_response_log_probs

    model = PositionLogitModel(vocab_size=7)
    ids = torch.tensor([[1, 2, 3, 4, 0]])
    mask = torch.tensor([[1, 1, 0]])
    actual = get_response_log_probs(model, ids, prompt_len=2, response_mask=mask)
    expected = torch.log_softmax(model(ids).logits[:, 1:-1], dim=-1)
    expected = expected.gather(-1, ids[:, 2:].unsqueeze(-1)).squeeze(-1) * mask
    torch.testing.assert_close(actual, expected)


def test_dpo_and_grpo_use_the_shared_helper():
    from src.model.post_training import log_probs
    from src.model.post_training.dpo import trainer as dpo_trainer
    from src.model.post_training.grpo import trainers as grpo_trainers

    assert dpo_trainer.get_response_log_probs is log_probs.get_response_log_probs
    assert grpo_trainers.get_response_log_probs is log_probs.get_response_log_probs


def test_grpo_lazy_export_accepts_input_ids_keyword():
    from src.model.post_training.grpo import get_response_log_probs

    model = PositionLogitModel(vocab_size=7)
    input_ids = torch.tensor([[1, 2, 3, 4, 0]])
    response_mask = torch.tensor([[1, 1, 0]])

    actual = get_response_log_probs(
        model,
        input_ids=input_ids,
        prompt_len=2,
        response_mask=response_mask,
    )
    expected = torch.log_softmax(model(input_ids).logits[:, 1:-1], dim=-1)
    expected = (
        expected.gather(-1, input_ids[:, 2:].unsqueeze(-1)).squeeze(-1) * response_mask
    )
    torch.testing.assert_close(actual, expected)
