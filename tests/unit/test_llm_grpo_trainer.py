"""Tests for LLMGRPOTrainer — token-sequence GRPO mapped from the bandit demo.

Uses a minimal toy causal-LM so tests run on CPU without any real weights.
"""

from __future__ import annotations

import copy
from unittest.mock import MagicMock

import pytest

pytest.importorskip("torch")
import torch
import torch.nn as nn

from src.training.grpo.llm_grpo_trainer import (
    LLMGRPOConfig,
    LLMGRPOTrainer,
    LLMRolloutResult,
    _sparse_reward_at_eos,
    get_response_log_probs,
)


# ---------------------------------------------------------------------------
# Minimal toy causal-LM for testing (no real weights needed)
# ---------------------------------------------------------------------------

VOCAB = 32
PROMPT_LEN = 4
RESPONSE_LEN = 6
HIDDEN = 16


class _ToyLM(nn.Module):
    """Tiny causal-LM with deterministic generate() for testing."""

    def __init__(self, vocab: int = VOCAB, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        # Store config-like attributes expected by generate() checks
        self.config = MagicMock()
        self.config.pad_token_id = 0

    def forward(self, input_ids: torch.Tensor, **_) -> MagicMock:
        # (B, L) → (B, L, vocab)
        h = self.embed(input_ids)
        logits = self.lm_head(h)
        out = MagicMock()
        out.logits = logits
        return out

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = RESPONSE_LEN,
        pad_token_id: int = 0,
        **_,
    ) -> torch.Tensor:
        """Greedy decode by always picking token id = 1."""
        B = input_ids.shape[0]
        responses = torch.ones(B, max_new_tokens, dtype=torch.long)
        return torch.cat([input_ids, responses], dim=1)


class _FakeTokenizer:
    """Minimal tokenizer that returns fixed-size zero tensors."""

    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, **_):
        B = len(texts) if isinstance(texts, list) else 1

        class _Enc:
            def __init__(self, b):
                self._b = b

            def __getitem__(self, key):
                if key == "input_ids":
                    return torch.zeros(self._b, PROMPT_LEN, dtype=torch.long)
                return torch.ones(self._b, PROMPT_LEN, dtype=torch.long)

            def to(self, device):
                return self

        return _Enc(B)

    def batch_decode(self, ids, skip_special_tokens=True):
        return [f"answer_{i}" for i in range(ids.shape[0])]


def _make_trainer(
    B: int = 2,
    G: int = 3,
    judge_fn=None,
) -> LLMGRPOTrainer:
    policy = _ToyLM()
    reference = copy.deepcopy(policy)
    tokenizer = _FakeTokenizer()
    optimizer = torch.optim.SGD(policy.parameters(), lr=1e-3)
    cfg = LLMGRPOConfig(num_rollouts=G, max_new_tokens=RESPONSE_LEN)
    resolved_judge = judge_fn or (lambda resp, gt: 1.0 if "answer" in resp else 0.0)
    return LLMGRPOTrainer(
        policy=policy,
        reference_policy=reference,
        tokenizer=tokenizer,
        optimizer=optimizer,
        judge_fn=resolved_judge,
        config=cfg,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sparse_reward_at_eos_places_reward_at_last_valid_token():
    rewards = torch.tensor([1.0, 2.0])
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])  # EOS at pos 1 and 2
    token_rewards = _sparse_reward_at_eos(rewards, mask)
    assert token_rewards[0, 1].item() == pytest.approx(1.0)
    assert token_rewards[1, 2].item() == pytest.approx(2.0)
    # Other positions must be zero.
    assert token_rewards[0, 0].item() == pytest.approx(0.0)
    assert token_rewards[1, 0].item() == pytest.approx(0.0)


def test_get_response_log_probs_shape_and_masking():
    B, P, T = 2, 4, 6
    model = _ToyLM()
    input_ids = torch.zeros(B, P + T, dtype=torch.long)
    mask = torch.ones(B, T, dtype=torch.long)
    mask[0, -2:] = 0  # Two padding positions in row 0

    lp = get_response_log_probs(model, input_ids, P, mask)

    assert lp.shape == (B, T)
    assert (lp[0, -2:] == 0).all(), "Masked positions must be zero"
    assert (lp <= 0).all(), "Log-probs must be non-positive"


def test_llm_grpo_trainer_reference_is_frozen():
    trainer = _make_trainer()
    for p in trainer.reference.parameters():
        assert not p.requires_grad


def test_llm_grpo_trainer_rollout_shapes():
    B, G = 2, 3
    trainer = _make_trainer(B=B, G=G)
    result = trainer.rollout(
        prompts=["What is FAISS?", "What is BM25?"],
        ground_truths=["faiss answer", "bm25 answer"],
    )

    assert isinstance(result, LLMRolloutResult)
    assert result.response_ids.shape[0] == B * G
    assert result.response_mask.shape == result.response_ids.shape
    assert result.old_log_probs.shape == result.response_ids.shape
    assert result.rewards.shape == (B * G,)
    assert result.advantages.shape == result.response_ids.shape
    assert result.group_ids.shape == (B * G,)


def test_llm_grpo_trainer_group_ids_are_correct():
    B, G = 2, 3
    trainer = _make_trainer(B=B, G=G)
    result = trainer.rollout(["p1", "p2"], ["g1", "g2"])
    # Rows 0..G-1 → group 0, rows G..2G-1 → group 1
    assert result.group_ids[:G].tolist() == [0] * G
    assert result.group_ids[G:].tolist() == [1] * G


def test_llm_grpo_trainer_compute_loss_returns_float_metrics():
    B, G = 2, 3
    T = RESPONSE_LEN

    policy = _ToyLM()
    reference = copy.deepcopy(policy)
    optimizer = torch.optim.SGD(policy.parameters(), lr=1e-3)
    cfg = LLMGRPOConfig(num_rollouts=G, max_new_tokens=T)

    trainer = LLMGRPOTrainer(
        policy=policy,
        reference_policy=reference,
        tokenizer=MagicMock(),
        optimizer=optimizer,
        judge_fn=lambda r, g: 1.0,
        config=cfg,
        device="cpu",
    )

    rollout = LLMRolloutResult(
        prompt_ids=torch.zeros(B, PROMPT_LEN, dtype=torch.long),
        response_ids=torch.ones(B * G, T, dtype=torch.long),
        response_mask=torch.ones(B * G, T, dtype=torch.long),
        old_log_probs=torch.full((B * G, T), -1.0),
        rewards=torch.rand(B * G),
        advantages=torch.randn(B * G, T),
        group_ids=torch.arange(B).repeat_interleave(G),
    )
    loss, metrics = trainer.compute_loss(rollout)

    assert loss.requires_grad
    for key in (
        "loss",
        "pg_loss",
        "mean_kl",
        "mean_reward",
        "mean_advantage",
        "clip_fraction",
    ):
        assert key in metrics
        assert isinstance(metrics[key], float)


def test_llm_grpo_config_defaults():
    cfg = LLMGRPOConfig()
    assert cfg.num_rollouts == 4
    assert cfg.kl_beta == pytest.approx(0.04)
    assert cfg.clip_epsilon == pytest.approx(0.2)
    assert cfg.kl_penalty_type == "low_var_kl"
