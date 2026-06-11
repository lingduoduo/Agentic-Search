"""Tests for SearchAgentGRPOTrainer — real AgentLoopOutput rollouts."""

from __future__ import annotations

import asyncio
import copy
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from src.agents.base import AgentLoopOutput
from src.training.ppo.llm_grpo_trainer import LLMGRPOConfig
from src.training.ppo.search_agent_grpo_trainer import SearchAgentGRPOTrainer
from src.training.reward import SearchRewardFunction, SearchRewardConfig


# ---------------------------------------------------------------------------
# Tiny toy model (same as test_llm_grpo_trainer.py)
# ---------------------------------------------------------------------------

VOCAB = 32
PROMPT_LEN = 4
RESPONSE_LEN = 5
HIDDEN = 16


class _ToyLM(nn.Module):
    def __init__(self, vocab: int = VOCAB, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.config = MagicMock()
        self.config.pad_token_id = 0

    def forward(self, input_ids: torch.Tensor, **_) -> MagicMock:
        h = self.embed(input_ids)
        logits = self.lm_head(h)
        out = MagicMock()
        out.logits = logits
        return out


# ---------------------------------------------------------------------------
# Fake agent loop
# ---------------------------------------------------------------------------


class _FakeAgentLoop:
    """Returns a canned AgentLoopOutput with realistic metrics."""

    def __init__(self, reward_override: float | None = None) -> None:
        self._reward_override = reward_override

    async def run(
        self,
        messages: list[dict],
        sampling_params: dict,
    ) -> AgentLoopOutput:
        # Vary the answer slightly based on temperature to create non-uniform rewards.
        temp = sampling_params.get("temperature", 0.7)
        answer = "correct answer" if temp < 0.9 else "wrong"
        return AgentLoopOutput(
            prompt_ids=list(range(PROMPT_LEN)),
            response_ids=[1, 2, 3, 4, 5],
            response_mask=[1, 1, 1, 1, 1],
            num_turns=2,
            final_answer=answer,
            metrics={
                "rounds_used": 2.0,
                "subquestion_coverage_ratio": 0.8,
                "final_evidence_sufficient": 1.0,
                "search_quality_score": 0.6,
                "repeated_search_queries": 0.0,
                "unnecessary_fetch_count": 0.0,
                "answer_when_evidence_insufficient": 0.0,
                "search_budget_exhausted_without_answer": 0.0,
            },
            context=None,
        )


def _fake_loop_factory(reward_override: float | None = None):
    def factory() -> _FakeAgentLoop:
        return _FakeAgentLoop(reward_override=reward_override)

    return factory


def _judge(response: str, ground_truth: str) -> float:
    return 1.0 if ground_truth.lower() in response.lower() else 0.0


def _make_trainer(
    B: int = 2,
    G: int = 3,
    reward_fn: SearchRewardFunction | None = None,
) -> SearchAgentGRPOTrainer:
    policy = _ToyLM()
    reference = copy.deepcopy(policy)
    optimizer = torch.optim.SGD(policy.parameters(), lr=1e-3)
    cfg = LLMGRPOConfig(num_rollouts=G, max_new_tokens=RESPONSE_LEN)
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    return SearchAgentGRPOTrainer(
        policy=policy,
        reference_policy=reference,
        tokenizer=tokenizer,
        optimizer=optimizer,
        judge_fn=_judge,
        loop_factory=_fake_loop_factory(),
        config=cfg,
        reward_fn=reward_fn,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rollout_shapes():
    B, G = 2, 3
    trainer = _make_trainer(B=B, G=G)
    result = asyncio.run(
        trainer.rollout_async(
            prompts=["What is FAISS?", "What is BM25?"],
            ground_truths=["correct answer", "correct answer"],
        )
    )
    N = B * G
    assert result.response_ids.shape == (N, RESPONSE_LEN)
    assert result.response_mask.shape == (N, RESPONSE_LEN)
    assert result.old_log_probs.shape == (N, RESPONSE_LEN)
    assert result.rewards.shape == (N,)
    assert result.advantages.shape == (N, RESPONSE_LEN)
    assert result.group_ids.shape == (N,)
    assert result.prompt_ids.shape == (B, PROMPT_LEN)


def test_group_ids_correct():
    B, G = 2, 3
    trainer = _make_trainer(B=B, G=G)
    result = asyncio.run(
        trainer.rollout_async(["p1", "p2"], ["correct answer", "correct answer"])
    )
    assert result.group_ids[:G].tolist() == [0] * G
    assert result.group_ids[G:].tolist() == [1] * G


def test_rewards_are_scalars_not_zero():
    """With judge_fn returning 1.0 for matching answers, rewards must be non-zero."""
    trainer = _make_trainer()
    result = asyncio.run(trainer.rollout_async(["Q"], ["correct answer"]))
    assert result.rewards.shape == (3,)  # G=3
    # At least one rollout should score non-zero (temp < 0.9 gets "correct answer")
    assert result.rewards.sum().item() > 0.0


def test_shaped_rewards_differ_from_judge_only():
    """SearchRewardFunction with search penalties should change the reward values."""
    reward_fn = SearchRewardFunction(
        SearchRewardConfig.simple_sparse_with_search_penalty()
    )
    trainer_shaped = _make_trainer(reward_fn=reward_fn)
    trainer_plain = _make_trainer(reward_fn=None)

    prompts = ["What is retrieval?"]
    gts = ["correct answer"]

    result_shaped = asyncio.run(trainer_shaped.rollout_async(prompts, gts))
    result_plain = asyncio.run(trainer_plain.rollout_async(prompts, gts))

    # With unnecessary_search_penalty=-0.05 and rounds_used=2, rewards should differ.
    assert not torch.allclose(result_shaped.rewards, result_plain.rewards), (
        "Shaped rewards must differ from judge-only rewards when search penalties apply"
    )


def test_compute_loss_after_agent_rollout():
    """compute_loss() must succeed and return float metrics on agent-loop rollout."""
    trainer = _make_trainer()
    result = asyncio.run(
        trainer.rollout_async(["Q1", "Q2"], ["correct answer", "correct answer"])
    )
    loss, metrics = trainer.compute_loss(result)

    assert loss.requires_grad
    for key in ("loss", "pg_loss", "mean_kl", "mean_reward", "clip_fraction"):
        assert key in metrics
        assert isinstance(metrics[key], float)


def test_step_runs_sync():
    """step() must complete without error and return metrics."""
    trainer = _make_trainer(B=1, G=2)
    metrics = trainer.step(["What is GRPO?"], ["correct answer"])
    assert isinstance(metrics, dict)
    assert "loss" in metrics


def test_step_async_avoids_nested_event_loop():
    """step_async() must work when the caller already manages the event loop."""
    trainer = _make_trainer(B=1, G=2)

    async def _run():
        return await trainer.step_async(["What is GRPO?"], ["correct answer"])

    metrics = asyncio.run(_run())
    assert isinstance(metrics, dict)
    assert "loss" in metrics


def test_max_concurrent_limits_parallelism():
    """max_concurrent=1 must serialise rollouts without deadlock."""
    policy = _ToyLM()
    reference = copy.deepcopy(policy)
    optimizer = torch.optim.SGD(policy.parameters(), lr=1e-3)
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    cfg = LLMGRPOConfig(num_rollouts=2, max_new_tokens=RESPONSE_LEN)

    trainer = SearchAgentGRPOTrainer(
        policy=policy,
        reference_policy=reference,
        tokenizer=tokenizer,
        optimizer=optimizer,
        judge_fn=_judge,
        loop_factory=_fake_loop_factory(),
        config=cfg,
        max_concurrent=1,
        device="cpu",
    )
    result = asyncio.run(trainer.rollout_async(["Q"], ["correct answer"]))
    assert result.rewards.shape == (2,)


def test_reference_policy_frozen():
    trainer = _make_trainer()
    for p in trainer.reference.parameters():
        assert not p.requires_grad
