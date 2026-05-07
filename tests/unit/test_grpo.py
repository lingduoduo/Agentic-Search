"""Tests for grouped rollout helpers used by GRPO-style training."""

from __future__ import annotations

import asyncio

import pytest

from src.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    GRPOAdvantageConfig,
    SearchRewardConfig,
    SearchRewardFunction,
    build_grpo_sampling_params,
    compute_grpo_outcome_advantage,
    sample_prompt_group,
    score_prompt_group,
)


class _DummyLoop(AgentLoopBase):
    def __init__(self, answer: str, metrics: dict[str, float] | None = None) -> None:
        super().__init__(tokenizer=object(), server_manager=object())
        self._answer = answer
        self._metrics = {
            "rounds_used": 0.0,
            "search_rounds": 0.0,
            "subquestion_coverage_ratio": 1.0,
            "repeated_search_queries": 0.0,
            "fetched_pages": 0.0,
            "unnecessary_fetch_count": 0.0,
            "answer_when_evidence_insufficient": 0.0,
            "search_budget_exhausted_without_answer": 0.0,
            "final_evidence_sufficient": 1.0,
            "answer_allowed": 1.0,
            "search_quality_score": 1.0,
        }
        if metrics is not None:
            self._metrics.update(metrics)

    async def run(self, messages, sampling_params):
        del messages, sampling_params
        return AgentLoopOutput(
            prompt_ids=[],
            response_ids=[],
            response_mask=[],
            num_turns=1,
            metrics=dict(self._metrics),
            final_answer=self._answer,
        )


def test_build_grpo_sampling_params_creates_one_variant_per_rollout():
    variants = build_grpo_sampling_params(
        {"temperature": 0.4, "top_p": 0.8, "max_tokens": 64},
        num_rollouts=4,
    )
    assert len(variants) == 4
    assert variants[0]["temperature"] == pytest.approx(0.4)
    assert variants[1]["temperature"] > variants[0]["temperature"]
    assert variants[3]["top_p"] >= variants[0]["top_p"]
    assert variants[0]["max_tokens"] == 64


def test_compute_grpo_outcome_advantage_centers_rewards_by_group_mean():
    advantages = compute_grpo_outcome_advantage([1.0, 0.7, 0.1])
    assert advantages[0] == pytest.approx(0.4)
    assert advantages[1] == pytest.approx(0.1)
    assert advantages[2] == pytest.approx(-0.5)


def test_compute_grpo_outcome_advantage_single_sample_group_returns_zero():
    assert compute_grpo_outcome_advantage([2.0]) == [0.0]


def test_sample_prompt_group_assigns_shared_group_id_and_rollout_indices():
    answers = iter(["direct", "search once", "decompose", "fetch"])

    def loop_factory():
        return _DummyLoop(next(answers))

    samples = asyncio.run(
        sample_prompt_group(
            loop_factory,
            question="What is the answer?",
            sampling_params={"temperature": 0.7, "top_p": 0.9},
            num_rollouts=4,
            group_id="q1",
        )
    )

    assert [sample.group_id for sample in samples] == ["q1", "q1", "q1", "q1"]
    assert [sample.rollout_index for sample in samples] == [0, 1, 2, 3]
    assert [sample.output.group_id for sample in samples] == ["q1", "q1", "q1", "q1"]
    assert [sample.output.rollout_index for sample in samples] == [0, 1, 2, 3]


def test_sample_prompt_group_accepts_prebuilt_messages():
    class _RecordingLoop(_DummyLoop):
        def __init__(self):
            super().__init__("ok")
            self.seen_messages = None

        async def run(self, messages, sampling_params):
            self.seen_messages = list(messages)
            return await super().run(messages, sampling_params)

    holder: dict[str, _RecordingLoop] = {}

    def loop_factory():
        loop = _RecordingLoop()
        holder["loop"] = loop
        return loop

    samples = asyncio.run(
        sample_prompt_group(
            loop_factory,
            messages=[
                {"role": "system", "content": "Available tools: search"},
                {
                    "role": "user",
                    "content": "Who won the Nobel Prize in Physics in 2024?",
                },
            ],
            sampling_params={"temperature": 0.7, "top_p": 0.9},
            num_rollouts=1,
            group_id="q-messages",
        )
    )

    assert samples[0].group_id == "q-messages"
    assert holder["loop"].seen_messages == [
        {"role": "system", "content": "Available tools: search"},
        {"role": "user", "content": "Who won the Nobel Prize in Physics in 2024?"},
    ]


def test_score_prompt_group_normalizes_rewards_within_shared_prompt_group():
    samples = [
        asyncio.run(
            sample_prompt_group(
                lambda: _DummyLoop(
                    "Paris",
                    metrics={"search_rounds": 0.0, "answer_allowed": 1.0},
                ),
                question="Capital of France?",
                sampling_params={"temperature": 0.7, "top_p": 0.9},
                num_rollouts=1,
                group_id="capital",
                sampling_variants=[{"temperature": 0.7, "top_p": 0.9}],
            )
        )[0],
        asyncio.run(
            sample_prompt_group(
                lambda: _DummyLoop(
                    "Wrong",
                    metrics={
                        "search_rounds": 2.0,
                        "rounds_used": 2.0,
                        "repeated_search_queries": 1.0,
                        "answer_when_evidence_insufficient": 1.0,
                        "final_evidence_sufficient": 0.0,
                        "answer_allowed": 0.0,
                        "search_quality_score": 0.2,
                    },
                ),
                question="Capital of France?",
                sampling_params={"temperature": 0.85, "top_p": 0.93},
                num_rollouts=1,
                group_id="capital",
                sampling_variants=[{"temperature": 0.85, "top_p": 0.93}],
            )
        )[0],
    ]

    scored = score_prompt_group(
        samples,
        ground_truth="Paris",
        judge_fn=lambda answer, truth: 1.0 if answer == truth else 0.0,
        reward_fn=SearchRewardFunction(
            SearchRewardConfig(
                citation_support_weight=0.0,
                subquestion_coverage_weight=0.0,
                fetch_usefulness_reward=0.0,
            )
        ),
    )

    assert scored[0].reward > scored[1].reward
    assert scored[0].advantage > 0.0
    assert scored[1].advantage < 0.0


def test_score_prompt_group_supports_group_outcome_advantage_mode():
    samples = [
        asyncio.run(
            sample_prompt_group(
                lambda: _DummyLoop("Paris"),
                question="Capital of France?",
                sampling_params={"temperature": 0.7, "top_p": 0.9},
                num_rollouts=1,
                group_id="capital",
                sampling_variants=[{"temperature": 0.7, "top_p": 0.9}],
            )
        )[0],
        asyncio.run(
            sample_prompt_group(
                lambda: _DummyLoop("Wrong"),
                question="Capital of France?",
                sampling_params={"temperature": 0.85, "top_p": 0.93},
                num_rollouts=1,
                group_id="capital",
                sampling_variants=[{"temperature": 0.85, "top_p": 0.93}],
            )
        )[0],
    ]

    scored = score_prompt_group(
        samples,
        ground_truth="Paris",
        judge_fn=lambda answer, truth: 1.0 if answer == truth else 0.0,
        reward_fn=SearchRewardFunction(SearchRewardConfig.sparse_final_only()),
        advantage_config=GRPOAdvantageConfig(mode="group_outcome"),
    )

    assert scored[0].reward == pytest.approx(1.0)
    assert scored[1].reward == pytest.approx(0.0)
    assert scored[0].advantage == pytest.approx(0.5)
    assert scored[1].advantage == pytest.approx(-0.5)


def test_score_prompt_group_rejects_unknown_advantage_mode():
    samples = asyncio.run(
        sample_prompt_group(
            lambda: _DummyLoop("Paris"),
            question="Capital of France?",
            sampling_params={"temperature": 0.7, "top_p": 0.9},
            num_rollouts=1,
            group_id="capital",
            sampling_variants=[{"temperature": 0.7, "top_p": 0.9}],
        )
    )

    with pytest.raises(ValueError, match="Unsupported GRPO advantage mode"):
        score_prompt_group(
            samples,
            ground_truth="Paris",
            judge_fn=lambda answer, truth: 1.0 if answer == truth else 0.0,
            advantage_config=GRPOAdvantageConfig(mode="unknown"),
        )


def test_sample_prompt_group_rejects_mismatched_sampling_variants():
    with pytest.raises(ValueError, match="sampling_variants length must equal"):
        asyncio.run(
            sample_prompt_group(
                lambda: _DummyLoop("x"),
                question="hello",
                sampling_params={"temperature": 0.7},
                num_rollouts=2,
                sampling_variants=[{"temperature": 0.7}],
            )
        )


def test_sample_prompt_group_requires_exactly_one_prompt_input():
    with pytest.raises(
        ValueError, match="Either `question` or `messages` must be provided."
    ):
        asyncio.run(
            sample_prompt_group(
                lambda: _DummyLoop("x"),
                sampling_params={"temperature": 0.7},
                num_rollouts=1,
            )
        )

    with pytest.raises(
        ValueError, match="Provide only one of `question` or `messages`."
    ):
        asyncio.run(
            sample_prompt_group(
                lambda: _DummyLoop("x"),
                question="hello",
                messages=[{"role": "user", "content": "hello"}],
                sampling_params={"temperature": 0.7},
                num_rollouts=1,
            )
        )
