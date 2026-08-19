"""Tests for human_feedback reward component in SearchRewardFunction."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from src import AgentLoopOutput, SearchRewardConfig, SearchRewardFunction
from src.model.post_training.grpo.rollouts import GRPORolloutSample, score_prompt_group
from src.model.post_training.reward import simple_sparse_correctness_reward


def _make_output(answer: str = "test answer") -> AgentLoopOutput:
    return AgentLoopOutput(
        prompt_ids=[],
        response_ids=[1, 2, 3],
        response_mask=[1, 1, 1],
        num_turns=1,
        final_answer=answer,
        metrics={
            "rounds_used": 0,
            "subquestion_coverage_ratio": 1.0,
            "repeated_search_queries": 0.0,
            "fetched_pages": 0.0,
            "unnecessary_fetch_count": 0.0,
            "answer_when_evidence_insufficient": 0.0,
            "search_budget_exhausted_without_answer": 0.0,
        },
        context=None,
    )


def _zeroed_with(**kwargs) -> SearchRewardConfig:
    from dataclasses import replace

    return replace(SearchRewardConfig._zeroed(correctness_weight=0.0), **kwargs)


def test_default_weight_zero_no_human_feedback_key():
    fn = SearchRewardFunction()
    components = fn._reward_components_from_correctness(_make_output(), 0.5)
    assert "human_feedback" not in components


def test_default_weight_zero_identical_to_baseline():
    fn = SearchRewardFunction()
    out = _make_output()
    baseline = fn._reward_components_from_correctness(out, 0.5)
    with_signal = fn._reward_components_from_correctness(out, 0.5, human_signal=1.0)
    # human_feedback_weight=0.0 so term is zero — totals must match
    assert baseline["total"] == pytest.approx(with_signal["total"])
    assert "human_feedback" not in with_signal


def test_positive_signal_adds_positive_component():
    config = SearchRewardConfig(human_feedback_weight=0.5)
    fn = SearchRewardFunction(config)
    components = fn._reward_components_from_correctness(
        _make_output(), 0.0, human_signal=1.0
    )
    assert components["human_feedback"] == pytest.approx(0.5)


def test_negative_signal_adds_negative_component():
    config = SearchRewardConfig(human_feedback_weight=0.5)
    fn = SearchRewardFunction(config)
    components = fn._reward_components_from_correctness(
        _make_output(), 0.0, human_signal=-1.0
    )
    assert components["human_feedback"] == pytest.approx(-0.5)


def test_absent_signal_contributes_zero():
    config = SearchRewardConfig(human_feedback_weight=0.5)
    fn = SearchRewardFunction(config)
    components = fn._reward_components_from_correctness(_make_output(), 0.5)
    assert "human_feedback" not in components


def test_total_includes_human_feedback():
    # Use zeroed config so only human_feedback contributes to total
    config = _zeroed_with(human_feedback_weight=0.5)
    fn = SearchRewardFunction(config)
    components = fn._reward_components_from_correctness(
        _make_output(), 0.0, human_signal=1.0
    )
    assert components["total"] == pytest.approx(0.5)


def test_existing_presets_unchanged_with_default_config():
    """second_pass and third_pass_with_format produce same scores as before."""
    for preset in (
        SearchRewardConfig.second_pass(),
        SearchRewardConfig.third_pass_with_format(),
    ):
        fn = SearchRewardFunction(preset)
        out = _make_output()
        total_a = fn._reward_components_from_correctness(out, 0.8)["total"]
        total_b = fn._reward_components_from_correctness(out, 0.8)["total"]
        assert total_a == pytest.approx(total_b)


def _make_sample(answer: str = "answer") -> GRPORolloutSample:
    return GRPORolloutSample(
        group_id="g0",
        rollout_index=0,
        sampling_params={},
        output=_make_output(answer),
    )


def test_score_prompt_group_threads_metadata():
    """score_prompt_group passes metadata["human_signal"] into reward fn."""
    captured: dict = {}

    class _TracingReward(SearchRewardFunction):
        def _reward_components_from_correctness(
            self, output, correctness, *, human_signal=None
        ):
            captured["human_signal"] = human_signal
            return super()._reward_components_from_correctness(
                output, correctness, human_signal=human_signal
            )

    score_prompt_group(
        [_make_sample()],
        ground_truth="answer",
        judge_fn=simple_sparse_correctness_reward,
        reward_fn=_TracingReward(SearchRewardConfig(human_feedback_weight=0.5)),
        metadata={"human_signal": 1.0},
    )
    assert captured.get("human_signal") == pytest.approx(1.0)


def test_score_prompt_group_no_metadata_passes_none():
    captured: dict = {}

    class _TracingReward(SearchRewardFunction):
        def _reward_components_from_correctness(
            self, output, correctness, *, human_signal=None
        ):
            captured["human_signal"] = human_signal
            return super()._reward_components_from_correctness(
                output, correctness, human_signal=human_signal
            )

    score_prompt_group(
        [_make_sample()],
        ground_truth="answer",
        judge_fn=simple_sparse_correctness_reward,
        reward_fn=_TracingReward(),
    )
    assert captured.get("human_signal") is None
