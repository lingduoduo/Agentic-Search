"""Contracts that make the reward optimizations safe to keep.

These are not timing tests. They pin the two things a fast path can silently
break: the numbers it produces, and how much work it does to produce them
(judge calls, context traversals). Every optimization in `reward.py` is
answerable to one of them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from src.agents.core.base import AgentLoopOutput  # noqa: E402
from src.context.search import AgentContext, SearchResult  # noqa: E402
from src.model.post_training.grpo.algorithms import (  # noqa: E402
    compute_grpo_outcome_advantage,
)
from src.model.post_training.reward import (  # noqa: E402
    SearchRewardConfig,
    SearchRewardFunction,
    compute_group_relative_advantages,
    token_f1_score,
)

BASELINE_PATH = Path(__file__).with_name("reward_breakdown_baseline.json")

PRESETS = {
    "default": SearchRewardConfig(),
    "sparse_final_only": SearchRewardConfig.sparse_final_only(),
    "simple_sparse_with_search_penalty": (
        SearchRewardConfig.simple_sparse_with_search_penalty()
    ),
    "second_pass": SearchRewardConfig.second_pass(),
    "third_pass_with_format": SearchRewardConfig.third_pass_with_format(),
    "retriever_aware": SearchRewardConfig.retriever_aware(),
}


@pytest.fixture(scope="module")
def reward_baseline():
    """Breakdowns captured from the pre-optimization implementation.

    Regenerate deliberately, never to make a red test green:

        python -m tests.unit.test_reward_performance_contracts --write-baseline
    """
    data = json.loads(BASELINE_PATH.read_text())

    def lookup(preset_name: str, answer: str | None) -> dict:
        return data[preset_name][_answer_key(answer)]

    return lookup


def _answer_key(answer: str | None) -> str:
    return "<none>" if answer is None else (answer or "<empty>")


def _context() -> AgentContext:
    ctx = AgentContext()
    ctx.add_round(
        ["q1", "q2"],
        [
            [
                SearchResult(contents=f"doc {i}", url=f"https://example.test/a/{i}")
                for i in range(3)
            ],
            [
                SearchResult(contents=f"doc {i}", url=f"https://example.test/b/{i}")
                for i in range(3)
            ],
        ],
    )
    ctx.record_fetched_pages(
        [SearchResult(contents="p", url="https://example.test/a/0")]
    )
    return ctx


def _output(
    answer: str | None = "Vectors [R1Q1D1] and more [R1Q2D2].",
) -> AgentLoopOutput:
    return AgentLoopOutput(
        prompt_ids=[1, 2, 3],
        response_ids=list(range(8)),
        response_mask=[1] * 6 + [0, 0],
        num_turns=2,
        final_answer=answer,
        context=_context(),
        metrics={
            "rounds_used": 2.0,
            "search_rounds": 2.0,
            "repeated_search_queries": 1.0,
            "unnecessary_fetch_count": 1.0,
            "web_searches": 1.0,
            "vdb_searches": 1.0,
            "rerank_calls": 1.0,
            "search_quality_score": 0.5,
            "final_evidence_sufficient": 1.0,
            "subquestion_coverage_ratio": 0.75,
            "evidence_gain_total": 0.25,
            "early_stops": 1.0,
            "answer_allowed": 1.0,
        },
    )


# ---------------------------------------------------------------------------
# One canonical group-advantage kernel
# ---------------------------------------------------------------------------


def test_outcome_advantages_delegate_to_the_canonical_kernel():
    rewards = [1.0, 0.4, 0.1, 0.9, 0.9]
    groups = ["a", "a", "a", "b", "b"]
    reward_fn = SearchRewardFunction()

    assert reward_fn.compute_grpo_outcome_advantages(rewards, groups) == (
        compute_group_relative_advantages(rewards, groups, normalize=False)
    )


def test_normalized_advantages_delegate_to_the_canonical_kernel():
    rewards = [1.0, 0.4, 0.1, 0.9, 0.2]
    groups = ["a", "a", "a", "b", "b"]
    reward_fn = SearchRewardFunction()

    assert reward_fn.compute_batch_advantages(rewards, groups) == (
        compute_group_relative_advantages(rewards, groups, normalize=True)
    )


def test_list_advantage_entry_point_delegates_to_the_canonical_kernel():
    rewards = [1.0, 0.4, 0.1]

    assert compute_grpo_outcome_advantage(rewards) == (
        compute_group_relative_advantages(rewards, ["g"] * 3, normalize=False)
    )


@pytest.mark.parametrize("normalize", [False, True])
def test_singleton_groups_have_no_within_group_signal(normalize: bool):
    result = compute_group_relative_advantages(
        [3.0, -1.0], ["a", "b"], normalize=normalize
    )

    assert result == [0.0, 0.0]


@pytest.mark.parametrize("normalize", [False, True])
def test_zero_variance_groups_produce_zero_advantages(normalize: bool):
    result = compute_group_relative_advantages(
        [0.5, 0.5, 0.5], ["a"] * 3, normalize=normalize
    )

    assert result == [0.0, 0.0, 0.0]


def test_groups_do_not_leak_into_one_another():
    interleaved = compute_group_relative_advantages(
        [1.0, 10.0, 0.0, 20.0], ["a", "b", "a", "b"], normalize=False
    )

    assert interleaved == [0.5, -5.0, -0.5, 5.0]


def test_results_stay_in_input_order_not_group_order():
    result = compute_group_relative_advantages(
        [0.0, 5.0, 1.0], ["b", "a", "b"], normalize=False
    )

    assert result == [-0.5, 0.0, 0.5]


def test_the_kernel_clips_after_normalizing():
    unclipped = compute_group_relative_advantages(
        [0.0, 1.0], ["a", "a"], normalize=True
    )
    clipped = compute_group_relative_advantages(
        [0.0, 1.0], ["a", "a"], normalize=True, clip_range=(-0.25, 0.25)
    )

    assert min(unclipped) < -0.25 and max(unclipped) > 0.25
    assert clipped == [-0.25, 0.25]


def test_the_kernel_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        compute_group_relative_advantages([1.0, 2.0], ["a"], normalize=False)


def test_the_kernel_handles_empty_input():
    assert compute_group_relative_advantages([], [], normalize=True) == []


# ---------------------------------------------------------------------------
# Judge invocation counts
# ---------------------------------------------------------------------------


def test_reward_components_invokes_the_judge_exactly_once():
    calls: list[tuple[str, str]] = []

    def judge(answer: str, gold: str) -> float:
        calls.append((answer, gold))
        return 1.0

    SearchRewardFunction().reward_components(_output(), "gold", judge)

    assert len(calls) == 1


def test_batch_sparse_rewards_invoke_the_judge_once_per_answer():
    calls = 0

    def judge(answer: str, gold: str) -> float:
        nonlocal calls
        calls += 1
        return 1.0

    outputs = [_output() for _ in range(4)]
    SearchRewardFunction().compute_batch_sparse_token_rewards(
        outputs, ["gold"] * 4, judge
    )

    assert calls == 4


def test_batch_sparse_rewards_prefer_a_single_batch_judge_call():
    single_calls = 0
    batch_calls = 0

    def judge(answer: str, gold: str) -> float:
        nonlocal single_calls
        single_calls += 1
        return 0.0

    def batch_judge(answers: list[str], golds: list[str]) -> list[float]:
        nonlocal batch_calls
        batch_calls += 1
        return [1.0] * len(answers)

    outputs = [_output() for _ in range(4)]
    result = SearchRewardFunction().compute_batch_sparse_token_rewards(
        outputs, ["gold"] * 4, judge, batch_judge_fn=batch_judge
    )

    assert batch_calls == 1
    assert single_calls == 0
    assert [row[5] for row in result] == [1.0] * 4


def test_batch_length_is_validated_before_the_judge_is_invoked():
    def judge(answers, golds):  # pragma: no cover - must never run
        raise AssertionError("judge invoked on invalid input")

    with pytest.raises(ValueError, match="same length"):
        SearchRewardFunction().compute_batch_sparse_token_rewards(
            [_output()], ["gold", "extra"], judge, batch_judge_fn=judge
        )

    with pytest.raises(ValueError, match="same length"):
        SearchRewardFunction().assign_grpo_outcome_token_advantages(
            [_output()], ["gold", "extra"], judge, ["g"], batch_judge_fn=judge
        )


# ---------------------------------------------------------------------------
# Doing less work must not change any number
# ---------------------------------------------------------------------------


def test_answer_citations_are_extracted_once_per_breakdown(monkeypatch):
    output = _output()
    ctx = output.context
    id_calls = 0
    result_calls = 0
    original_ids = ctx.cited_result_ids
    original_results = ctx.cited_results

    def counted_ids(answer_text: str):
        nonlocal id_calls
        id_calls += 1
        return original_ids(answer_text)

    def counted_results(answer_text: str):
        nonlocal result_calls
        result_calls += 1
        return original_results(answer_text)

    monkeypatch.setattr(ctx, "cited_result_ids", counted_ids)
    monkeypatch.setattr(ctx, "cited_results", counted_results)

    SearchRewardFunction().reward_components(output, "gold", token_f1_score)

    assert id_calls <= 1
    assert result_calls <= 1


def test_a_fully_zeroed_config_never_traverses_the_search_context(monkeypatch):
    output = _output()

    def forbidden(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("zero-weight config traversed the search context")

    monkeypatch.setattr(output.context, "cited_result_ids", forbidden)
    monkeypatch.setattr(output.context, "cited_results", forbidden)

    reward_fn = SearchRewardFunction(SearchRewardConfig.sparse_final_only())
    components = reward_fn.reward_components(output, "gold", lambda a, g: 0.75)

    assert components["total"] == pytest.approx(0.75)
    assert components["citation_support"] == 0.0
    assert components["fetch_usefulness_reward"] == 0.0


@pytest.mark.parametrize("preset_name", sorted(PRESETS))
@pytest.mark.parametrize("answer", ["Vectors [R1Q1D1] and more [R1Q2D2].", "", None])
def test_every_preset_keeps_its_exact_breakdown(
    preset_name: str, answer: str | None, reward_baseline
):
    reward_fn = SearchRewardFunction(PRESETS[preset_name])

    components = reward_fn.reward_components(_output(answer), "gold", token_f1_score)

    assert components == reward_baseline(preset_name, answer)


@pytest.mark.parametrize("preset_name", sorted(PRESETS))
def test_every_preset_keeps_its_scalar_total(preset_name: str):
    reward_fn = SearchRewardFunction(PRESETS[preset_name])
    output = _output()

    assert reward_fn.compute(output, "gold", token_f1_score) == pytest.approx(
        reward_fn.reward_components(output, "gold", token_f1_score)["total"]
    )


def test_sparse_token_rewards_land_on_the_last_action_token():
    reward_fn = SearchRewardFunction()
    output = _output()

    vector = reward_fn.compute_sparse_token_rewards(output, "gold", lambda a, g: 1.0)

    assert len(vector) == 8
    assert vector[5] == pytest.approx(reward_fn.config.correctness_weight)
    assert [v for i, v in enumerate(vector) if i != 5] == [0.0] * 7


def test_sparse_token_rewards_fall_back_to_the_last_token_without_a_mask():
    reward_fn = SearchRewardFunction()
    output = _output()
    output.response_mask = [0] * 8

    vector = reward_fn.compute_sparse_token_rewards(output, "gold", lambda a, g: 1.0)

    assert vector[7] == pytest.approx(reward_fn.config.correctness_weight)


def test_token_advantages_and_batch_rewards_share_group_statistics():
    reward_fn = SearchRewardFunction()
    outputs = [_output() for _ in range(4)]
    scores = [1.0, 0.5, 0.0, 0.25]
    judge = dict(zip(range(4), scores))
    seen = iter(scores)

    token_advs = reward_fn.assign_grpo_outcome_token_advantages(
        outputs, ["gold"] * 4, lambda a, g: next(seen), ["x", "x", "y", "y"]
    )

    rewards = [reward_fn.config.correctness_weight * s for s in scores]
    expected = compute_group_relative_advantages(
        rewards, ["x", "x", "y", "y"], normalize=True
    )
    assert [row[5] for row in token_advs] == pytest.approx(expected)
    assert judge  # fixture guard: scores consumed in order


def _write_baseline() -> None:
    """Capture the current breakdowns as the equivalence baseline."""
    data = {
        preset_name: {
            _answer_key(answer): SearchRewardFunction(config).reward_components(
                _output(answer), "gold", token_f1_score
            )
            for answer in ("Vectors [R1Q1D1] and more [R1Q2D2].", "", None)
        }
        for preset_name, config in PRESETS.items()
    }
    BASELINE_PATH.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    _write_baseline()
    print(f"wrote {BASELINE_PATH}")
