import pytest

pytest.importorskip("torch")

from src.agents.core.base import AgentLoopOutput
from src.training.grpo import GRPORolloutSample, score_prompt_group
from src.training.judge import SimulatedPreferenceJudge, judge_gold_agreement


def test_empty_answer_scores_zero():
    judge = SimulatedPreferenceJudge()
    assert judge.score("") == 0.0
    assert judge.score("   ") == 0.0


def test_score_is_deterministic():
    judge = SimulatedPreferenceJudge()
    a = "James Madison was president when Citibank was founded"
    assert judge.score(a) == judge.score(a)


def test_scores_stay_in_unit_interval():
    judge = SimulatedPreferenceJudge()
    for answer in ["", "x", "paris " * 200, "James Madison was president"]:
        s = judge.score(answer)
        assert 0.0 <= s <= 1.0


def test_varied_answer_beats_degenerate_repetition():
    judge = SimulatedPreferenceJudge()
    varied = "James Madison was the president at that time"
    degenerate = "paris paris paris paris paris paris paris"
    assert judge.score(varied) > judge.score(degenerate)


def test_hedging_answer_is_penalized():
    judge = SimulatedPreferenceJudge()
    concrete = "The answer is James Madison the fourth president"
    hedge = "I don't know the answer to this question really"
    assert judge.score(concrete) > judge.score(hedge)


def test_as_batch_judge_fn_length_and_ignores_ground_truth():
    judge = SimulatedPreferenceJudge()
    fn = judge.as_batch_judge_fn()
    answers = ["James Madison was president", ""]
    scores_no_gt = fn(answers, [])  # ground_truths ignored
    scores_with_gt = fn(answers, ["madison", "madison"])
    assert len(scores_no_gt) == 2
    assert scores_no_gt == scores_with_gt  # ground truth has no effect


def test_agreement_gap_positive_when_correct_scores_higher():
    pairs = [(0.9, True), (0.8, True), (0.2, False), (0.1, False)]
    report = judge_gold_agreement(pairs)
    assert report["mean_score_correct"] == pytest.approx(0.85)
    assert report["mean_score_incorrect"] == pytest.approx(0.15)
    assert report["gap"] > 0
    assert report["n_correct"] == 2.0
    assert report["n_incorrect"] == 2.0


def test_agreement_handles_all_correct():
    pairs = [(0.7, True), (0.9, True)]
    report = judge_gold_agreement(pairs)
    assert report["mean_score_correct"] == 0.8
    assert report["mean_score_incorrect"] == 0.0
    assert report["gap"] == 0.8


def test_agreement_handles_empty_input():
    report = judge_gold_agreement([])
    assert report["gap"] == 0.0
    assert report["n_correct"] == 0.0
    assert report["n_incorrect"] == 0.0


def _fake_sample(group_id: str, idx: int, answer: str) -> GRPORolloutSample:
    output = AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        final_answer=answer,
    )
    return GRPORolloutSample(
        group_id=group_id,
        rollout_index=idx,
        sampling_params={},
        output=output,
    )


def test_sim_judge_drives_nondegenerate_grpo_advantages():
    judge = SimulatedPreferenceJudge()
    samples = [
        _fake_sample("g", 0, "James Madison was the president at that time"),
        _fake_sample("g", 1, ""),
        _fake_sample("g", 2, "paris paris paris paris paris paris"),
    ]
    scored = score_prompt_group(
        samples,
        ground_truth="james madison",
        judge_fn=lambda pred, gold: 0.0,
        batch_judge_fn=judge.as_batch_judge_fn(),
    )
    advantages = [s.advantage for s in scored]
    assert len(advantages) == 3
    # Not all advantages collapse to zero — the judge produced a real spread.
    assert any(abs(a) > 1e-6 for a in advantages)
    # The empty answer must not be the best-advantaged rollout.
    assert scored[1].advantage < max(advantages)


def _scored(idx: int, answer: str, reward: float, advantage: float):
    from src.training.grpo import ScoredGRPORollout

    output = AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        final_answer=answer,
    )
    return ScoredGRPORollout(
        group_id="g",
        rollout_index=idx,
        sampling_params={},
        output=output,
        reward=reward,
        reward_component="total",
        reward_components={"correctness": reward},
        advantage=advantage,
    )


def test_build_synthetic_record_schema():
    from examples.run_bamboogle_synthetic_grpo import build_synthetic_record

    judge = SimulatedPreferenceJudge()
    scored = [
        _scored(0, "James Madison", 0.9, 0.4),
        _scored(1, "", 0.0, -0.4),
    ]
    record = build_synthetic_record(
        prompt="Who was president when Citibank was founded?",
        gold=["james madison"],
        judge=judge,
        scored=scored,
    )
    assert record["prompt"] == "Who was president when Citibank was founded?"
    assert record["gold"] == ["james madison"]
    assert len(record["rollouts"]) == 2
    first = record["rollouts"][0]
    assert set(first) == {
        "answer",
        "judge_score",
        "reward",
        "advantage",
        "exact_match",
        "contains_match",
    }
    assert first["contains_match"] == 1.0  # "James Madison" contains gold
    assert first["judge_score"] == judge.score("James Madison")
    assert record["rollouts"][1]["contains_match"] == 0.0  # empty answer
