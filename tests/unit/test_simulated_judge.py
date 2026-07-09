import pytest

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
