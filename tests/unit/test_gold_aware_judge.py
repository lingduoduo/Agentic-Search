"""The judge must read the gold answer.

The judge these replace scored an answer from its own text — length, vocabulary
variety, absence of hedging — so a confidently worded wrong answer outscored a
correct terse one *by construction*. A policy trained against it optimises
answer shape, not correctness.

These tests assert the property that fixes it, and the failure modes that come
with introducing an LLM into a training loop.
"""

import pytest

from src.training.grpo.judge import (
    GoldAgreementJudge,
    JudgeParseError,
    LLMJudge,
    SimulatedPreferenceJudge,
    is_degenerate_group,
    parse_judge_score,
)

_FLUENT_WRONG = (
    "The capital city in question is unquestionably Berlin, a major European "
    "hub with a rich and varied cultural history."
)
_TERSE_RIGHT = "paris"


def test_score_depends_on_the_gold_answer():
    """The single property the previous judge could not have.

    Holding the answer fixed and changing the gold must change the score. This
    is the whole point of the change, so it is asserted directly rather than
    inferred from an accuracy number.
    """
    judge = GoldAgreementJudge()

    assert judge.score("Paris", "Paris") != judge.score("Paris", "Berlin")


def test_a_terse_correct_answer_beats_a_fluent_wrong_one():
    """The exact inversion of the reference-free judge's behaviour.

    Asserted against both judges in one test so the regression is impossible to
    reintroduce quietly: whatever the shape heuristic prefers, the gold-aware
    judge must prefer the correct answer.
    """
    gold = "Paris"
    gold_aware = GoldAgreementJudge()
    shape_only = SimulatedPreferenceJudge()

    assert gold_aware.score(_TERSE_RIGHT, gold) > gold_aware.score(_FLUENT_WRONG, gold)
    # And the judge being replaced gets it backwards, which is why this matters.
    assert shape_only.score(_FLUENT_WRONG) > shape_only.score(_TERSE_RIGHT)


def test_partial_credit_is_graded_rather_than_binary():
    """All-zero groups produce all-zero GRPO advantages.

    A binary judge gives every wrong rollout the same 0.0, the within-group
    advantages are then all 0.0, and the prompt contributes no gradient at all
    while still logging as a completed step. Partial credit keeps near-misses
    informative.
    """
    judge = GoldAgreementJudge()
    near = judge.score("The Eiffel Tower is in Paris, France", "Paris")
    far = judge.score("Berlin", "Paris")

    assert 0.0 < far <= near <= 1.0 or far == 0.0 < near


@pytest.mark.parametrize(
    "raw,expected", [("1", 1.0), ("0", 0.0), ("0.75", 0.75), ("  Score: 0.5 ", 0.5)]
)
def test_parse_judge_score_reads_a_number(raw: str, expected: float):
    assert parse_judge_score(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", "   ", "not a number", "1.5", "-0.2"])
def test_parse_judge_score_raises_rather_than_defaulting(raw: str):
    """Defaulting here would silently turn training into a no-op.

    A judge that returns a middling constant for unparseable replies gives every
    rollout in a group the same score, every advantage becomes 0.0, and the run
    logs as if it worked. Raising forces the caller to decide.
    """
    with pytest.raises(JudgeParseError):
        parse_judge_score(raw)


def test_llm_judge_with_no_provider_is_the_deterministic_judge():
    """The no-network smoke path must keep working unchanged."""
    judge = LLMJudge(llm=None)
    gold_only = GoldAgreementJudge()

    assert judge.score(_TERSE_RIGHT, "Paris") == gold_only.score(_TERSE_RIGHT, "Paris")
    assert judge.llm_calls == 0
    assert judge.parse_failures == 0


def test_llm_judge_falls_back_per_item_and_counts_the_failure():
    """One bad reply must not flatten the group or lose the good rollouts."""

    class _FlakyLLM:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, **kwargs):
            self.calls += 1
            return "banana" if self.calls == 1 else "0.9"

    judge = LLMJudge(llm=_FlakyLLM())
    first = judge.score("some answer", "Paris")
    second = judge.score("another answer", "Paris")

    assert judge.parse_failures == 1
    assert second == pytest.approx(0.9)
    # The failed item still got a real, gold-derived score rather than a constant.
    assert first == pytest.approx(GoldAgreementJudge().score("some answer", "Paris"))


def test_llm_judge_caches_by_answer_and_gold():
    """GRPO scores G rollouts per prompt against one gold, and prompts recur."""

    class _CountingLLM:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, **kwargs):
            self.calls += 1
            return "0.5"

    llm = _CountingLLM()
    judge = LLMJudge(llm=llm)
    for _ in range(4):
        judge.score("same answer", "Paris")
    judge.score("same answer", "Berlin")

    assert llm.calls == 2  # one per distinct (answer, gold), not per call


def test_degenerate_group_is_detectable():
    """All-equal scores mean all-zero advantages -- a step that does nothing."""
    assert is_degenerate_group([0.5, 0.5, 0.5])
    assert is_degenerate_group([0.0, 0.0])
    assert not is_degenerate_group([0.1, 0.9])


def test_the_gold_aware_judge_satisfies_the_batch_seam():
    """It has to be a drop-in for the BatchJudgeFn GRPO already consumes."""
    fn = GoldAgreementJudge().as_batch_judge_fn()

    scores = fn(["Paris", "Berlin"], ["Paris", "Paris"])

    assert len(scores) == 2
    assert scores[0] > scores[1]
