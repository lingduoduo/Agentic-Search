from src.agents.components.loop_controller import (
    LoopController,
    LoopSnapshot,
    StopReason,
)
from src.agents.search import SearchAgentLoopConfig


def _ctl(**over):
    return LoopController(SearchAgentLoopConfig(**over))


def _snap(**over):
    base = dict(
        rounds_used=1,
        num_subquestions=1,
        evidence_sufficient=True,
        prev_evidence_score=0.5,
        curr_evidence_score=0.9,
        consecutive_rejections=0,
        model_emitted_answer=False,
    )
    base.update(over)
    return LoopSnapshot(**base)


def test_effective_limit_single_subquestion_unchanged():
    ctl = _ctl(
        max_search_limit=4, search_budget_per_subquestion=1, max_search_limit_cap=10
    )
    assert ctl.effective_search_limit(1) == 4
    assert ctl.effective_search_limit(0) == 4


def test_effective_limit_scales_then_caps():
    ctl = _ctl(
        max_search_limit=4, search_budget_per_subquestion=1, max_search_limit_cap=6
    )
    assert ctl.effective_search_limit(3) == 6  # 4 + (3-1)=6
    assert ctl.effective_search_limit(20) == 6  # capped


def test_continue_while_evidence_climbing():
    ctl = _ctl(max_search_limit=5, evidence_plateau_min_gain=0.05)
    d = ctl.should_continue_searching(
        _snap(prev_evidence_score=0.2, curr_evidence_score=0.9)
    )
    assert d.reason is StopReason.CONTINUE


def test_stop_budget_exhausted():
    ctl = _ctl(max_search_limit=2, max_search_limit_cap=10)
    d = ctl.should_continue_searching(_snap(rounds_used=2, num_subquestions=1))
    assert d.reason is StopReason.BUDGET_EXHAUSTED


def test_plateau_stops_only_when_sufficient():
    ctl = _ctl(
        max_search_limit=5,
        evidence_plateau_min_gain=0.05,
        plateau_requires_sufficient=True,
    )
    stalled = dict(
        prev_evidence_score=0.80, curr_evidence_score=0.82
    )  # gain 0.02 < 0.05
    assert (
        ctl.should_continue_searching(_snap(evidence_sufficient=True, **stalled)).reason
        is StopReason.PLATEAU
    )
    assert (
        ctl.should_continue_searching(
            _snap(evidence_sufficient=False, **stalled)
        ).reason
        is StopReason.CONTINUE
    )
