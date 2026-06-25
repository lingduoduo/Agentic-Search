from src.agents.components.loop_controller import LoopController
from src.agents.search import SearchAgentLoopConfig


def _ctl(**over):
    return LoopController(SearchAgentLoopConfig(**over))


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
