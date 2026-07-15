# Generated Context Pack

# Agent Framework Optimization Implementation Plan

## Sources

- [Plan: 2026-06-25-agent-framework-optimization-plan.md](../plans/2026-06-25-agent-framework-optimization-plan.md)

## Implementation Plan Context

### Global Constraints

- **Never commit to `main`** — work on branch `feat/agent-framework-optimization` (already created); open a PR at the end.
- **New component args are optional** — existing call sites and tests must pass unchanged.
- **New reward weight defaults to `0.0`** — `sparse_final_only`, `second_pass`, `third_pass_with_format`, and `retriever_aware` totals must be byte-identical when the weight is 0.
- **New loop metric is opt-in** — `evidence_plateau_min_gain` defaults to `None`; with `None` the loop is byte-identical and `early_stops` stays `0.0`.
- **Degrade-don't-crash** on backend failure (log a warning, fall back) — matches the existing pattern.
- **Run `pytest` + `ruff check . --fix && ruff format .` before every commit.**
- **`SearchResult`** fields used in tests: `SearchResult(contents: str, score: float, title: str | None)` (from `src/context/search.py`).
- Commit message trailer on every commit: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

### Task 1: Planner — duplicate-query guard + bounded fallback

**Files:**
- Modify: `src/agents/components/planner.py`
- Test: `tests/unit/test_components.py`

**Interfaces:**
- Consumes: `Retriever` enum (`src/agents/state.py`).
- Produces: `SearchAction(query: str, retriever: Retriever, is_duplicate: bool = False)`; `Planner.decide(self, text: str, previous_queries: Sequence[str] = ()) -> PlannerDecision`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_components.py` (Planner section):

```python
def test_planner_flags_duplicate_query() -> None:
    from src.agents.components.planner import Planner, SearchAction

    decision = Planner().decide(
        "<search>what is faiss</search>", previous_queries=["what is faiss"]
    )

    assert isinstance(decision, SearchAction)
    assert decision.query == "what is faiss"
    assert decision.is_duplicate is True


def test_planner_new_query_not_flagged_duplicate() -> None:
    from src.agents.components.planner import Planner, SearchAction

    decision = Planner().decide(
        "<search>brand new</search>", previous_queries=["what is faiss"]
    )

    assert isinstance(decision, SearchAction)
    assert decision.is_duplicate is False


def test_planner_duplicate_match_ignores_whitespace_and_case() -> None:
    from src.agents.components.planner import Planner

    decision = Planner().decide(
        "<search>  What  Is   FAISS </search>", previous_queries=["what is faiss"]
    )

    assert decision.is_duplicate is True


def test_planner_fallback_query_is_bounded() -> None:
    from src.agents.components.planner import Planner, SearchAction

_[Section compacted.]_

### Task 2: Search Tool — per-instance result cache + web-exception degradation

**Files:**
- Modify: `src/agents/components/search_tool.py`
- Test: `tests/unit/test_components.py`

**Interfaces:**
- Consumes: `Retriever`, `SearchAgentState` (`src/agents/state.py`), `SearchResult`.
- Produces: unchanged `SearchTool(vector_db_fn, web_fn=None)` and `async run(state, query, retriever=Retriever.VECTOR_DB) -> list[SearchResult]`; internally caches by `(retriever, normalized_query)` and degrades web→vdb on exception.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_components.py` (SearchTool section):

```python
async def test_search_tool_caches_repeated_query() -> None:
    from src.agents.components.search_tool import SearchTool

    calls = {"n": 0}

    async def vdb(query: str) -> list[SearchResult]:
        calls["n"] += 1
        return [_result(title="vdb")]

    tool = SearchTool(vdb)
    state = SearchAgentState(question="q")
    await tool.run(state, "faiss index")
    await tool.run(state, "  FAISS   index ")  # same query, different spacing/case

    assert calls["n"] == 1  # second call served from cache, no extra backend hit


async def test_search_tool_caches_per_backend() -> None:
    from src.agents.components.search_tool import SearchTool

    vdb_calls = {"n": 0}
    web_calls = {"n": 0}

    async def vdb(query: str) -> list[SearchResult]:
        vdb_calls["n"] += 1
        return [_result(title="vdb")]

    async def web(query: str) -> list[SearchResult]:
        web_calls["n"] += 1
        return [_result(title="web")]

    tool = SearchTool(vdb, web_fn=web)
    state = SearchAgentState(question="q")

_[Section compacted.]_

### Task 3: Reranker Tool — bounded candidate window + ≤1-doc skip

**Files:**
- Modify: `src/agents/components/reranker_tool.py`
- Test: `tests/unit/test_components.py`

**Interfaces:**
- Consumes: `SearchAgentState`, `SearchResult`, `RerankFn = Callable[[str, list[SearchResult]], list[SearchResult]]`.
- Produces: `RerankerTool(rerank_fn, max_candidates: int | None = None)`; unchanged `run(state, query=None) -> list[SearchResult]`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_components.py` (RerankerTool section):

```python
def test_reranker_tool_skips_when_single_doc() -> None:
    from src.agents.components.reranker_tool import RerankerTool

    called = {"n": 0}

    def fake_rerank(query: str, docs: list[SearchResult]) -> list[SearchResult]:
        called["n"] += 1
        return docs

    state = SearchAgentState(question="q")
    state.record_search("q", [_result(title="only")])

    reordered = RerankerTool(fake_rerank).run(state)

    assert called["n"] == 0  # nothing to reorder with one doc
    assert [d.title for d in reordered] == ["only"]


def test_reranker_tool_limits_to_max_candidates() -> None:
    from src.agents.components.reranker_tool import RerankerTool

    seen_lengths: list[int] = []

    def fake_rerank(query: str, docs: list[SearchResult]) -> list[SearchResult]:
        seen_lengths.append(len(docs))
        return list(reversed(docs))

    state = SearchAgentState(question="q")
    state.record_search(
        "q",
        [_result(title="a"), _result(title="b"), _result(title="c"), _result(title="d")],
    )

    reordered = RerankerTool(fake_rerank, max_candidates=2).run(state)

_[Section compacted.]_

### Task 4: Evidence Judge — marginal-gain + should_stop plateau helpers

**Files:**
- Modify: `src/agents/components/evidence_judge.py`
- Test: `tests/unit/test_components.py`

**Interfaces:**
- Produces (both static, pure): `EvidenceJudge.marginal_gain(prev: float, curr: float) -> float`; `EvidenceJudge.should_stop(prev: float, curr: float, min_gain: float) -> bool`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_components.py` (EvidenceJudge section):

```python
def test_evidence_judge_marginal_gain_is_delta() -> None:
    from src.agents.components.evidence_judge import EvidenceJudge

    assert EvidenceJudge.marginal_gain(0.4, 0.7) == 0.3
    assert EvidenceJudge.marginal_gain(0.7, 0.7) == 0.0


def test_evidence_judge_should_stop_on_plateau() -> None:
    from src.agents.components.evidence_judge import EvidenceJudge

    # gain 0.01 < min_gain 0.05 -> plateau -> stop
    assert EvidenceJudge.should_stop(0.70, 0.71, min_gain=0.05) is True


def test_evidence_judge_should_not_stop_on_real_gain() -> None:
    from src.agents.components.evidence_judge import EvidenceJudge

    # gain 0.20 >= min_gain 0.05 -> keep searching
    assert EvidenceJudge.should_stop(0.50, 0.70, min_gain=0.05) is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_components.py -k "marginal_gain or should_stop or should_not_stop" -v`
Expected: FAIL (`AttributeError: type object 'EvidenceJudge' has no attribute 'marginal_gain'`).

- [ ] **Step 3: Implement the change**

In `src/agents/components/evidence_judge.py`, add two static methods to the `EvidenceJudge` class (place them next to `score_round`):

_[Section compacted.]_

### Task 5: Answer Generator — appearance-ordered, de-duplicated citations

**Files:**
- Modify: `src/agents/components/answer_generator.py`
- Test: `tests/unit/test_components.py`

**Interfaces:**
- Consumes: `AgentContext` (`src/context/search.py`), `Citation`, `SearchAgentState`.
- Produces: unchanged `AnswerGenerator.generate(answer_text, ctx) -> AnswerResult`; citations now ordered by first appearance in `answer_text` and de-duplicated by doc contents.

- [ ] **Step 1: Write the failing tests**

Add these helpers + tests to `tests/unit/test_components.py` (AnswerGenerator section). The two-round context lets us reference markers out of retrieval order:

```python
def _ctx_two_rounds_dup() -> AgentContext:
    ctx = AgentContext()
    ctx.add_round(["q1"], [[_result(text="alpha body", title="A")]])
    # Second round re-retrieves the same "alpha body" plus a new doc.
    ctx.add_round(
        ["q2"],
        [[_result(text="alpha body", title="A"), _result(text="gamma body", title="G")]],
    )
    return ctx


def test_answer_generator_orders_citations_by_appearance() -> None:
    from src.agents.components.answer_generator import AnswerGenerator

    ctx = _ctx_two_rounds_dup()
    # Reference the round-2 doc first, then the round-1 doc.
    answer = "See gamma [R2Q1D2] and also alpha [R1Q1D1]."
    result = AnswerGenerator().generate(answer, ctx)

    assert [c.doc_id for c in result.citations] == ["R2Q1D2", "R1Q1D1"]


def test_answer_generator_collapses_duplicate_doc_contents() -> None:
    from src.agents.components.answer_generator import AnswerGenerator

    ctx = _ctx_two_rounds_dup()

_[Section compacted.]_

### Task 6: Reward — `early_stop_bonus` term (zero-default)

**Files:**
- Modify: `src/training/reward.py`
- Test: `tests/unit/test_reward.py`

**Interfaces:**
- Consumes: `output.metrics["early_stops"]` (a float, default 0 when absent).
- Produces: `SearchRewardConfig.early_stop_bonus: float = 0.0`; reward breakdown gains an `"early_stop_bonus"` component; `retriever_aware(..., early_stop_bonus=0.05)` surfaces it.

- [ ] **Step 1: Write the failing tests**

First, read an existing test in `tests/unit/test_reward.py` to match the construction pattern for `AgentLoopOutput` / `compute`. Then add:

```python
def test_early_stop_bonus_zero_by_default_is_inert() -> None:
    from src.training.reward import SearchRewardConfig

    cfg = SearchRewardConfig.second_pass()
    assert cfg.early_stop_bonus == 0.0


def test_retriever_aware_surfaces_early_stop_bonus() -> None:
    from src.training.reward import SearchRewardConfig

    cfg = SearchRewardConfig.retriever_aware(early_stop_bonus=0.05)
    assert cfg.early_stop_bonus == 0.05
```

Add a breakdown test mirroring the existing `_breakdown`/`compute` style already used in the file (use the same `AgentLoopOutput` factory the neighboring tests use). The assertion that matters:

```python
def test_early_stop_bonus_rewards_early_stops_metric() -> None:
    from src.training.reward import SearchRewardConfig, SearchRewardFunction

    cfg = SearchRewardConfig.retriever_aware(early_stop_bonus=0.1)
    fn = SearchRewardFunction(cfg)
    # Build a metrics dict with two plateau rounds flagged.
    components = fn._reward_components_from_parts(  # match the helper used elsewhere

_[Section compacted.]_

### Task 7: Loop — opt-in `evidence_plateau_min_gain` early-stop metric

**Files:**
- Modify: `src/agents/search.py` (config dataclass; `_initial_metrics`; round-scoring site near line 846)
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Consumes: `EvidenceJudge.should_stop` (Task 4).
- Produces: `SearchAgentLoopConfig.evidence_plateau_min_gain: float | None = None`; `metrics["early_stops"]` initialized to `0.0` and incremented per plateau round when the config is set.

- [ ] **Step 1: Read the surrounding code**

Open `src/agents/search.py`:
- Find `SearchAgentLoopConfig` (around lines 169-229) to add the new field.
- Find `_initial_metrics` (the dict that contains `"web_searches": 0.0, ... "evidence_gain_total": 0.0` near lines 357-363) to register `"early_stops": 0.0`.
- Find the round-scoring block (lines 846-849) where `round_score`, `prev_score` are computed.

- [ ] **Step 2: Write the failing tests**

Open `tests/unit/test_agent_loop.py`, find the existing helper that builds a `SearchAgentLoop` with a fake retriever and runs it (the tests that assert on `output.metrics`). Reusing that harness, add two tests. Replace `make_loop(...)` / `run_loop(...)` below with the file's actual harness names:

```python
async def test_loop_emits_early_stops_when_plateau_configured() -> None:
    # A retriever that returns the SAME strong docs every round → evidence
    # score plateaus after round 1, so round 2's gain is below min_gain.
    loop = make_loop(  # existing harness; add the new config kwarg
        evidence_plateau_min_gain=0.05,
        # ...whatever scripted model output drives >=2 search rounds...
    )

_[Section compacted.]_

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
