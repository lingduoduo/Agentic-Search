# Generated Context Pack

# Agent Framework Optimization Implementation Plan

## Sources

- [Plan: 2026-06-25-agent-framework-optimization-plan.md](../plans/2026-06-25-agent-framework-optimization-plan.md)

## Implementation Plan Context

### Task 1: Planner — duplicate-query guard + bounded fallback

**Files:**
- Modify: `src/agents/components/planner.py`
- Test: `tests/unit/test_components.py`

**Interfaces:**
- Consumes: `Retriever` enum (`src/agents/state.py`).
- Produces: `SearchAction(query: str, retriever: Retriever, is_duplicate: bool = False)`; `Planner.decide(self, text: str, previous_queries: Sequence[str] = ()) -> PlannerDecision`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_components.py` (Planner section):

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_components.py -k "planner_flags_duplicate or planner_new_query_not or planner_duplicate_match or planner_fallback_query" -v`

…

### Task 2: Search Tool — per-instance result cache + web-exception degradation

**Files:**
- Modify: `src/agents/components/search_tool.py`
- Test: `tests/unit/test_components.py`

**Interfaces:**
- Consumes: `Retriever`, `SearchAgentState` (`src/agents/state.py`), `SearchResult`.
- Produces: unchanged `SearchTool(vector_db_fn, web_fn=None)` and `async run(state, query, retriever=Retriever.VECTOR_DB) -> list[SearchResult]`; internally caches by `(retriever, normalized_query)` and degrades web→vdb on exception.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_components.py` (SearchTool section):

- [ ] **Step 2: Run the tests to verify they fail**

…

### Task 3: Reranker Tool — bounded candidate window + ≤1-doc skip

**Files:**
- Modify: `src/agents/components/reranker_tool.py`
- Test: `tests/unit/test_components.py`

**Interfaces:**
- Consumes: `SearchAgentState`, `SearchResult`, `RerankFn = Callable[[str, list[SearchResult]], list[SearchResult]]`.
- Produces: `RerankerTool(rerank_fn, max_candidates: int | None = None)`; unchanged `run(state, query=None) -> list[SearchResult]`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_components.py` (RerankerTool section):

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_components.py -k "reranker_tool_skips_when_single or reranker_tool_limits" -v`

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
