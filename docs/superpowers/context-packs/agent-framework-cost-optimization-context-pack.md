# Generated Context Pack

# Agent Framework Cost Optimization

## Sources

- [Specification: 2026-06-25-agent-framework-cost-optimization-design.md](../archive/specs/2026-06-25-agent-framework-cost-optimization-design.md)
- [Plan: 2026-06-25-agent-framework-cost-optimization.md](../archive/plans/2026-06-25-agent-framework-cost-optimization.md)

## Specification Context

### Out of Scope

- Learned stop classifier.
- Reward-weight tuning from real training runs.
- Semantic duplicate detection.
- Broad rewiring of `SearchAgentLoop` onto all component classes.
- New retriever or reranker dependencies.

## Implementation Plan Context

### Task 1: Rerank Request Metrics And Cost Gating

**Files:**
- Modify: `src/agents/search.py`
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Consumes: existing `_execute_search_round(..., metrics: dict[str, float], rerank: bool) -> SearchRoundResult`
- Produces: metrics `rerank_requested`, `rerank_calls`, `rerank_skipped`; helper `_should_rerank(results_by_query: list[list[SearchResult]]) -> bool`

- [x] **Step 1: Write failing tests for skipped reranks**

Add these tests near the existing rerank tests in `tests/unit/test_agent_loop.py`:

Add a second test for empty results:

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_agent_loop.py -k "rerank" -v`

…

### Task 2: Normalized Repeated Query Blocking

**Files:**
- Modify: `src/agents/search.py`
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Consumes: existing `_partition_search_requests(query_specs, executed_queries, rounds_used)`
- Produces: helper `_normalize_query_key(query: str) -> str`; `executed_queries` stores normalized query keys

- [x] **Step 1: Write failing test for normalized repeats**

Add this test near `test_search_agent_loop_skips_repeated_queries_with_feedback` in `tests/unit/test_agent_loop.py`:

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_agent_loop.py -k "repeated_queries" -v`

Expected: the new test fails because the second query is currently treated as distinct.

…

### Final Verification

- [x] Run `git diff --check`.
- [x] Run `pytest tests/unit/test_agent_loop.py -k "rerank or repeated" -v`.
- [x] Run `pytest tests/unit/test_components.py -v`.
- [x] Run `pytest tests/unit/test_reward.py -k "rerank or retriever_aware" -v`.
- [x] Run `git status --short` and confirm only intended files are changed or the branch is clean.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
