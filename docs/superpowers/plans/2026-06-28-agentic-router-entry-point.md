# Agentic Router Entry Point — Plan

Spec: [2026-06-28-agentic-router-entry-point-design.md](../specs/2026-06-28-agentic-router-entry-point-design.md)
Status: shipped (consolidated in PR #347).

**Goal:** `mode=None` picks a `RouteStrategy` via `route_query`, then dispatches
capability-aware. Response contract `(answer, citations, documents, intent, extra)`
unchanged; no degradation path worse than the old binary router.

## Tasks

1. **Decision logic (TDD).** `tests/unit/servers/web/test_agent_router.py`:
   explicit source → SEARCH_AGENT; tool/search/conversational verbs →
   TOOL/SEARCH/DIRECT; default → AGENTIC_RAG; `llm=None` → rule-based;
   `classify_route` parses each label, garbage → AGENTIC_RAG. → red.
2. **Implement** `RouteStrategy`, `_rule_based_route`, `classify_route`,
   `route_query` in `intent_routing.py` (keep `_infer_intent_from_output`). → green.
3. **Dispatch** in `_run_auto_routed`: `route_query` → branch per strategy with
   the degradation table from the spec; set `extra["route"]` /
   `extra["route_degraded"]`. → verify `tests/unit/servers/web/` green.
4. **Regression + lint:** web-experience + SSE suites; `ruff`.

> Note: step 3's per-strategy loop construction was later refactored into shared
> runners — see `2026-06-29-dispatch-consolidation.md`.

## Done when

`route_query` returns the right strategy across the cascade; each route dispatches
its loop (and degrades when its backend is absent); contract + `intent` unchanged;
web/SSE tests pass; `ruff` clean.
