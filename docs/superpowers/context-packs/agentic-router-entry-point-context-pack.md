# Generated Context Pack

# Agentic Router Entry Point

## Sources

- [Specification: 2026-06-28-agentic-router-entry-point-design.md](../specs/2026-06-28-agentic-router-entry-point-design.md)
- [Plan: 2026-06-28-agentic-router-entry-point.md](../plans/2026-06-28-agentic-router-entry-point.md)

## Specification Context

### Goal

Make the no-mode (`mode=None`) entry point a real **strategy router**: pick *how*
to answer, then dispatch the matching agent loop — instead of the old 2-way
search-vs-chat branch that always ran a one-shot retrieval.

| Strategy | Loop | Behavior |
|---|---|---|
| `direct_llm` | `llm.complete` / `PlainGenerationLoop` | parametric answer, no retrieval |
| `agentic_rag` | `AgenticRAGLoop` | query decompose + HyDE + grounded synthesis |
| `search_agent` | `SearchAgentLoop` | multi-turn search until evidence suffices |
| `tool_agent` | `ToolAgentLoop` | OpenAPI / MCP function calling |

### Out of scope

- Wiring the M10 retriever-backend router (`src/internal/routing/`) inside the
  chosen loop's retrieval — a separate follow-up (PR B). M10 routes a query to a
  *retriever backend*; this routes to an *agent strategy*.

### Tests

- `tests/unit/servers/web/test_agent_router.py` — pure decision-logic tests for
  the cascade, rule-based routing, and the classifier (incl. garbage → default).
- Auto-route regression in `test_web_experience_app.py` (dispatch contract).

## Implementation Plan Context

### Tasks

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

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
