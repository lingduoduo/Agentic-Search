# Generated Context Pack

# Intent Capture Parity

## Sources

- [Specification: 2026-07-05-intent-capture-parity-design.md](../archive/specs/2026-07-05-intent-capture-parity-design.md)
- [Plan: 2026-07-05-intent-capture-parity.md](../archive/plans/2026-07-05-intent-capture-parity.md)

## Specification Context

### Goal

Emit exactly **one** `intent` capture stage per request from a single point in
`route_query`, covering every decision path (explicit source, regex, LLM
classifier, no-LLM rule-based), so the Request Inspector always shows how a query
was routed and by which mechanism. Preserve the richer classifier detail
(`prompt`, `raw_label`) that exists today.

## Implementation Plan Context

### Task 1: Centralize intent capture in `route_query`

**Files:**
- Modify: `src/internal/servers/web/intent_routing.py` (`classify_route`; add `_record_intent`; `route_query`)
- Test: `tests/unit/servers/web/test_stage_emits_intent.py` (rewrite), `tests/unit/servers/web/test_agent_router.py` (update `test_classify_route_*` to unpack the tuple)

**Interfaces:**
- Consumes: existing `_regex_route`, `_rule_based_route`, `_capture.record_stage`, `RouteStrategy`, `_FakeLLM` (in `test_agent_router.py`).
- Produces: `classify_route(query, llm) -> tuple[RouteStrategy, dict]`; `_record_intent(mechanism: str, strategy: RouteStrategy, detail: dict) -> None`.

- [ ] **Step 1: Rewrite `test_stage_emits_intent.py` to drive `route_query` (failing)**

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
