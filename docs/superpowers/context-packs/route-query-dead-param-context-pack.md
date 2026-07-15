# Generated Context Pack

# Route Query Dead Param

## Sources

- [Specification: 2026-07-07-route-query-dead-param-design.md](../archive/specs/2026-07-07-route-query-dead-param-design.md)
- [Plan: 2026-07-07-route-query-dead-param.md](../archive/plans/2026-07-07-route-query-dead-param.md)

## Specification Context

### Goal

Remove the unused parameter from `route_query` and its call argument. Pure
cleanup — **no behavior change**.

## Implementation Plan Context

### Task 1: Remove the parameter and update call sites

**Files:**
- Modify: `src/internal/servers/web/intent_routing.py` (`route_query` signature, docstring, `del` line)
- Modify: `src/internal/servers/web/app.py` (the `route_query(...)` call in `_run_auto_routed`)
- Modify: `tests/unit/servers/web/test_agent_router.py`, `tests/unit/servers/web/test_stage_emits_intent.py` (strip `has_local_model=...` from `route_query` calls)

- [ ] **Step 1: Edit source** — drop `has_local_model: bool` from the `route_query` signature, delete `del has_local_model`, trim the docstring sentence about it; remove `has_local_model=has_local_model` from the caller in `app.py` (keep the local variable).

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
