# Generated Context Pack

# Dispatch Consolidation

## Sources

- [Specification: 2026-06-29-dispatch-consolidation-design.md](../specs/2026-06-29-dispatch-consolidation-design.md)
- [Plan: 2026-06-29-dispatch-consolidation.md](../plans/2026-06-29-dispatch-consolidation.md)

## Specification Context

### Goal

One place builds and runs each agent loop; one place assembles the response.

### Tests

- `tests/unit/servers/web/test_loop_runners.py` — runner tuple/`extra` contracts.
- Existing web suite updated to the converged contract.

### Out of scope

- M10 `Router`-into-loop wiring (PR B); `route_query` / auto-search internals.

## Implementation Plan Context

### Tasks

1. **Runner tests (red).** New `tests/unit/servers/web/test_loop_runners.py`:
   assert each of `_run_search_agent`, `_run_agentic_rag`, `_run_tool_agent`
   returns `(answer, citations, documents, intent, extra)` with the right `extra`
   keys, driven by fake loops. → verify: tests fail (helpers absent).

2. **Extract runners (green).** Add `_run_search_agent`, `_run_agentic_rag`,
   `_run_tool_agent` to `app.py` per spec signatures. → verify: task-1 tests pass.

3. **Extract `_finalize_response` + rewire call sites.** Route `_run_auto_routed`
   loop branches and every explicit-mode branch (incl. `search_tool`,

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
