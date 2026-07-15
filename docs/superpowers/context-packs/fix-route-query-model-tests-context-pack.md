# Generated Context Pack

# Fix Route Query Model Tests

## Sources

- [Specification: 2026-07-07-fix-route-query-model-tests-design.md](../specs/2026-07-07-fix-route-query-model-tests-design.md)
- [Plan: 2026-07-07-fix-route-query-model-tests.md](../plans/2026-07-07-fix-route-query-model-tests.md)

## Specification Context

### Goal

Green `main`: remove the stale `has_local_model=` keyword from the 4 model-step
`route_query` calls in `test_agent_router.py`. Test-only; no production change.

## Implementation Plan Context

### Task 1: Strip the removed kwarg from the model-step tests

**Files:**
- Modify: `tests/unit/servers/web/test_agent_router.py` (4 `route_query` calls)

- [ ] **Step 1:** Remove every `has_local_model=True|False,` from `route_query(...)` calls in the file.
- [ ] **Step 2:** `python -m pytest tests/unit/servers/web/test_agent_router.py -q` → 40 pass; `grep -n has_local_model tests/unit/servers/web/test_agent_router.py` → nothing.
- [ ] **Step 3:** `ruff check --fix && ruff format`; commit.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
