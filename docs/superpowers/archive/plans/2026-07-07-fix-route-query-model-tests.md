# Hotfix: model-step route_query tests pass a removed param — plan

> Single mechanical test-only fix for a red `main` (a #383/#384 merge collision).

**Goal:** Remove the stale `has_local_model=` keyword from the 4 model-step `route_query` calls in `test_agent_router.py`.

## Global Constraints

- Test-only; no production code touched. Assertions unchanged.
- `ruff check --fix && ruff format` the file before commit.

---

### Task 1: Strip the removed kwarg from the model-step tests

**Files:**
- Modify: `tests/unit/servers/web/test_agent_router.py` (4 `route_query` calls)

- [ ] **Step 1:** Remove every `has_local_model=True|False,` from `route_query(...)` calls in the file.
- [ ] **Step 2:** `python -m pytest tests/unit/servers/web/test_agent_router.py -q` → 40 pass; `grep -n has_local_model tests/unit/servers/web/test_agent_router.py` → nothing.
- [ ] **Step 3:** `ruff check --fix && ruff format`; commit.

## Self-Review

- Root cause: #383 removed the param; #384's tests (built pre-#383) still passed it. Removing the kwarg reconciles them. No production change, so no behavior risk.
