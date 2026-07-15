# Simplify intent routing: drop dead `has_local_model` param — plan

> Single mechanical task; no TDD cycle (dead-param removal, behavior-preserving).

**Goal:** Remove the unused `has_local_model` parameter from `route_query` and every call site.

**Tech Stack:** Python 3.12, pytest, ruff.

## Global Constraints

- No behavior change: routing decisions must be identical before/after.
- Keep the local `has_local_model` variable in `_run_auto_routed` (used by dispatch branches); only the `route_query` param + arguments are removed.
- `ruff check --fix && ruff format` the touched files before commit.

---

### Task 1: Remove the parameter and update call sites

**Files:**
- Modify: `src/internal/servers/web/intent_routing.py` (`route_query` signature, docstring, `del` line)
- Modify: `src/internal/servers/web/app.py` (the `route_query(...)` call in `_run_auto_routed`)
- Modify: `tests/unit/servers/web/test_agent_router.py`, `tests/unit/servers/web/test_stage_emits_intent.py` (strip `has_local_model=...` from `route_query` calls)

- [ ] **Step 1: Edit source** — drop `has_local_model: bool` from the `route_query` signature, delete `del has_local_model`, trim the docstring sentence about it; remove `has_local_model=has_local_model` from the caller in `app.py` (keep the local variable).

- [ ] **Step 2: Sweep tests** — remove every `has_local_model=True|False` keyword argument from `route_query` calls in the two test files.

- [ ] **Step 3: Verify** —
  `grep -n has_local_model src/internal/servers/web/intent_routing.py` → no output.
  `python -c "import src.internal.servers.web.app"` → OK.
  `python -m pytest tests/unit/servers/web/test_agent_router.py tests/unit/servers/web/test_stage_emits_intent.py tests/unit/test_execution_fallbacks.py -q` → all pass.

- [ ] **Step 4: Commit** — `ruff check --fix && ruff format` the four files, then commit.

## Self-Review

- Spec coverage: signature/docstring/`del` (Task 1 Step 1), caller (Step 1), tests (Step 2), verification (Step 3). All covered.
- No behavior change: the removed param was `del`-ed unused; routing logic untouched.
