# Generated Context Pack

# Route Query Dead Param

## Sources

- [Specification: 2026-07-07-route-query-dead-param-design.md](../specs/2026-07-07-route-query-dead-param-design.md)
- [Plan: 2026-07-07-route-query-dead-param.md](../plans/2026-07-07-route-query-dead-param.md)

## Specification Context

### Goal

Remove the unused parameter from `route_query` and its call argument. Pure
cleanup — **no behavior change**.

### Non-goals

- Do not touch the two-router design (`_regex_route` / `_rule_based_route`) —
  that is deliberate and its consolidation is a separate, behavior-changing
  decision.
- Do not change the `src/internal/routing/` retrieval subsystem (unrelated to
  intent routing).
- The local `has_local_model` variable in `_run_auto_routed` stays — it is
  still read by the TOOL/SEARCH/CHAT dispatch branches.

### Testing

- `test_agent_router.py`, `test_stage_emits_intent.py`, and
  `test_execution_fallbacks.py` pass unchanged in behavior; `app` imports; ruff
  clean.

## Implementation Plan Context

### Global Constraints

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

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
