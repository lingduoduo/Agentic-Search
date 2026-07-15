# Simplify intent routing: drop dead `has_local_model` param — design

## Problem

`route_query` (`src/internal/servers/web/intent_routing.py`) accepts a
`has_local_model: bool` keyword argument and then immediately discards it
(`del has_local_model  # dispatch layer handles capability degradation`). The
caller in `_run_auto_routed` computes `has_local_model` (still needed for its
own dispatch branches) and threads it into `route_query` for nothing. It is a
dead parameter: routing never depends on model capability — capability-aware
degradation happens later, at dispatch.

## Goal

Remove the unused parameter from `route_query` and its call argument. Pure
cleanup — **no behavior change**.

## Non-goals

- Do not touch the two-router design (`_regex_route` / `_rule_based_route`) —
  that is deliberate and its consolidation is a separate, behavior-changing
  decision.
- Do not change the `src/internal/routing/` retrieval subsystem (unrelated to
  intent routing).
- The local `has_local_model` variable in `_run_auto_routed` stays — it is
  still read by the TOOL/SEARCH/CHAT dispatch branches.

## Change

- `intent_routing.py`: drop `has_local_model` from the `route_query` signature,
  delete the `del has_local_model` line, and trim the docstring sentence that
  explained the accepted-but-unused param.
- `app.py`: remove `has_local_model=has_local_model` from the `route_query(...)`
  call (keep the local variable).
- Tests: strip the `has_local_model=...` keyword from every `route_query` call
  in `test_agent_router.py` and `test_stage_emits_intent.py`.

## Testing

- `test_agent_router.py`, `test_stage_emits_intent.py`, and
  `test_execution_fallbacks.py` pass unchanged in behavior; `app` imports; ruff
  clean.

## Success criteria

- `grep has_local_model src/internal/servers/web/intent_routing.py` returns
  nothing; the router test suites are green; no routing decision changes.
