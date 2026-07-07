# Hotfix: model-step route_query tests pass a removed param — design

## Problem

`main` is red. Two PRs merged independently and collided:

- **#383** removed the `has_local_model` parameter from `route_query`
  (`src/internal/servers/web/intent_routing.py`) and updated the tests it knew about.
- **#384** (built off a pre-#383 `main`) added new model-step tests to
  `tests/unit/servers/web/test_agent_router.py` that call
  `route_query(..., has_local_model=True/False, ...)`.

After both merged, those 4 model-step tests call a parameter that no longer
exists → `TypeError: route_query() got an unexpected keyword argument
'has_local_model'`. 4 failing tests on `main`.

## Goal

Green `main`: remove the stale `has_local_model=` keyword from the 4 model-step
`route_query` calls in `test_agent_router.py`. Test-only; no production change.

## Non-goals

- No change to `route_query`, `ml_intent`, or any production code.
- No change to the model-step tests' assertions — only the removed kwarg.

## Change

Delete `has_local_model=True`/`has_local_model=False` from the four
`route_query(...)` calls in `tests/unit/servers/web/test_agent_router.py`
(the model-step tests added by #384). The param is gone; the calls otherwise
stand.

## Testing

- `pytest tests/unit/servers/web/test_agent_router.py` → all pass (40).
- `grep has_local_model tests/unit/servers/web/test_agent_router.py` → nothing.

## Success criteria

- `main` test suite green again; no production behavior change.
