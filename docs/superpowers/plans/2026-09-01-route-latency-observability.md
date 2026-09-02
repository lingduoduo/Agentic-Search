# Plan: per-route latency observability

Spec: `docs/superpowers/specs/2026-09-01-route-latency-observability-design.md`

## Task 1 — `RouteLatencyStats`

1. Failing tests in `tests/unit/servers/test_latency_stats.py`: percentiles over
   a known sample set, bounded window, error counting, sort order.
2. Implement the class in
   `src/internal/servers/middleware/latency_logging.py`.
   → verify: tests pass.

## Task 2 — middleware records by route template

1. Failing test: two requests to `/echo/{item}` with different path parameters
   collapse to one row with `count == 2`; an unmatched path falls back to the
   raw path.
2. Extend `add_latency_logging_middleware` with the optional `stats` argument
   and the `request.scope["route"].path_format` lookup.
   → verify: tests pass; mutation — key on `request.url.path` and the collapse
   test goes red.

## Task 3 — register it

1. Failing test: an app from `create_web_app` has the latency middleware in its
   stack.
2. Call `add_latency_logging_middleware(app, logger)` in `create_web_app`.
   → verify: test passes; mutation — remove the call and it goes red.

## Task 4 — endpoint

1. Failing test: `GET /api/debug/latency` returns recorded routes, and every
   numeric field is finite.
2. Add the route to `create_debug_router`.
   → verify: test passes.

## Task 5 — panel

1. `web/src/components/debug/LatencyPanel.tsx` + `getRouteLatency` in
   `web/src/api.ts` + the `RouteLatencyRow` type.
2. Mount it in `DevConsole`.
3. Tests in `web/src/components/debug/__tests__/LatencyPanel.test.tsx`: rows,
   empty state, missing percentile.
   → verify: `npm test` and `npm run typecheck` pass.

## Task 6 — verification

1. `pytest` (full default suite).
2. `cd web && npm run typecheck && npm test`.
3. `ruff check . --fix && ruff format .`
4. Drive the real app: start the backend with `AGENTIC_SEARCH_DEBUG_PANELS=1`,
   issue a few requests, and read `/api/debug/latency`.
5. Mutation-check the route-template test and the registration test; clear stale
   `.pyc` before re-running.
