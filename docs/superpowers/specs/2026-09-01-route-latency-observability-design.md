# Per-route latency observability

Date: 2026-09-01

## Problem

`src/internal/servers/middleware/latency_logging.py` defines
`add_latency_logging_middleware` and nothing calls it. The other four modules in
that directory — `license_enforcement`, `rate_limiting`, `tenant_tracking`,
`tier_gate` — are all registered in `create_web_app`. This one has zero call
sites, so the web backend measures no request latency at all.

#561 established what does exist: `GenerationTimings` measures
`llm_first_token_ms` and `time_to_first_claim_ms` *inside* `_generate_guarded_answer`,
and `/api/debug/requests` captures per-stage records for a single request. Both
answer "where did this one request spend its time". Neither answers "which route
is slow, and how often" — the question you ask before you know which request to
inspect.

## Design

### `RouteLatencyStats`

A bounded per-route sample store beside the middleware:

```python
class RouteLatencyStats:
    def __init__(self, max_samples_per_route: int = 512) -> None: ...
    def record(self, *, method: str, route: str, status_code: int, elapsed_ms: float) -> None
    def snapshot(self) -> list[dict]
```

`snapshot()` returns one row per `(method, route)`: `count`, `errors` (status
>= 500), `p50_ms`, `p95_ms`, `max_ms`, sorted by `p95_ms` descending so the
slowest route is first.

A `collections.deque(maxlen=N)` per route holds the most recent samples, so the
percentiles describe recent behaviour and memory is bounded by
`routes x max_samples`. There is no reset endpoint: the rolling window is what a
reset would be for.

### Keying by route template, not path

The middleware records `request.scope["route"].path_format` — the matched
template, e.g. `/api/session/{session_id}` — and falls back to
`request.url.path` when no route matched (404s, static files).

This is the decision the feature stands on. Keying on the raw path makes every
session id, request id and document id its own bucket; the table becomes
thousands of rows of one sample each within minutes, every percentile equals its
single sample, and the panel is worthless while still looking populated. A test
asserts the template is used.

### Registration

`add_latency_logging_middleware(app, logger)` in `create_web_app`, beside the
other three. Unconditional: the cost is one `time.monotonic()` pair and a deque
append per request, and the existing `logger.debug` line is already level-gated.
The default `RouteLatencyStats` instance is a module-level singleton, matching
how `request_capture` holds its state; the middleware takes an optional `stats`
argument so tests inject their own.

### Endpoint

`GET /api/debug/latency` on the existing debug router, so it stays behind
`AGENTIC_SEARCH_DEBUG_PANELS` with every other panel feed:

```json
{"routes": [{"method": "POST", "route": "/api/agent", "count": 12,
             "errors": 0, "p50_ms": 812.4, "p95_ms": 2140.9, "max_ms": 2210.0}]}
```

Only finite floats are emitted. A route with no samples is not listed at all,
so a percentile is never `null` in the payload — but the panel still handles a
missing value, because #437 shipped a crash from exactly that shape reaching
`toFixed`.

### Panel

`web/src/components/debug/LatencyPanel.tsx`, a route table added to
`DevConsole`, following `EvalResultsPanel`: fetch on mount, tolerate failure by
rendering the empty state, format numbers through a helper that returns a dash
for anything non-finite.

## Testing

- **The template, not the path.** Two requests to the same parameterised route
  with different path parameters produce one row with `count == 2`.
- `p50_ms` / `p95_ms` / `max_ms` over a known sample set.
- The window is bounded: recording more than `max_samples_per_route` keeps the
  newest and drops the oldest.
- A 500 response increments `errors`; a 200 does not.
- `create_web_app` registers the middleware — remove the call and this test goes
  red.
- The endpoint returns only finite numbers.
- Panel: renders rows, renders the empty state, and survives a row with a
  missing percentile.

## Out of scope

No reset endpoint, no persistence across restarts, no per-request timeline —
`/api/debug/requests` already gives the single-request view. This is the
aggregate that one cannot provide.
