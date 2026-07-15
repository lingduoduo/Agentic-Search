# Generated Context Pack

# Request Inspector (Dev Console) Implementation Plan

## Sources

- [Plan: 2026-07-05-request-inspector-devconsole.md](../plans/2026-07-05-request-inspector-devconsole.md)

## Implementation Plan Context

### Task 1: `request_capture` module (contextvar + data model)

**Files:**
- Create: `src/internal/servers/web/request_capture.py`
- Test: `tests/unit/servers/web/test_request_capture.py`

**Interfaces:**
- Produces:
  - `StageRecord(stage: str, label: str, timestamp: float, duration_ms: float | None, payload: dict)`
  - `RequestCapture(request_id: str, query: str, created_at: float, route: str | None, route_degraded: str | None, total_ms: float | None, stages: list[StageRecord])` with `.add(stage, label, payload, duration_ms=None)`, `.finish()`, `.snapshot() -> dict`
  - `start_capture(request_id: str, query: str) -> contextvars.Token`
  - `reset_capture(token: contextvars.Token) -> None`

…

### Task 2: Ring buffer store + wire capture into the request lifecycle

**Files:**
- Create: `src/internal/servers/web/request_capture_store.py`
- Modify: `src/internal/servers/web/app.py` (`_run_agent_impl` at 1048; `run_agent` at 1321; `stream_agent._generate` done event at ~1430; lifespan/app init to create the store)
- Test: `tests/unit/servers/web/test_request_capture_store.py`

**Interfaces:**
- Consumes: `request_capture.start_capture`, `reset_capture`, `active` (Task 1).
- Produces:
  - `RequestCaptureStore(max_size: int = 20)` with `.put(snapshot: dict) -> None`, `.list() -> list[dict]` (newest first, summary fields only: `request_id, query, created_at, route, stage_count`), `.get(request_id: str) -> dict | None`

…

### Task 3: Debug endpoints `/api/debug/requests` and `/api/debug/request/{id}`

**Files:**
- Modify: `src/internal/servers/web/debug_router.py` (add two routes inside `create_debug_router`)
- Test: `tests/unit/servers/web/test_debug_request_endpoints.py`

**Interfaces:**
- Consumes: `app.state.request_captures` (a `RequestCaptureStore`, Task 2), accessed via the endpoint's `Request`.
- Produces: `GET /api/debug/requests -> {"requests": [...]}`; `GET /api/debug/request/{request_id} -> snapshot | 404`.

- [ ] **Step 1: Write the failing endpoint test**

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/servers/web/test_debug_request_endpoints.py -v`
Expected: FAIL — 404 for `/api/debug/requests` (route not defined yet)

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
