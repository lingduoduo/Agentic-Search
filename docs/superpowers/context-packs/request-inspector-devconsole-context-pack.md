# Generated Context Pack

# Request Inspector (Dev Console) Implementation Plan

## Sources

- [Plan: 2026-07-05-request-inspector-devconsole.md](../plans/2026-07-05-request-inspector-devconsole.md)

## Implementation Plan Context

### Global Constraints

- Capture is active ONLY when `settings.debug_panels` is true (env `AGENTIC_SEARCH_DEBUG_PANELS`). When false, `active()` returns `None` and no snapshot is stored.
- Do NOT modify the sanitized `ControlFlowRecorder` / `_control_flow_event_view` path or its tests.
- Ring buffer default size `N=20`, overridable via env `AGENTIC_SEARCH_REQUEST_CAPTURE_MAX`.
- No persistence — snapshots are in-memory, cleared on restart.
- New debug endpoints live under the existing `/api/debug` router (already gated by `debug_panels`).
- Follow existing style; run `ruff check . --fix && ruff format .` before each commit. Frontend: `cd web && npm run typecheck`.
- All work on branch `feat/request-inspector-devconsole`.

---

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
  - `record_stage(stage: str, label: str, payload: dict, duration_ms: float | None = None) -> None`  (no-op if inactive)
  - `capture_stage(stage: str, label: str)` — context manager yielding a mutable `dict` payload; times the block; records on exit (no-op if inactive)
  - `active() -> RequestCapture | None`

- [ ] **Step 1: Write the failing tests**

```python

### tests/unit/servers/web/test_request_capture.py

from __future__ import annotations

from src.internal.servers.web import request_capture as rc


def test_record_stage_noops_when_inactive():
    # No active capture → record_stage is a silent no-op, active() is None.
    assert rc.active() is None
    rc.record_stage("intent", "classify_route", {"label": "chat"})
    assert rc.active() is None


def test_start_capture_records_and_snapshots():
    token = rc.start_capture("req-1", "what is faiss")
    try:
        cap = rc.active()
        assert cap is not None and cap.request_id == "req-1"
        rc.record_stage("intent", "classify_route", {"raw": "chat"}, duration_ms=12.0)
        rc.record_stage("final", "answer", {"answer": "hi"})
        cap.route = "chat"
        cap.finish()
        snap = cap.snapshot()
    finally:
        rc.reset_capture(token)
    assert rc.active() is None
    assert snap["request_id"] == "req-1"
    assert snap["query"] == "what is faiss"
    assert snap["route"] == "chat"
    assert [s["stage"] for s in snap["stages"]] == ["intent", "final"]
    assert snap["stages"][0]["payload"] == {"raw": "chat"}
    assert snap["stages"][0]["duration_ms"] == 12.0
    assert snap["total_ms"] is not None


def test_capture_stage_times_and_records():
    token = rc.start_capture("req-2", "q")
    try:
        with rc.capture_stage("llm", "complete") as payload:
            payload["model"] = "gpt-4o-mini"
            payload["completion"] = "ok"
        cap = rc.active()
        assert cap.stages[0].stage == "llm"
        assert cap.stages[0].payload == {"model": "gpt-4o-mini", "completion": "ok"}

_[Section compacted.]_

### Task 2: Ring buffer store + wire capture into the request lifecycle

**Files:**
- Create: `src/internal/servers/web/request_capture_store.py`
- Modify: `src/internal/servers/web/app.py` (`_run_agent_impl` at 1048; `run_agent` at 1321; `stream_agent._generate` done event at ~1430; lifespan/app init to create the store)
- Test: `tests/unit/servers/web/test_request_capture_store.py`

**Interfaces:**
- Consumes: `request_capture.start_capture`, `reset_capture`, `active` (Task 1).
- Produces:
  - `RequestCaptureStore(max_size: int = 20)` with `.put(snapshot: dict) -> None`, `.list() -> list[dict]` (newest first, summary fields only: `request_id, query, created_at, route, stage_count`), `.get(request_id: str) -> dict | None`
  - `_run_agent_impl(..., request_id: str)` — new required keyword arg
  - stream `done` event gains `"request_id"`

- [ ] **Step 1: Write the failing store test**

```python

### tests/unit/servers/web/test_request_capture_store.py

from __future__ import annotations

from src.internal.servers.web.request_capture_store import RequestCaptureStore


def _snap(rid: str) -> dict:
    return {
        "request_id": rid,
        "query": f"q-{rid}",
        "created_at": 1.0,
        "route": "chat",
        "stages": [{"stage": "intent", "label": "x", "timestamp": 0, "duration_ms": 1, "payload": {}}],
    }


def test_put_get_roundtrip():
    store = RequestCaptureStore(max_size=3)
    store.put(_snap("a"))
    assert store.get("a")["query"] == "q-a"
    assert store.get("missing") is None


def test_list_is_newest_first_with_summary_fields():
    store = RequestCaptureStore(max_size=3)
    store.put(_snap("a"))
    store.put(_snap("b"))
    listed = store.list()
    assert [r["request_id"] for r in listed] == ["b", "a"]
    assert listed[0] == {
        "request_id": "b", "query": "q-b", "created_at": 1.0,
        "route": "chat", "stage_count": 1,
    }


def test_evicts_beyond_max_size():
    store = RequestCaptureStore(max_size=2)
    store.put(_snap("a"))
    store.put(_snap("b"))
    store.put(_snap("c"))
    assert store.get("a") is None
    assert [r["request_id"] for r in store.list()] == ["c", "b"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/servers/web/test_request_capture_store.py -v`
Expected: FAIL with `ModuleNotFoundError: ... request_capture_store`

- [ ] **Step 3: Write the store**

```python

### Task 3: Debug endpoints `/api/debug/requests` and `/api/debug/request/{id}`

**Files:**
- Modify: `src/internal/servers/web/debug_router.py` (add two routes inside `create_debug_router`)
- Test: `tests/unit/servers/web/test_debug_request_endpoints.py`

**Interfaces:**
- Consumes: `app.state.request_captures` (a `RequestCaptureStore`, Task 2), accessed via the endpoint's `Request`.
- Produces: `GET /api/debug/requests -> {"requests": [...]}`; `GET /api/debug/request/{request_id} -> snapshot | 404`.

- [ ] **Step 1: Write the failing endpoint test**

```python

### tests/unit/servers/web/test_debug_request_endpoints.py

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.internal.servers.web.debug_router import create_debug_router
from src.internal.servers.web.request_capture_store import RequestCaptureStore


def _app_with_store(store: RequestCaptureStore) -> FastAPI:
    app = FastAPI()
    app.state.request_captures = store
    app.include_router(create_debug_router(search_url="http://x/retrieve"))
    return app


def _snap(rid: str) -> dict:
    return {
        "request_id": rid, "query": f"q-{rid}", "created_at": 1.0, "route": "chat",
        "route_degraded": None, "total_ms": 5.0,
        "stages": [{"stage": "intent", "label": "x", "timestamp": 0.0, "duration_ms": 1.0, "payload": {"raw": "chat"}}],
    }


def test_list_requests_newest_first():
    store = RequestCaptureStore()
    store.put(_snap("a"))
    store.put(_snap("b"))
    client = TestClient(_app_with_store(store))
    body = client.get("/api/debug/requests").json()
    assert [r["request_id"] for r in body["requests"]] == ["b", "a"]
    assert body["requests"][0]["stage_count"] == 1


def test_get_request_returns_full_snapshot():
    store = RequestCaptureStore()
    store.put(_snap("a"))
    client = TestClient(_app_with_store(store))
    body = client.get("/api/debug/request/a").json()
    assert body["stages"][0]["payload"] == {"raw": "chat"}


def test_get_missing_request_is_404():
    client = TestClient(_app_with_store(RequestCaptureStore()))
    assert client.get("/api/debug/request/nope").status_code == 404
```

_[Section compacted.]_

### Task 4: Intent + final stage emits

**Files:**
- Modify: `src/internal/servers/web/intent_routing.py` (`classify_route` at 150)
- Modify: `src/internal/servers/web/app.py` (`_run_auto_routed` at 750; `_finalize_response` at 696 — but emit from `_run_auto_routed`/`_run_agent_impl` where the capture context lives)
- Test: `tests/unit/servers/web/test_stage_emits_intent.py`

**Interfaces:**
- Consumes: `record_stage` (Task 1).
- Produces: an `intent` stage (label `classify_route`) and a `final` stage (label `answer`) in the active capture.

- [ ] **Step 1: Write the failing test**

```python

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
