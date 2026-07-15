# Request Inspector (Dev Console) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture the full raw payload of every pipeline stage (intent, search, tool, llm, final) for a request and surface it as a one-request inspector in the debug-gated Dev Console, backed by a rolling in-memory history.

**Architecture:** Ambient capture via a `contextvars.ContextVar` holding a per-request `RequestCapture`. One-line `record_stage(...)` emits at existing choke points (router, agent loops, LLM provider, finalize) no-op instantly when no capture is active. Snapshots land in an in-memory ring buffer on `app.state`, read by two new `/api/debug/*` endpoints and rendered by a new `RequestInspector` React panel. Entirely separate from the sanitized `ControlFlowRecorder`.

**Tech Stack:** Python 3.12, FastAPI, pytest; React 19 + Vite + TypeScript (no component library).

## Global Constraints

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
# tests/unit/servers/web/test_request_capture.py
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
        assert cap.stages[0].duration_ms is not None
    finally:
        rc.reset_capture(token)


def test_capture_stage_noops_when_inactive():
    # Must not raise and must record nothing when no capture is active.
    with rc.capture_stage("llm", "complete") as payload:
        payload["model"] = "x"
    assert rc.active() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/servers/web/test_request_capture.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.internal.servers.web.request_capture'`

- [ ] **Step 3: Write the module**

```python
# src/internal/servers/web/request_capture.py
"""Ambient per-request capture of full raw stage payloads for the Dev Console.

A ``RequestCapture`` is stashed in a ContextVar for the duration of one
``/api/agent`` request when debug panels are enabled. Instrumentation points
call ``record_stage`` / ``capture_stage``, which no-op instantly when no capture
is active — so the hot path pays only one ContextVar ``.get()`` when the flag is
off. This is a SEPARATE channel from the sanitized ControlFlowRecorder: payloads
here are raw (full prompts, document bodies, completions) and never persisted.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, field
from time import monotonic

_current: contextvars.ContextVar["RequestCapture | None"] = contextvars.ContextVar(
    "request_capture", default=None
)


@dataclass
class StageRecord:
    stage: str
    label: str
    timestamp: float
    duration_ms: float | None
    payload: dict


@dataclass
class RequestCapture:
    request_id: str
    query: str
    created_at: float
    route: str | None = None
    route_degraded: str | None = None
    total_ms: float | None = None
    stages: list[StageRecord] = field(default_factory=list)
    _start: float = field(default_factory=monotonic, repr=False)

    def add(
        self, stage: str, label: str, payload: dict, duration_ms: float | None = None
    ) -> None:
        self.stages.append(
            StageRecord(
                stage=stage,
                label=label,
                timestamp=monotonic() - self._start,
                duration_ms=duration_ms,
                payload=payload,
            )
        )

    def finish(self) -> None:
        self.total_ms = (monotonic() - self._start) * 1000.0

    def snapshot(self) -> dict:
        return {
            "request_id": self.request_id,
            "query": self.query,
            "created_at": self.created_at,
            "route": self.route,
            "route_degraded": self.route_degraded,
            "total_ms": self.total_ms,
            "stages": [
                {
                    "stage": s.stage,
                    "label": s.label,
                    "timestamp": s.timestamp,
                    "duration_ms": s.duration_ms,
                    "payload": s.payload,
                }
                for s in self.stages
            ],
        }


def start_capture(request_id: str, query: str) -> contextvars.Token:
    """Begin a capture for this request; returns a token to reset with."""
    return _current.set(
        RequestCapture(request_id=request_id, query=query, created_at=monotonic())
    )


def reset_capture(token: contextvars.Token) -> None:
    _current.reset(token)


def active() -> "RequestCapture | None":
    return _current.get()


def record_stage(
    stage: str, label: str, payload: dict, duration_ms: float | None = None
) -> None:
    cap = _current.get()
    if cap is not None:
        cap.add(stage, label, payload, duration_ms)


@contextlib.contextmanager
def capture_stage(stage: str, label: str):
    """Time a block and record a stage from the payload dict the caller mutates.

    No-op (still yields a throwaway dict) when no capture is active.
    """
    cap = _current.get()
    if cap is None:
        yield {}
        return
    payload: dict = {}
    started = monotonic()
    try:
        yield payload
    finally:
        cap.add(stage, label, payload, (monotonic() - started) * 1000.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/servers/web/test_request_capture.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
ruff check src/internal/servers/web/request_capture.py tests/unit/servers/web/test_request_capture.py --fix && ruff format src/internal/servers/web/request_capture.py tests/unit/servers/web/test_request_capture.py
git add src/internal/servers/web/request_capture.py tests/unit/servers/web/test_request_capture.py
git commit -m "feat(devconsole): ambient per-request stage capture module"
```

---

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
# tests/unit/servers/web/test_request_capture_store.py
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
# src/internal/servers/web/request_capture_store.py
"""In-memory rolling store of request-capture snapshots for the Dev Console.

Bounded ring buffer keyed by request_id. Not persisted — cleared on restart.
"""

from __future__ import annotations

from collections import OrderedDict


class RequestCaptureStore:
    def __init__(self, max_size: int = 20) -> None:
        self._max = max(1, max_size)
        self._items: "OrderedDict[str, dict]" = OrderedDict()

    def put(self, snapshot: dict) -> None:
        rid = snapshot["request_id"]
        self._items[rid] = snapshot
        self._items.move_to_end(rid)
        while len(self._items) > self._max:
            self._items.popitem(last=False)

    def get(self, request_id: str) -> dict | None:
        return self._items.get(request_id)

    def list(self) -> list[dict]:
        out = [
            {
                "request_id": s["request_id"],
                "query": s["query"],
                "created_at": s["created_at"],
                "route": s.get("route"),
                "stage_count": len(s.get("stages", [])),
            }
            for s in self._items.values()
        ]
        out.reverse()  # newest first
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/servers/web/test_request_capture_store.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Wire the store + capture lifecycle into `app.py`**

In `create_web_app`, after `app = FastAPI(...)` (line ~971) and near `app.state.tool_approval_broker = ...` (line 972), add:

```python
    import os as _os
    from src.internal.servers.web.request_capture_store import RequestCaptureStore

    app.state.request_captures = RequestCaptureStore(
        max_size=int(_os.environ.get("AGENTIC_SEARCH_REQUEST_CAPTURE_MAX", "20"))
    )
```

Add the import near the other web imports at the top of `app.py`:

```python
from src.internal.servers.web import request_capture as _capture
```

Change `_run_agent_impl` signature (line 1048) to accept `request_id`:

```python
    async def _run_agent_impl(
        request: AgentExperienceRequest,
        http_request: Request,
        *,
        request_id: str,
        on_turn: "OnTurnCallback | None" = None,
        on_trace: EventSink | None = None,
        on_approval=None,
    ) -> AgentExperienceResponse:
```

Immediately after `query = request.query.strip()` and the empty check (line ~1057), start the capture and arrange teardown. Wrap the existing body: right before the existing `try:` at line 1110, insert:

```python
        capture_on = getattr(http_request.app.state, "request_captures", None) is not None
        capture_token = (
            _capture.start_capture(request_id, query)
            if capture_on and settings.debug_panels
            else None
        )
```

Then convert the function's tail so the snapshot is stored and the contextvar is always reset. Find the outer `try:` at line 1110 and its matching `except`/end; add a `finally` that runs after the response is built. Concretely, wrap the whole `try/except` block in an outer `try/finally`:

```python
        try:
            # ... existing try/except that returns _finalize_response(...) ...
        finally:
            cap = _capture.active()
            if cap is not None:
                cap.finish()
                http_request.app.state.request_captures.put(cap.snapshot())
            if capture_token is not None:
                _capture.reset_capture(capture_token)
```

In the non-streaming `run_agent` handler (line 1321) generate an id and pass it:

```python
        request_id = _uuid.uuid4().hex
        return await _run_agent_impl(request, http_request, request_id=request_id)
```

Add `import uuid as _uuid` at the top of `app.py` if not already present.

In `stream_agent._generate` (line 1407) generate the id before the task and thread it in:

```python
        request_id = _uuid.uuid4().hex
        async def _generate():
            task = asyncio.create_task(
                _run_agent_impl(
                    request,
                    http_request,
                    request_id=request_id,
                    on_turn=on_turn,
                    on_trace=on_trace,
                    on_approval=on_approval if auth_user is not None else None,
                )
            )
```

(Note: `request_id` must be defined in the `stream_agent` scope, above `_generate`, so it is in closure.)

Add `"request_id": request_id,` to the `done` event dict (line ~1430):

```python
                yield _sse(
                    {
                        "type": "done",
                        "request_id": request_id,
                        "session_id": result.session_id,
                        # ... existing fields ...
```

- [ ] **Step 6: Run the web app suite to verify no regressions**

Run: `python -m pytest tests/unit/servers/web/ -q`
Expected: PASS (all existing tests green; new store test green)

- [ ] **Step 7: Commit**

```bash
ruff check src/internal/servers/web/ tests/unit/servers/web/test_request_capture_store.py --fix && ruff format src/internal/servers/web/request_capture_store.py src/internal/servers/web/app.py tests/unit/servers/web/test_request_capture_store.py
git add src/internal/servers/web/request_capture_store.py src/internal/servers/web/app.py tests/unit/servers/web/test_request_capture_store.py
git commit -m "feat(devconsole): ring-buffer store + request-capture lifecycle wiring"
```

---

### Task 3: Debug endpoints `/api/debug/requests` and `/api/debug/request/{id}`

**Files:**
- Modify: `src/internal/servers/web/debug_router.py` (add two routes inside `create_debug_router`)
- Test: `tests/unit/servers/web/test_debug_request_endpoints.py`

**Interfaces:**
- Consumes: `app.state.request_captures` (a `RequestCaptureStore`, Task 2), accessed via the endpoint's `Request`.
- Produces: `GET /api/debug/requests -> {"requests": [...]}`; `GET /api/debug/request/{request_id} -> snapshot | 404`.

- [ ] **Step 1: Write the failing endpoint test**

```python
# tests/unit/servers/web/test_debug_request_endpoints.py
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

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/servers/web/test_debug_request_endpoints.py -v`
Expected: FAIL — 404 for `/api/debug/requests` (route not defined yet)

- [ ] **Step 3: Add the routes**

In `debug_router.py`, add `Request` to the fastapi import:

```python
from fastapi import APIRouter, Request, Response
```

Inside `create_debug_router`, before `return router` (line 147):

```python
    @router.get("/requests")
    def list_requests(request: Request) -> dict:
        """Summaries of recent captured runs (newest first). Empty when capture off."""
        store = getattr(request.app.state, "request_captures", None)
        return {"requests": store.list() if store is not None else []}

    @router.get("/request/{request_id}")
    def get_request(request_id: str, request: Request) -> Response:
        """Full raw stage snapshot for one run; 404 if evicted or capture off."""
        import json as _json

        store = getattr(request.app.state, "request_captures", None)
        snap = store.get(request_id) if store is not None else None
        if snap is None:
            return Response(
                content=f'{{"detail":"unknown request {request_id!r}"}}',
                status_code=404,
                media_type="application/json",
            )
        return Response(
            content=_json.dumps(snap), status_code=200, media_type="application/json"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/servers/web/test_debug_request_endpoints.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
ruff check src/internal/servers/web/debug_router.py tests/unit/servers/web/test_debug_request_endpoints.py --fix && ruff format src/internal/servers/web/debug_router.py tests/unit/servers/web/test_debug_request_endpoints.py
git add src/internal/servers/web/debug_router.py tests/unit/servers/web/test_debug_request_endpoints.py
git commit -m "feat(devconsole): /api/debug/requests + /request/{id} endpoints"
```

---

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
# tests/unit/servers/web/test_stage_emits_intent.py
from __future__ import annotations

from src.context.models import ChatMessage
from src.internal.servers.web import request_capture as rc
from src.internal.servers.web.intent_routing import classify_route


class _FakeLLM:
    def complete(self, messages: list[ChatMessage], **_) -> str:
        return "search"


def test_classify_route_emits_intent_stage_when_capturing():
    token = rc.start_capture("r", "vector database benchmarks")
    try:
        classify_route("vector database benchmarks", _FakeLLM())
        cap = rc.active()
        intent = [s for s in cap.stages if s.stage == "intent"]
        assert intent, "expected an intent stage"
        assert intent[0].payload["raw_label"] == "search"
        assert "prompt" in intent[0].payload
    finally:
        rc.reset_capture(token)


def test_classify_route_no_capture_does_not_raise():
    # With no active capture the emit is a silent no-op.
    classify_route("q", _FakeLLM())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/servers/web/test_stage_emits_intent.py -v`
Expected: FAIL — `KeyError: 'raw_label'` / no intent stage (emit not added yet)

- [ ] **Step 3: Add the emits**

In `intent_routing.py`, at the top add:

```python
from src.internal.servers.web import request_capture as _capture
```

In `classify_route`, after computing `content` (line ~165) and determining the strategy, record. Replace the body's tail so it records before returning. Simplest: wrap the return points by computing `strategy` into a variable, then emit once:

```python
    prompt = _ROUTE_PROMPT.format(user_query=query)
    response = llm.complete(
        [ChatMessage(role="user", content=prompt)], temperature=0.0
    )
    content = (
        (response if isinstance(response, str) else response.content).strip().lower()
    )
    strategy = RouteStrategy.CHAT
    if not content:
        logger.warning("Route classification empty; defaulting to chat.")
    else:
        for value, mapped in _LABEL_BY_VALUE.items():
            if re.search(rf"\b{value}\b", content):
                strategy = mapped
                break
        else:
            logger.warning(
                "Route classification returned unexpected response %r; defaulting to chat.",
                content,
            )
    _capture.record_stage(
        "intent", "classify_route", {"prompt": prompt, "raw_label": content, "strategy": strategy.value}
    )
    return strategy
```

In `app.py` `_run_auto_routed`, after `extra["route"] = strategy.value` (line 784) and after any degradation is set, record the resolved route onto the capture (so the snapshot's top-level `route`/`route_degraded` are populated). At the end of `_run_auto_routed`, before each `return`, this is noisy — instead set it once in `_run_agent_impl` after the auto call. In `_run_agent_impl`, right after the auto-routed call returns `extra` (line ~1128, before `return _finalize_response(...)`), add:

```python
                _cap = _capture.active()
                if _cap is not None:
                    _cap.route = extra.get("route")
                    _cap.route_degraded = extra.get("route_degraded")
```

For the **final** stage, in `_run_agent_impl` add a helper emit just before EACH `return _finalize_response(...)` is heavy; instead emit inside `_finalize_response`. But `_finalize_response` has no capture guard concerns — `record_stage` is a no-op when inactive, so it is safe to call unconditionally. In `_finalize_response` (line ~732), after building `metadata` and before `db.add_chat_message`, add:

```python
    _capture.record_stage(
        "final",
        "answer",
        {
            "answer": answer,
            "citations": citations,
            "documents": [d.id for d in documents],
            "intent": intent,
            "route": extra.get("route"),
            "route_degraded": extra.get("route_degraded"),
        },
    )
```

Ensure `app.py` imports `request_capture as _capture` (added in Task 2).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/servers/web/test_stage_emits_intent.py tests/unit/servers/web/test_agent_router.py -v`
Expected: PASS (existing router tests still green — they run with no active capture, so emits no-op)

- [ ] **Step 5: Commit**

```bash
ruff check src/internal/servers/web/intent_routing.py src/internal/servers/web/app.py tests/unit/servers/web/test_stage_emits_intent.py --fix && ruff format src/internal/servers/web/intent_routing.py src/internal/servers/web/app.py tests/unit/servers/web/test_stage_emits_intent.py
git add src/internal/servers/web/intent_routing.py src/internal/servers/web/app.py tests/unit/servers/web/test_stage_emits_intent.py
git commit -m "feat(devconsole): capture intent + final stages"
```

---

### Task 5: LLM stage emit (OpenAI provider + local manager)

**Files:**
- Modify: `src/internal/llm/providers.py` (`complete` at 227)
- Modify: `src/model/serving.py` (the local `manager.generate` entry — locate via grep)
- Test: `tests/unit/test_llm_providers.py` (add), `tests/unit/servers/web/test_stage_emits_llm.py`

**Interfaces:**
- Consumes: `record_stage` (Task 1).
- Produces: an `llm` stage (label `complete` or `generate`) with `{model, messages, completion, ...}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/servers/web/test_stage_emits_llm.py
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.internal.llm.interfaces import LLMConfig
from src.internal.llm.providers import OpenAICompatibleLLM
from src.internal.servers.web import request_capture as rc


def test_complete_emits_llm_stage_when_capturing():
    llm = OpenAICompatibleLLM(
        LLMConfig(model_provider="openai", model_name="gpt-4o-mini", api_key="sk")
    )
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": "hi there"}}]}
    mock_resp.raise_for_status.return_value = None

    token = rc.start_capture("r", "q")
    try:
        with patch.object(llm._session, "post", return_value=mock_resp):
            llm.complete([{"role": "user", "content": "hello"}])
        cap = rc.active()
        llm_stages = [s for s in cap.stages if s.stage == "llm"]
        assert llm_stages, "expected an llm stage"
        assert llm_stages[0].payload["completion"] == "hi there"
        assert llm_stages[0].payload["model"] == "gpt-4o-mini"
        assert llm_stages[0].payload["messages"] == [{"role": "user", "content": "hello"}]
    finally:
        rc.reset_capture(token)


def test_complete_no_capture_does_not_raise():
    llm = OpenAICompatibleLLM(
        LLMConfig(model_provider="openai", model_name="gpt-4o-mini", api_key="sk")
    )
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": "x"}}]}
    mock_resp.raise_for_status.return_value = None
    with patch.object(llm._session, "post", return_value=mock_resp):
        assert llm.complete([{"role": "user", "content": "hi"}]) == "x"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/servers/web/test_stage_emits_llm.py -v`
Expected: FAIL — no `llm` stage recorded

- [ ] **Step 3: Add the emit to `complete`**

In `providers.py` add near the top imports:

```python
from src.internal.servers.web import request_capture as _capture
```

In `complete` (line 227), replace the final `return`:

```python
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"] or ""
        _capture.record_stage(
            "llm",
            "complete",
            {
                "model": self._config.model_name,
                "messages": normalised,
                "completion": content,
                "usage": data.get("usage"),
            },
        )
        return content
```

For the local manager: `grep -n "def generate" src/model/serving.py` and, in the method that returns generated text, wrap the return the same way with `record_stage("llm", "generate", {"model": ..., "prompt": ..., "completion": ...})`. If the manager returns structured token output, record the decoded text and any available token counts. (Keep payload keys `model`, `messages` or `prompt`, `completion`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/servers/web/test_stage_emits_llm.py tests/unit/test_llm_providers.py -v`
Expected: PASS (existing provider tests green — they run without an active capture)

- [ ] **Step 5: Commit**

```bash
ruff check src/internal/llm/providers.py src/model/serving.py tests/unit/servers/web/test_stage_emits_llm.py --fix && ruff format src/internal/llm/providers.py src/model/serving.py tests/unit/servers/web/test_stage_emits_llm.py
git add src/internal/llm/providers.py src/model/serving.py tests/unit/servers/web/test_stage_emits_llm.py
git commit -m "feat(devconsole): capture llm stage (provider complete + local generate)"
```

---

### Task 6: Search + tool stage emits

**Files:**
- Modify: `src/agents/search/agentic_rag.py` (retrieval call — locate the retrieval invocation, near `_emit`/round loop)
- Modify: `src/agents/search/search.py` (search_tool invocation, ~915)
- Modify: `src/agents/core/base.py` (tool dispatch — locate where a tool is invoked and its result returned)
- Test: `tests/unit/servers/web/test_stage_emits_search_tool.py`

**Interfaces:**
- Consumes: `record_stage` (Task 1).
- Produces: `search` stages `{query, top_k, round, documents: [{id, title, text, score, source}]}`; `tool` stages `{name, args, result}`.

- [ ] **Step 1: Write the failing test**

Locate the smallest callable that performs retrieval in `AgenticRAGLoop` (e.g. a `_retrieve`/`_search` helper). Write a test that constructs the loop with a stub retrieval client returning two docs, calls that helper under an active capture, and asserts a `search` stage with the docs. Example shape (adjust to the real helper name/signature found by grep):

```python
# tests/unit/servers/web/test_stage_emits_search_tool.py
from __future__ import annotations

from src.internal.servers.web import request_capture as rc


def test_search_helper_emits_search_stage(faiss_stub_loop):
    # faiss_stub_loop: an AgenticRAGLoop wired with a stub retriever returning
    # two docs. See conftest for construction.
    token = rc.start_capture("r", "q")
    try:
        faiss_stub_loop._retrieve("dense retrieval", round_index=0)  # real helper name
        cap = rc.active()
        search = [s for s in cap.stages if s.stage == "search"]
        assert search and len(search[0].payload["documents"]) == 2
        assert search[0].payload["query"] == "dense retrieval"
    finally:
        rc.reset_capture(token)
```

If no clean helper exists, add a thin private `_record_search(query, top_k, round_index, docs)` method to the loop that only calls `record_stage`, call it from the existing retrieval site, and test that method directly. Do the same for the tool dispatch in `base.py` with a `_record_tool(name, args, result)` helper.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/servers/web/test_stage_emits_search_tool.py -v`
Expected: FAIL — no `search` stage

- [ ] **Step 3: Add the emits**

Add to each loop module:

```python
from src.internal.servers.web import request_capture as _capture
```

At the retrieval site in `agentic_rag.py` (right after documents come back for a round):

```python
        _capture.record_stage(
            "search",
            "retrieve",
            {
                "query": query,
                "top_k": top_k,
                "round": round_index,
                "documents": [
                    {"id": d.id, "title": getattr(d, "title", ""), "text": d.text,
                     "score": getattr(d, "score", None), "source": getattr(d, "source", None)}
                    for d in documents
                ],
            },
        )
```

At the `search_tool` site in `search.py` (~915), the same `record_stage("search", "search_tool", {...})` with the returned docs.

At the tool dispatch in `base.py`, after a tool returns its result:

```python
        _capture.record_stage("tool", tool_name, {"name": tool_name, "args": args, "result": result})
```

(Use the real local variable names for `documents`, `query`, `top_k`, `tool_name`, `args`, `result` at each site.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/servers/web/test_stage_emits_search_tool.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
ruff check src/agents/ tests/unit/servers/web/test_stage_emits_search_tool.py --fix && ruff format src/agents/search/agentic_rag.py src/agents/search/search.py src/agents/core/base.py tests/unit/servers/web/test_stage_emits_search_tool.py
git add src/agents/search/agentic_rag.py src/agents/search/search.py src/agents/core/base.py tests/unit/servers/web/test_stage_emits_search_tool.py
git commit -m "feat(devconsole): capture search + tool stages"
```

---

### Task 7: Frontend — API client, types, `RequestInspector` panel

**Files:**
- Modify: `web/src/types.ts` (add `RequestSummary`, `RequestSnapshot`, `StageRecordView`)
- Modify: `web/src/api.ts` (add `listDebugRequests`, `getDebugRequest`)
- Create: `web/src/components/debug/RequestInspector.tsx`
- Modify: `web/src/components/debug/DevConsole.tsx` (render `<RequestInspector />`)
- Modify: `web/src/App.tsx` (capture `request_id` from the `done` SSE event, pass as `selectedRequestId` to `DevConsole` → `RequestInspector`)

**Interfaces:**
- Consumes: `GET /api/debug/requests`, `GET /api/debug/request/{id}` (Task 3).
- Produces: a Dev Console panel rendering the 5 stages top-to-bottom for a selected run.

- [ ] **Step 1: Add types**

In `web/src/types.ts`:

```typescript
export interface StageRecordView {
  stage: "intent" | "search" | "tool" | "llm" | "final";
  label: string;
  timestamp: number;
  duration_ms: number | null;
  payload: Record<string, unknown>;
}

export interface RequestSummary {
  request_id: string;
  query: string;
  created_at: number;
  route: string | null;
  stage_count: number;
}

export interface RequestSnapshot {
  request_id: string;
  query: string;
  created_at: number;
  route: string | null;
  route_degraded: string | null;
  total_ms: number | null;
  stages: StageRecordView[];
}
```

- [ ] **Step 2: Add API client functions**

In `web/src/api.ts` (follow the existing `requestJson` helper used by `getServerHealth` at line ~63):

```typescript
export async function listDebugRequests(): Promise<{ requests: RequestSummary[] }> {
  return requestJson<{ requests: RequestSummary[] }>("/api/debug/requests");
}

export async function getDebugRequest(id: string): Promise<RequestSnapshot> {
  return requestJson<RequestSnapshot>(`/api/debug/request/${encodeURIComponent(id)}`);
}
```

Add the imports for `RequestSummary`, `RequestSnapshot` at the top of `api.ts`.

- [ ] **Step 3: Create `RequestInspector.tsx`**

```tsx
// web/src/components/debug/RequestInspector.tsx
import { useCallback, useEffect, useState } from "react";
import { getDebugRequest, listDebugRequests } from "../../api";
import type { RequestSnapshot, RequestSummary } from "../../types";

const STAGE_ORDER = ["intent", "search", "tool", "llm", "final"] as const;

interface Props {
  /** Auto-select the run that just finished streaming. */
  selectedRequestId?: string | null;
}

export function RequestInspector({ selectedRequestId }: Props) {
  const [runs, setRuns] = useState<RequestSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [snap, setSnap] = useState<RequestSnapshot | null>(null);

  const refresh = useCallback(async () => {
    try {
      const { requests } = await listDebugRequests();
      setRuns(requests);
    } catch {
      setRuns([]);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh, selectedRequestId]);

  useEffect(() => {
    const id = selectedRequestId ?? selected;
    if (!id) return;
    let cancelled = false;
    getDebugRequest(id)
      .then((s) => !cancelled && setSnap(s))
      .catch(() => !cancelled && setSnap(null));
    return () => {
      cancelled = true;
    };
  }, [selected, selectedRequestId]);

  const orderedStages = snap
    ? [...snap.stages].sort(
        (a, b) => STAGE_ORDER.indexOf(a.stage) - STAGE_ORDER.indexOf(b.stage),
      )
    : [];

  return (
    <div className="request-inspector" aria-label="Request inspector">
      <div className="request-inspector__list">
        <button type="button" onClick={() => void refresh()}>
          Refresh
        </button>
        <ul>
          {runs.map((r) => (
            <li key={r.request_id}>
              <button type="button" onClick={() => setSelected(r.request_id)}>
                <span>{r.query || "(empty)"}</span>
                <span>{r.route ?? "?"}</span>
                <span>{r.stage_count} stages</span>
              </button>
            </li>
          ))}
          {runs.length === 0 && <li>No captured runs (enable debug panels).</li>}
        </ul>
      </div>
      <div className="request-inspector__detail">
        {snap ? (
          orderedStages.map((s, i) => (
            <details key={`${s.stage}-${i}`} open>
              <summary>
                {s.stage} · {s.label}
                {s.duration_ms != null ? ` · ${s.duration_ms.toFixed(0)}ms` : ""}
              </summary>
              <pre>{JSON.stringify(s.payload, null, 2)}</pre>
            </details>
          ))
        ) : (
          <p>Select a run to inspect its stages.</p>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Wire into DevConsole + App**

In `DevConsole.tsx`, import and render, threading an optional selected id:

```tsx
import { RequestInspector } from "./RequestInspector";
// add to Props:
//   selectedRequestId?: string | null;
// inside the section, first child:
      <RequestInspector selectedRequestId={selectedRequestId} />
```

In `App.tsx`, in the `streamAgent` loop where the `"done"` event is handled (near `setControlFlowTrace` sort at ~168), capture `done.request_id` into state and pass it to `<DevConsole selectedRequestId={lastRequestId} ... />`:

```tsx
const [lastRequestId, setLastRequestId] = useState<string | null>(null);
// in the done branch:
if (event.type === "done") {
  setLastRequestId(event.request_id ?? null);
  // ... existing done handling ...
}
```

(Adjust to the actual SSE event typing in `App.tsx`; add `request_id?: string` to the done event type if the stream events are typed.)

- [ ] **Step 5: Typecheck**

Run: `cd web && npm run typecheck`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add web/src/types.ts web/src/api.ts web/src/components/debug/RequestInspector.tsx web/src/components/debug/DevConsole.tsx web/src/App.tsx
git commit -m "feat(devconsole): RequestInspector panel + API client"
```

---

### Task 8: End-to-end verification + docs

**Files:**
- Modify: `docs/superpowers/plans/2026-07-05-request-inspector-devconsole.md` (check off steps)
- Test: `tests/unit/servers/web/test_request_capture_e2e.py`

- [ ] **Step 1: Write an end-to-end capture test (flag on)**

Build a `create_web_app` TestClient with `debug_panels=True`, a stub retrieval + a fake LLM so `_run_agent_impl` completes without a model load (follow `tests/unit/servers/web/test_web_experience_app.py` patterns and `route_query` monkeypatching). POST `/api/agent`, then GET `/api/debug/requests` and `/api/debug/request/{id}`; assert the snapshot contains at least `intent` and `final` stages, and that with `debug_panels=False` the requests list is empty.

```python
# tests/unit/servers/web/test_request_capture_e2e.py — skeleton; fill using existing app test fixtures
def test_auto_routed_request_is_captured_when_flag_on(web_client_debug_on):
    r = web_client_debug_on.post("/api/agent", json={"query": "vector database"})
    assert r.status_code == 200
    listed = web_client_debug_on.get("/api/debug/requests").json()["requests"]
    assert listed, "expected a captured run"
    snap = web_client_debug_on.get(f"/api/debug/request/{listed[0]['request_id']}").json()
    stages = {s["stage"] for s in snap["stages"]}
    assert "final" in stages


def test_no_capture_when_flag_off(web_client_debug_off):
    web_client_debug_off.post("/api/agent", json={"query": "vector database"})
    assert web_client_debug_off.get("/api/debug/requests").json()["requests"] == []
```

- [ ] **Step 2: Run the full web suite**

Run: `python -m pytest tests/unit/servers/web/ -q`
Expected: PASS

- [ ] **Step 3: Full regression + lint**

Run: `python -m pytest -q && ruff check . && cd web && npm run typecheck`
Expected: all green.

- [ ] **Step 4: Manual smoke (optional, documented)**

Start the stack with `AGENTIC_SEARCH_DEBUG_PANELS=1` (backend) and `VITE_DEBUG_PANELS=1` (frontend), run a query, open the Console → Request Inspector, confirm the run shows intent/search/llm/final with raw payloads.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/servers/web/test_request_capture_e2e.py docs/superpowers/plans/2026-07-05-request-inspector-devconsole.md
git commit -m "test(devconsole): end-to-end request-capture coverage"
```

---

## Self-Review

**Spec coverage:**
- Ambient contextvar capture → Task 1. Ring buffer + lifecycle → Task 2. Endpoints → Task 3. Five stage emits → Tasks 4 (intent+final), 5 (llm), 6 (search+tool). Frontend one-request inspector → Task 7. Flag-gating + off-path zero-cost → Tasks 1/2/8. Separate from sanitized trace → enforced (no `ControlFlowRecorder` edits). Rolling in-memory history → Task 2 store. All spec sections covered.

**Placeholder scan:** Tasks 1–5, 7 contain complete code. Task 6's exact emit sites and Task 8's fixtures depend on real local variable/helper names discovered by grep at implementation time — the plan gives the exact `record_stage` payloads and a fallback (add a thin `_record_*` helper) so there is no ambiguity about WHAT to record, only WHERE, which the implementer resolves by reading the cited lines.

**Type consistency:** `RequestCapture`/`StageRecord` fields match across module (Task 1), store summary keys (Task 2), endpoint payloads (Task 3), and TS types (Task 7): `request_id, query, created_at, route, route_degraded, total_ms, stages[{stage,label,timestamp,duration_ms,payload}]`. Summary uses `stage_count` consistently in Task 2 store, Task 3 test, and Task 7 `RequestSummary`.
