# Tool-Agent Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the tool engine its own `/tool/*` API (a streaming conversational tool-agent endpoint + history) and its own frontend surface (a dedicated "Tool Agent" tab), parallel to search and chat.

**Architecture:** A thin `/tool/*` router reuses the existing `_run_tool_agent` loop runner (relocated out of `app.py` into a neutral module to avoid a circular import). The endpoint streams live progress via the loop's `on_turn` hook and emits the full tool-call trace + answer at completion, using SSE framing identical to `/api/agent/stream`. The frontend adds a top-level Assistant | Tool Agent switcher; the Tool Agent view drives the new endpoint with a trace-first layout.

**Tech Stack:** Python 3, FastAPI, Pydantic, pytest (backend); React 19, TypeScript, Vitest, React Testing Library (frontend).

## Global Constraints

- Never commit to `main`; work on branch `feat/tool-agent-surface` (already created).
- The tool-agent loop **requires a local model**. When `app.state.search_agent_manager` or `search_agent_tokenizer` is `None`, endpoints that run the loop return **HTTP 400** with the exact message: `"tool_agent mode requires a local model. Set SEARCH_AGENT_MODEL or SEARCH_AGENT_SERVER_URL in .env and restart."`
- Streaming uses **SSE framing** (`media_type="text/event-stream"`, each line `data: <json>\n\n`) to reuse the existing `streamAgent` parser. (Refines the spec's "NDJSON" mention — chosen for frontend-parser reuse.)
- New router mounts under prefix `/tool`, registered in `_register_routers` next to `create_search_router`.
- The new frontend tab is named **"Tool Agent"** (the existing header "Tools" wrench button opens the Manage-tools admin panel — do not reuse that label).
- Backend tests must NOT trigger model loading — mount the router on a bare `FastAPI()` in tests and inject fake `app.state` model handles; never spin up the full `create_web_app` lifespan.
- Run `ruff check . --fix && ruff format .` before each backend commit; `cd web && npm run typecheck` before each frontend commit. Pre-commit runs ruff-format and will abort the commit if it reformats — re-add and re-commit if so.

---

## File Structure

New:
- `src/internal/servers/web/tool_agent_runner.py` — relocated `ToolCallView`, `_extract_tool_calls_and_docs`, `_run_tool_agent`.
- `src/internal/servers/query_and_chat/tool_backend.py` — `create_tool_router`.
- `tests/unit/test_tool_backend.py` — backend router tests.
- `web/src/components/ToolAgentView.tsx` — the Tool Agent surface.
- `web/src/components/__tests__/ToolAgentView.test.tsx` — frontend tests.

Modified:
- `src/internal/servers/web/app.py` — import relocated symbols; register the tool router.
- `src/internal/servers/query_and_chat/models.py` — request/response models for `/tool/*`.
- `web/src/api.ts` — `sendToolMessage` generator + `getToolHistory`.
- `web/src/types.ts` — tool packet + history types.
- `web/src/App.tsx` — Assistant | Tool Agent view switcher.
- `docs/tool-engine.md` — document the new `/tool/*` surface.

---

## Task 1: Relocate the loop runner into a neutral module

Pure relocation so `tool_backend.py` (in `query_and_chat/`) can reuse the runner without importing `app.py` (which imports the router → cycle). No logic changes.

**Files:**
- Create: `src/internal/servers/web/tool_agent_runner.py`
- Modify: `src/internal/servers/web/app.py` (remove the three definitions; import them)
- Test: `tests/unit/test_tool_agent_runner.py`

**Interfaces:**
- Produces:
  - `ToolCallView(BaseModel)` — fields `tool_name: str`, `status: str`, `arguments: dict[str, object]`, `result_summary: str`, `latency_ms: int`, `error: str | None = None`.
  - `_extract_tool_calls_and_docs(output) -> tuple[list[ToolCallView], list]`
  - `async _run_tool_agent(query, *, manager, tokenizer, search_url, history, resolved, on_turn=None, on_approval=None, with_search_tool) -> tuple[str, list, list, str, dict]`

- [ ] **Step 1: Create the new module with the three symbols moved verbatim**

Create `src/internal/servers/web/tool_agent_runner.py`. Copy the bodies of `ToolCallView` (app.py:231–238), `_extract_tool_calls_and_docs` (app.py:414–469), and `_run_tool_agent` (app.py:728–796) **unchanged**, with these module-level imports:

```python
"""Runner for the ToolAgentLoop, shared by the /api/agent path and /tool/* router.

Relocated out of app.py so query_and_chat routers can reuse it without importing
app.py (which imports those routers — that would be circular).
"""
from __future__ import annotations

import json as _json

from pydantic import BaseModel

from src.context.models import ContextDocument
from .intent_routing import _infer_intent_from_output


class ToolCallView(BaseModel):
    tool_name: str
    status: str
    arguments: dict[str, object]
    result_summary: str
    latency_ms: int
    error: str | None = None


def _extract_tool_calls_and_docs(output) -> tuple[list[ToolCallView], list]:
    # ... verbatim body from app.py:415-469 ...


async def _run_tool_agent(
    query: str,
    *,
    manager,
    tokenizer,
    search_url: str,
    history: list,
    resolved,
    on_turn=None,
    on_approval=None,
    with_search_tool: bool,
) -> tuple:
    # ... verbatim body from app.py:740-796 ...
```

- [ ] **Step 2: Remove the moved definitions from app.py and import them**

Delete the three definitions from `app.py`. Add to app.py's import block (near the other `from .intent_routing import` / `from .static import` lines):

```python
from .tool_agent_runner import (
    ToolCallView,
    _extract_tool_calls_and_docs,
    _run_tool_agent,
)
```

Leave every existing *use* of these three symbols in app.py untouched (e.g. `AgentExperienceResponse.tool_calls`, the `mode == "tool_agent"` path). `_infer_intent_from_output` stays imported in app.py from `.intent_routing` as before.

- [ ] **Step 3: Write the relocation regression test**

Create `tests/unit/test_tool_agent_runner.py`:

```python
def test_symbols_importable_from_new_module():
    from src.internal.servers.web.tool_agent_runner import (
        ToolCallView,
        _extract_tool_calls_and_docs,
        _run_tool_agent,
    )
    assert ToolCallView.__name__ == "ToolCallView"
    assert callable(_extract_tool_calls_and_docs)
    assert callable(_run_tool_agent)


def test_app_reexports_same_objects():
    # app.py must import (not redefine) the relocated symbols.
    from src.internal.servers.web import app, tool_agent_runner
    assert app.ToolCallView is tool_agent_runner.ToolCallView
    assert app._run_tool_agent is tool_agent_runner._run_tool_agent


def test_extract_empty_trace_returns_empty():
    from src.internal.servers.web.tool_agent_runner import _extract_tool_calls_and_docs

    class _Out:
        action_trace = ""

    calls, docs = _extract_tool_calls_and_docs(_Out())
    assert calls == [] and docs == []
```

- [ ] **Step 4: Run tests + existing tool-agent suite**

Run: `pytest tests/unit/test_tool_agent_runner.py -v && pytest tests/unit -k "tool_agent or tool_calling" -q`
Expected: all PASS (relocation preserved behavior).

- [ ] **Step 5: Lint + commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/servers/web/tool_agent_runner.py src/internal/servers/web/app.py tests/unit/test_tool_agent_runner.py
git commit -m "refactor: relocate ToolAgentLoop runner into tool_agent_runner.py"
```

---

## Task 2: Backend `/tool/*` router

**Files:**
- Modify: `src/internal/servers/query_and_chat/models.py` (append request/response models)
- Create: `src/internal/servers/query_and_chat/tool_backend.py`
- Modify: `src/internal/servers/web/app.py` (register the router in `_register_routers`)
- Test: `tests/unit/test_tool_backend.py`

**Interfaces:**
- Consumes: `_run_tool_agent`, `ToolCallView` from `src.internal.servers.web.tool_agent_runner` (Task 1); `resolve_request_user` from `src.internal.servers.users.api`; `AgenticSearchStore` methods `get_chat_session`, `create_chat_session(user_id, title, metadata, session_id)`, `list_chat_messages`, `add_chat_message`, `list_sessions_for_user`.
- Produces: `create_tool_router(store, *, search_url="http://localhost:8000/retrieve", resolved) -> APIRouter` mounting `POST /tool/send-tool-message` and `GET /tool/tool-history`.

- [ ] **Step 1: Add the models**

Append to `src/internal/servers/query_and_chat/models.py` (it already imports `BaseModel`, `Field`, and uses `datetime`; add `from datetime import datetime` if absent):

```python
class SendToolMessageRequest(BaseModel):
    session_id: str | None = None
    message: str
    run_search_tool: bool = True
    stream: bool = True


class ToolAgentMessageResponse(BaseModel):
    session_id: str
    answer: str
    tool_calls: list[dict] = Field(default_factory=list)
    num_turns: int = 0
    error: str | None = None


class ToolSessionSummary(BaseModel):
    session_id: str
    title: str
    created_at: datetime


class ToolHistoryResponse(BaseModel):
    sessions: list[ToolSessionSummary] = Field(default_factory=list)
```

- [ ] **Step 2: Write the router's failing test (no-model 400 + history)**

Create `tests/unit/test_tool_backend.py`. This mounts ONLY the router on a bare app, injecting fake model handles into `app.state` — no lifespan, no model load.

```python
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.db import AgenticSearchStore
from src.internal.configs import load_app_settings
from src.internal.servers.query_and_chat.tool_backend import create_tool_router


def _make_app(*, with_model: bool) -> FastAPI:
    store = AgenticSearchStore(":memory:")
    app = FastAPI()
    app.include_router(
        create_tool_router(store, search_url="http://x/retrieve", resolved=load_app_settings())
    )
    app.state.search_agent_manager = object() if with_model else None
    app.state.search_agent_tokenizer = object() if with_model else None
    app.state.tool_approval_broker = None
    app.state._store = store  # keep a handle for assertions
    return app


def test_send_tool_message_no_model_returns_400():
    client = TestClient(_make_app(with_model=False))
    resp = client.post("/tool/send-tool-message", json={"message": "hi", "stream": False})
    assert resp.status_code == 400
    assert "requires a local model" in resp.json()["detail"]


def test_tool_history_anonymous_is_empty():
    client = TestClient(_make_app(with_model=True))
    resp = client.get("/tool/tool-history")
    assert resp.status_code == 200
    assert resp.json() == {"sessions": []}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/unit/test_tool_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: ... tool_backend` (router not created yet).

- [ ] **Step 4: Implement the router**

Create `src/internal/servers/query_and_chat/tool_backend.py`:

```python
"""Tool-agent API router — the tool engine's own conversational surface.

Parallels search_backend/chat_backend. Endpoints:
  POST /tool/send-tool-message  — run ToolAgentLoop, stream progress + tool calls
  GET  /tool/tool-history       — past sessions for the caller (session proxy)
"""
from __future__ import annotations

import asyncio
import json as _json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.internal.db import AgenticSearchStore
from src.internal.llm.interfaces import ChatMessage
from src.internal.servers.query_and_chat.models import (
    SendToolMessageRequest,
    ToolAgentMessageResponse,
    ToolHistoryResponse,
    ToolSessionSummary,
)
from src.internal.servers.users.api import resolve_request_user
from src.internal.servers.web.tool_agent_runner import _run_tool_agent

logger = logging.getLogger(__name__)

_NO_MODEL_MSG = (
    "tool_agent mode requires a local model. "
    "Set SEARCH_AGENT_MODEL or SEARCH_AGENT_SERVER_URL in .env and restart."
)
_MAX_HISTORY_MESSAGES = 40


def create_tool_router(
    store: AgenticSearchStore,
    *,
    search_url: str = "http://localhost:8000/retrieve",
    resolved,
) -> APIRouter:
    router = APIRouter(prefix="/tool", tags=["tool"])

    def _model_backend(request: Request):
        manager = getattr(request.app.state, "search_agent_manager", None)
        tokenizer = getattr(request.app.state, "search_agent_tokenizer", None)
        return manager, tokenizer

    def _ensure_session(body: SendToolMessageRequest, user_id: str | None) -> str:
        if body.session_id and store.get_chat_session(body.session_id):
            return body.session_id
        session = store.create_chat_session(
            user_id=user_id,
            title=body.message[:80],
            metadata={"source": "tool"},
            session_id=body.session_id,
        )
        return session.id

    def _history(session_id: str) -> list[ChatMessage]:
        msgs = [
            ChatMessage(role=m.role, content=m.content)
            for m in store.list_chat_messages(session_id)
        ]
        return msgs[-_MAX_HISTORY_MESSAGES:]

    @router.post("/send-tool-message", response_model=None)
    async def send_tool_message(body: SendToolMessageRequest, http_request: Request):
        manager, tokenizer = _model_backend(http_request)
        if manager is None or tokenizer is None:
            raise HTTPException(status_code=400, detail=_NO_MODEL_MSG)

        user = resolve_request_user(http_request)
        user_id = user.id if user and not user.is_anonymous else None
        session_id = _ensure_session(body, user_id)
        history = _history(session_id)
        store.add_chat_message(session_id, role="user", content=body.message)

        async def _run(on_turn=None):
            answer, _citations, documents, _intent, extra = await _run_tool_agent(
                body.message,
                manager=manager,
                tokenizer=tokenizer,
                search_url=search_url,
                history=history,
                resolved=resolved,
                on_turn=on_turn,
                on_approval=None,
                with_search_tool=body.run_search_tool,
            )
            answer = answer or extra.pop("_assistant_fallback", "")
            tool_calls = extra.get("tool_calls", [])
            return answer, tool_calls, extra.get("num_turns", 0)

        if not body.stream:
            try:
                answer, tool_calls, num_turns = await _run()
                store.add_chat_message(session_id, role="assistant", content=answer)
                return ToolAgentMessageResponse(
                    session_id=session_id,
                    answer=answer,
                    tool_calls=[tc.model_dump() for tc in tool_calls],
                    num_turns=num_turns,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Tool agent failed for: %r", body.message)
                return ToolAgentMessageResponse(
                    session_id=session_id, answer="", error=str(exc)
                )

        async def _gen() -> AsyncGenerator[str, None]:
            def _sse(data: dict) -> str:
                return f"data: {_json.dumps(data)}\n\n"

            queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=100)

            async def on_turn(turn: int, tool_name, doc_count: int) -> None:
                text = f"{tool_name} · {doc_count} docs" if tool_name else "writing answer..."
                await queue.put({"type": "progress", "turn": turn, "text": text})

            task = asyncio.create_task(_run(on_turn=on_turn))
            try:
                while not task.done():
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=0.05)
                        yield _sse(item)
                    except asyncio.TimeoutError:
                        continue
                while not queue.empty():
                    yield _sse(queue.get_nowait())

                answer, tool_calls, num_turns = task.result()
                store.add_chat_message(session_id, role="assistant", content=answer)
                for tc in tool_calls:
                    yield _sse({"type": "tool_call", **tc.model_dump()})
                yield _sse({"type": "answer", "text": answer})
                yield _sse(
                    {
                        "type": "done",
                        "session_id": session_id,
                        "tool_calls": [tc.model_dump() for tc in tool_calls],
                        "num_turns": num_turns,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Streaming tool agent failed for: %r", body.message)
                yield _sse({"type": "error", "detail": str(exc)})

        return StreamingResponse(_gen(), media_type="text/event-stream")

    @router.get("/tool-history")
    def tool_history(
        limit: int = 100,
        filter_days: int | None = None,
        http_request: Request = None,
    ) -> ToolHistoryResponse:
        if limit <= 0 or limit > 1000:
            raise HTTPException(status_code=400, detail="limit must be between 1 and 1000")
        if filter_days is not None and filter_days <= 0:
            raise HTTPException(status_code=400, detail="filter_days must be > 0")

        user = resolve_request_user(http_request) if http_request else None
        if user is None or user.is_anonymous:
            return ToolHistoryResponse(sessions=[])

        sessions = store.list_sessions_for_user(
            user.id, limit=limit, filter_days=filter_days
        )
        return ToolHistoryResponse(
            sessions=[
                ToolSessionSummary(
                    session_id=s.id, title=s.title or s.id, created_at=s.created_at
                )
                for s in sessions
                if s.created_at
            ]
        )

    return router


__all__ = ["create_tool_router"]
```

- [ ] **Step 5: Run the two tests to confirm they pass**

Run: `pytest tests/unit/test_tool_backend.py -v`
Expected: both PASS.

- [ ] **Step 6: Add the streaming happy-path test (monkeypatched loop)**

Append to `tests/unit/test_tool_backend.py`:

```python
def test_send_tool_message_streams_progress_then_done(monkeypatch):
    from src.internal.servers.query_and_chat import tool_backend
    from src.internal.servers.web.tool_agent_runner import ToolCallView

    async def fake_run_tool_agent(query, *, on_turn=None, **kw):
        if on_turn is not None:
            await on_turn(1, "search", 3)
        tc = ToolCallView(
            tool_name="search", status="completed", arguments={"q": query},
            result_summary="3 items", latency_ms=12, error=None,
        )
        return ("the answer", [], [], "tool", {"tool_calls": [tc], "num_turns": 1})

    monkeypatch.setattr(tool_backend, "_run_tool_agent", fake_run_tool_agent)

    client = TestClient(_make_app(with_model=True))
    with client.stream(
        "POST", "/tool/send-tool-message", json={"message": "find X", "stream": True}
    ) as resp:
        assert resp.status_code == 200
        events = [
            json.loads(line[len("data:"):].strip())
            for line in resp.iter_lines()
            if line.startswith("data:")
        ]

    types = [e["type"] for e in events]
    assert "progress" in types
    assert types[-1] == "done"
    tool_call = next(e for e in events if e["type"] == "tool_call")
    assert tool_call["tool_name"] == "search"
    done = events[-1]
    assert done["num_turns"] == 1 and done["session_id"]
```

- [ ] **Step 7: Run the streaming test**

Run: `pytest tests/unit/test_tool_backend.py::test_send_tool_message_streams_progress_then_done -v`
Expected: PASS.

- [ ] **Step 8: Register the router in app.py**

In `src/internal/servers/web/app.py`, inside `_register_routers` (right after the `create_search_router` line, ~363):

```python
        from src.internal.servers.query_and_chat.tool_backend import create_tool_router

        app.include_router(
            create_tool_router(db, search_url=search_url, resolved=settings)
        )
```

(`settings` in `_register_routers` is the resolved `AppSettings`.)

- [ ] **Step 9: Run the web app import smoke test**

Run: `pytest tests/unit -k "web_app or create_web_app or routers" -q`
Expected: PASS (app builds with the new router). If no such test exists, run: `python -c "from src.internal.servers.web.app import create_web_app; create_web_app()"` and expect no import error.

- [ ] **Step 10: Lint + commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/servers/query_and_chat/models.py src/internal/servers/query_and_chat/tool_backend.py src/internal/servers/web/app.py tests/unit/test_tool_backend.py
git commit -m "feat: add /tool/* router (streaming tool-agent endpoint + history)"
```

---

## Task 3: Frontend API layer + types

**Files:**
- Modify: `web/src/types.ts` (packet + history types)
- Modify: `web/src/api.ts` (`sendToolMessage` generator + `getToolHistory`)
- Test: `web/src/components/__tests__/toolApi.test.ts`

**Interfaces:**
- Produces:
  - `type ToolStreamEvent` — a discriminated union on `type`: `progress` `{turn:number;text:string}`, `tool_call` (`ToolCallTraceView` + `type`), `answer` `{text:string}`, `done` `{session_id:string;tool_calls:ToolCallTraceView[];num_turns:number}`, `error` `{detail:string}`.
  - `async function* sendToolMessage(body, init?): AsyncGenerator<ToolStreamEvent>`
  - `function getToolHistory(): Promise<{sessions: ToolSessionSummary[]}>`

- [ ] **Step 1: Add types**

Append to `web/src/types.ts`:

```typescript
export interface ToolSessionSummary {
  session_id: string;
  title: string;
  created_at: string;
}

export type ToolStreamEvent =
  | { type: "progress"; turn: number; text: string }
  | ({ type: "tool_call" } & ToolCallTraceView)
  | { type: "answer"; text: string }
  | { type: "done"; session_id: string; tool_calls: ToolCallTraceView[]; num_turns: number }
  | { type: "error"; detail: string };

export interface SendToolMessageBody {
  session_id?: string;
  message: string;
  run_search_tool?: boolean;
  stream?: boolean;
}
```

- [ ] **Step 2: Add the API functions**

Append to `web/src/api.ts` (mirror the existing `streamAgent` SSE-parsing loop at api.ts:206–244):

```typescript
export async function* sendToolMessage(
  body: SendToolMessageBody,
  init?: Pick<RequestInit, "signal">,
): AsyncGenerator<ToolStreamEvent> {
  const response = await fetch("/tool/send-tool-message", {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stream: true, ...body }),
    signal: init?.signal,
  });
  if (response.status === 400) {
    throw new Error("NO_LOCAL_MODEL");
  }
  if (!response.ok || !response.body) {
    throw new Error(`Tool stream failed: ${response.status}`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith("data:")) {
        const payload = trimmed.slice("data:".length).trim();
        if (payload) yield JSON.parse(payload) as ToolStreamEvent;
      }
    }
  }
}

export function getToolHistory(): Promise<{ sessions: ToolSessionSummary[] }> {
  return requestJson<{ sessions: ToolSessionSummary[] }>("/tool/tool-history");
}
```

Add `SendToolMessageBody`, `ToolStreamEvent`, `ToolSessionSummary` to the existing type import from `./types` at the top of `api.ts`.

- [ ] **Step 3: Write a parser test**

Create `web/src/components/__tests__/toolApi.test.ts`:

```typescript
import { describe, expect, it, vi } from "vitest";
import { sendToolMessage } from "../../api";

function sseStream(chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(enc.encode(c));
      controller.close();
    },
  });
}

describe("sendToolMessage", () => {
  it("parses SSE events in order", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        body: sseStream([
          'data: {"type":"progress","turn":1,"text":"search · 3 docs"}\n\n',
          'data: {"type":"answer","text":"hi"}\n\n',
          'data: {"type":"done","session_id":"s1","tool_calls":[],"num_turns":1}\n\n',
        ]),
      }),
    );
    const types: string[] = [];
    for await (const e of sendToolMessage({ message: "q" })) types.push(e.type);
    expect(types).toEqual(["progress", "answer", "done"]);
  });

  it("throws NO_LOCAL_MODEL on 400", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 400 }));
    await expect(async () => {
      for await (const _ of sendToolMessage({ message: "q" })) void _;
    }).rejects.toThrow("NO_LOCAL_MODEL");
  });
});
```

- [ ] **Step 4: Run the test + typecheck**

Run: `cd web && npx vitest run src/components/__tests__/toolApi.test.ts && npm run typecheck`
Expected: PASS, no type errors.

- [ ] **Step 5: Commit**

```bash
cd web && npm run typecheck
git add web/src/types.ts web/src/api.ts web/src/components/__tests__/toolApi.test.ts
git commit -m "feat(web): sendToolMessage stream client + getToolHistory + types"
```

---

## Task 4: Tool Agent view + header switcher

**Files:**
- Create: `web/src/components/ToolAgentView.tsx`
- Modify: `web/src/App.tsx` (Assistant | Tool Agent switcher)
- Test: `web/src/components/__tests__/ToolAgentView.test.tsx`

**Interfaces:**
- Consumes: `sendToolMessage` (Task 3), existing `ToolCallTracePanel`, `ToolCallTraceView`.
- Produces: `ToolAgentView` (default export or named) — a self-contained surface with its own composer, live tool-call trace, and a no-model banner.

- [ ] **Step 1: Implement ToolAgentView**

Create `web/src/components/ToolAgentView.tsx`:

```typescript
import { useState } from "react";
import { sendToolMessage } from "../api";
import type { ToolCallTraceView } from "../types";
import { ToolCallTracePanel } from "./ToolCallTracePanel";

export function ToolAgentView() {
  const [message, setMessage] = useState("");
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const [answer, setAnswer] = useState("");
  const [progress, setProgress] = useState<string[]>([]);
  const [toolCalls, setToolCalls] = useState<ToolCallTraceView[]>([]);
  const [noModel, setNoModel] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    const text = message.trim();
    if (!text || busy) return;
    setBusy(true);
    setError(null);
    setNoModel(false);
    setAnswer("");
    setProgress([]);
    setToolCalls([]);
    try {
      for await (const e of sendToolMessage({ message: text, session_id: sessionId })) {
        if (e.type === "progress") setProgress((p) => [...p, e.text]);
        else if (e.type === "tool_call") setToolCalls((c) => [...c, e]);
        else if (e.type === "answer") setAnswer(e.text);
        else if (e.type === "done") setSessionId(e.session_id);
        else if (e.type === "error") setError(e.detail);
      }
      setMessage("");
    } catch (err) {
      if (err instanceof Error && err.message === "NO_LOCAL_MODEL") setNoModel(true);
      else setError(err instanceof Error ? err.message : "Tool agent failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="tool-agent-view" aria-label="Tool Agent">
      {noModel && (
        <div className="error-banner" role="alert">
          Tool Agent needs a local model — set <code>SEARCH_AGENT_MODEL</code> (or{" "}
          <code>SEARCH_AGENT_SERVER_URL</code>) in <code>.env</code> and restart the backend.
        </div>
      )}
      <div className="tool-agent-view__composer">
        <input
          aria-label="Tool agent message"
          value={message}
          onChange={(ev) => setMessage(ev.target.value)}
          onKeyDown={(ev) => ev.key === "Enter" && submit()}
          placeholder="Ask the tool agent to do something…"
          disabled={busy}
        />
        <button onClick={submit} disabled={busy}>
          {busy ? "Running…" : "Send"}
        </button>
      </div>
      {error && <div className="error-banner">{error}</div>}
      {progress.length > 0 && (
        <ul className="tool-agent-view__progress">
          {progress.map((p, i) => (
            <li key={i}>{p}</li>
          ))}
        </ul>
      )}
      {toolCalls.length > 0 && <ToolCallTracePanel calls={toolCalls} />}
      {answer && <div className="tool-agent-view__answer">{answer}</div>}
    </section>
  );
}
```

- [ ] **Step 2: Write the component test**

Create `web/src/components/__tests__/ToolAgentView.test.tsx`:

```typescript
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ToolAgentView } from "../ToolAgentView";
import * as api from "../../api";

describe("ToolAgentView", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("renders streamed tool calls and the answer", async () => {
    async function* fake() {
      yield { type: "progress", turn: 1, text: "search · 3 docs" } as const;
      yield {
        type: "tool_call",
        tool_name: "search",
        status: "completed",
        arguments: {},
        result_summary: "3 items",
        latency_ms: 10,
        error: null,
      } as const;
      yield { type: "answer", text: "done answer" } as const;
      yield { type: "done", session_id: "s1", tool_calls: [], num_turns: 1 } as const;
    }
    vi.spyOn(api, "sendToolMessage").mockImplementation(fake as never);

    render(<ToolAgentView />);
    fireEvent.change(screen.getByLabelText("Tool agent message"), {
      target: { value: "find X" },
    });
    fireEvent.click(screen.getByText("Send"));

    await waitFor(() => expect(screen.getByText("done answer")).toBeInTheDocument());
    expect(screen.getByText(/search · 3 docs/)).toBeInTheDocument();
  });

  it("shows the no-model banner on NO_LOCAL_MODEL", async () => {
    vi.spyOn(api, "sendToolMessage").mockImplementation((() => {
      async function* g() {
        throw new Error("NO_LOCAL_MODEL");
        // eslint-disable-next-line no-unreachable
        yield undefined as never;
      }
      return g();
    }) as never);

    render(<ToolAgentView />);
    fireEvent.change(screen.getByLabelText("Tool agent message"), {
      target: { value: "hi" },
    });
    fireEvent.click(screen.getByText("Send"));

    await waitFor(() =>
      expect(screen.getByText(/needs a local model/)).toBeInTheDocument(),
    );
  });
});
```

- [ ] **Step 3: Run the component test**

Run: `cd web && npx vitest run src/components/__tests__/ToolAgentView.test.tsx`
Expected: both PASS.

- [ ] **Step 4: Add the view switcher to App.tsx**

In `web/src/App.tsx`:

1. Import: `import { ToolAgentView } from "./components/ToolAgentView";`
2. Add state near the other `useState` hooks (~line 60): `const [surface, setSurface] = useState<"assistant" | "tool">("assistant");`
3. In the header controls block (near the Connectors/Tools buttons, ~line 293), add a switcher:

```tsx
<div className="surface-switcher" role="tablist" aria-label="Surface">
  <button
    role="tab"
    aria-selected={surface === "assistant"}
    className={`icon-button${surface === "assistant" ? " active" : ""}`}
    onClick={() => setSurface("assistant")}
  >
    Assistant
  </button>
  <button
    role="tab"
    aria-selected={surface === "tool"}
    className={`icon-button${surface === "tool" ? " active" : ""}`}
    onClick={() => setSurface("tool")}
  >
    Tool Agent
  </button>
</div>
```

4. Wrap the existing composer + results layout so it renders only for the assistant surface, and render `ToolAgentView` for the tool surface. Locate the `<SearchComposer .../>` (app.tsx:338) through the end of the `results-layout` block (~line 411) and gate them:

```tsx
{surface === "assistant" ? (
  <>
    {/* existing SearchComposer ... results-layout ... unchanged */}
  </>
) : (
  <ToolAgentView />
)}
```

Leave the header, connectors/tools/history/console panels outside the gate (they apply to both surfaces).

- [ ] **Step 5: Typecheck + run the full frontend suite**

Run: `cd web && npm run typecheck && npx vitest run`
Expected: no type errors; all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add web/src/components/ToolAgentView.tsx web/src/components/__tests__/ToolAgentView.test.tsx web/src/App.tsx
git commit -m "feat(web): Tool Agent surface + Assistant/Tool Agent switcher"
```

---

## Task 5: Document the `/tool/*` surface

**Files:**
- Modify: `docs/tool-engine.md`

- [ ] **Step 1: Add a "Dedicated tool-agent surface" section**

Append to `docs/tool-engine.md`, after the "Routing into the tool engine" section:

```markdown
## Dedicated tool-agent surface (`/tool/*`)

Beyond the unified `/api/agent`, the tool engine has its own conversational
surface, parallel to `/search/*` and `/chat/*`:

- `POST /tool/send-tool-message` — runs `ToolAgentLoop` and streams Server-Sent
  Events: `progress` (per turn), `tool_call` (each completed call), `answer`,
  and a final `done`. Requires a local model (`SEARCH_AGENT_MODEL` /
  `SEARCH_AGENT_SERVER_URL`); returns **400** otherwise. Pass `stream:false` for
  a single JSON response.
- `GET /tool/tool-history` — past sessions for the caller (session proxy, like
  `/search/search-history`).

The router (`create_tool_router`, `src/internal/servers/query_and_chat/tool_backend.py`)
reuses the shared loop runner in `src/internal/servers/web/tool_agent_runner.py`.
In the web UI, the **Tool Agent** tab (Assistant | Tool Agent switcher) drives
this endpoint with a live tool-call trace.
```

- [ ] **Step 2: Commit**

```bash
git add docs/tool-engine.md
git commit -m "docs: document the /tool/* tool-agent surface"
```

---

## Self-Review

- **Spec coverage:** `/tool/send-tool-message` streaming (Task 2) ✓; `/tool/tool-history` (Task 2) ✓; the surgical relocation (Task 1) ✓; Tool Agent tab + switcher (Task 4) ✓; live SSE packet vocabulary (Task 2/3) ✓; no-model 400 + banner (Task 2/4) ✓; approvals — wired via `on_approval` param but passed `None` in the router for now (loop still runs; gated tools would block without a broker) — the spec calls for broker wiring; **deferred** to keep the first cut anonymous-friendly and testable. If full approval parity is required, a follow-up wires `on_approval` through `app.state.tool_approval_broker` for authenticated users, mirroring `/api/agent/stream`. Docs (Task 5) ✓; tests backend+frontend ✓.
- **Placeholder scan:** No TBD/TODO; all steps contain runnable code and exact commands.
- **Type consistency:** `ToolCallView` (backend) fields match `ToolCallTraceView` (frontend). `sendToolMessage`/`getToolHistory`/`ToolStreamEvent` names are consistent across Tasks 3–4. `create_tool_router(store, *, search_url, resolved)` signature matches its registration call and its test construction.

> **Note on approvals:** The design specifies wiring `on_approval` through the approval broker. This plan ships the endpoint with `on_approval=None` (loop runs end-to-end; no interactive gating) to keep Task 2 self-contained and anonymous-testable, and flags full broker parity as an explicit follow-up. Confirm during review whether approval parity must land in this PR.
