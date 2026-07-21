# Three Direct Surfaces (Search · Chat · Tool) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three explicit direct-surface tabs — Search (retrieval), Chat (pure LLM), Tool Agent (web) — alongside the existing auto-routing Assistant tab, and make the Tool Agent's `web_search` genuinely fetch the web.

**Architecture:** A new `/chat/send-chat-message` endpoint runs `PlainGenerationLoop` over the local model (no retrieval/tools), mirroring the existing `/tool/send-tool-message`. The Tool Agent's `web_search` is retargeted from `provider="retrieval"` to a serpapi→browser cascade built from existing primitives (`serpapi_search` + a `retrieval_search` call pointed at the browser server, which shares the `/retrieve` contract). The frontend grows a four-tab switcher with dedicated Search and Chat views.

**Tech Stack:** Python 3, FastAPI, Pydantic, pytest (backend); React 19, TypeScript, Vitest, React Testing Library (frontend).

## Global Constraints

- Never commit to `main`; work on branch `feat/tool-agent-surface` (extends open PR #447).
- Chat requires a local model. When `app.state.search_agent_manager` or `search_agent_tokenizer` is `None`, `/chat/send-chat-message` returns **HTTP 400** with the shared `NO_LOCAL_MODEL_MESSAGE` (import from `src.internal.servers.web.tool_agent_runner`).
- Streaming uses **SSE framing** (`media_type="text/event-stream"`, each line `data: <json>\n\n`), matching `/tool/send-tool-message` and the `streamAgent`/`sendToolMessage` parsers. Chat has no tools, so its stream emits only `answer` then `done` (and `error` on failure). It is NOT token-by-token — `PlainGenerationLoop` returns a complete answer.
- The web cascade is **sequential**: SerpAPI first; if it returns a non-empty, error-free result, use it and STOP; otherwise fall back to the browser server; if neither is usable, return `[]` (logged, no crash).
- The `web_search` retarget is **global** (seeded once in `tool_knowledge_base`), so it affects both the Tool tab and the `/api/agent` tool path. Do NOT change the `search`/`search_routing_tool`/`rag_routing_tool` tools — only `web_search`.
- Frontend: the new tab is a 4-way switcher `Assistant | Search | Chat | Tool Agent`; the pre-existing header "Tools" wrench button (Manage-tools admin) is untouched. Run vitest from `web/` (repo-root run fails with "document is not defined").
- Run `ruff check . --fix && ruff format .` before each backend commit; `cd web && npm run typecheck` before each frontend commit. Pre-commit runs ruff-format and aborts on reformat — re-add and re-commit if so.

---

## File Structure

New:
- `src/internal/servers/web/plain_chat_runner.py` — `_run_plain_chat` (PlainGenerationLoop runner).
- `web/src/components/ChatView.tsx`, `web/src/components/SearchView.tsx`.
- `tests/unit/test_chat_backend.py`, tests for the cascade, frontend tests.

Modified:
- `src/tools/search.py` — `make_web_cascade_search`.
- `src/tools/knowledge_base.py` — seed `web_search` with the cascade.
- `src/internal/servers/query_and_chat/chat_backend.py` — `/chat/send-chat-message`.
- `src/internal/servers/query_and_chat/models.py` — chat request/response models.
- `web/src/App.tsx`, `web/src/api.ts`, `web/src/types.ts`.
- `docs/tool-engine.md`, `docs/chat-engine.md`.

---

## Task 1: Web cascade for `web_search` (serpapi → browser)

**Files:**
- Modify: `src/tools/search.py` (add `make_web_cascade_search`)
- Modify: `src/tools/knowledge_base.py` (seed `web_search` with it)
- Test: `tests/unit/test_web_cascade_search.py`

**Interfaces:**
- Consumes: existing `serpapi_search(query, *, page=, page_size=, api_key=, timeout_seconds=) -> list[SearchPage]` and `search_tool(query, *, provider="retrieval", search_url=, page=, page_size=) -> list[SearchPage]` (both in `src/tools/search.py`); `SearchPage` (has `.url`, `.error`).
- Produces: `make_web_cascade_search(*, browser_search_url: str | None = None, serpapi_fn=serpapi_search, browser_fn=search_tool) -> Callable` returning an async `search_fn` with signature `(query, *, provider="serpapi", search_url=..., page_size=5, timeout_seconds=15) -> list[SearchPage]`.

- [ ] **Step 1: Write the failing cascade tests**

Create `tests/unit/test_web_cascade_search.py`:

```python
import asyncio

import pytest

from src.tools.search import SearchPage, make_web_cascade_search


def _ok(url):
    return SearchPage(title="t", summary="s", url=url)


def _err():
    return SearchPage(error="boom")


def test_serpapi_hit_skips_browser():
    async def fake_serp(query, **kw):
        return [_ok("http://serp/1")]

    async def fake_browser(query, **kw):  # must NOT be called
        raise AssertionError("browser should not run when serpapi returns results")

    fn = make_web_cascade_search(
        browser_search_url="http://browser/retrieve",
        serpapi_fn=fake_serp,
        browser_fn=fake_browser,
    )
    pages = asyncio.run(fn("q"))
    assert [p.url for p in pages] == ["http://serp/1"]


def test_serpapi_empty_falls_back_to_browser():
    async def fake_serp(query, **kw):
        return []

    async def fake_browser(query, *, provider, search_url, **kw):
        assert provider == "retrieval"
        assert search_url == "http://browser/retrieve"
        return [_ok("http://browser/1")]

    fn = make_web_cascade_search(
        browser_search_url="http://browser/retrieve",
        serpapi_fn=fake_serp,
        browser_fn=fake_browser,
    )
    pages = asyncio.run(fn("q"))
    assert [p.url for p in pages] == ["http://browser/1"]


def test_serpapi_error_falls_back_to_browser():
    async def fake_serp(query, **kw):
        return [_err()]

    async def fake_browser(query, **kw):
        return [_ok("http://browser/2")]

    fn = make_web_cascade_search(
        browser_search_url="http://browser/retrieve",
        serpapi_fn=fake_serp,
        browser_fn=fake_browser,
    )
    pages = asyncio.run(fn("q"))
    assert [p.url for p in pages] == ["http://browser/2"]


def test_no_browser_configured_returns_serp_result_as_is():
    async def fake_serp(query, **kw):
        return [_err()]

    fn = make_web_cascade_search(browser_search_url=None, serpapi_fn=fake_serp)
    pages = asyncio.run(fn("q"))
    assert pages and pages[0].error == "boom"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_web_cascade_search.py -v`
Expected: FAIL — `ImportError: cannot import name 'make_web_cascade_search'`.

- [ ] **Step 3: Implement `make_web_cascade_search`**

Add to `src/tools/search.py` (near the other search helpers; `serpapi_search`, `search_tool`, `SearchPage` are already defined in this file):

```python
def _pages_are_usable(pages: list[SearchPage]) -> bool:
    """True when at least one page carries a result and none is an error page."""
    if not pages:
        return False
    return any(p.url for p in pages) and not any(p.error for p in pages)


def make_web_cascade_search(
    *,
    browser_search_url: str | None = None,
    serpapi_fn=serpapi_search,
    browser_fn=search_tool,
):
    """Return a ``search_fn`` that tries SerpAPI, then falls back to the browser
    search server (retrieval-shaped ``/retrieve``). First usable result wins.

    Compatible with ``MultiQueryWebSearchTool(search_fn=...)``.
    """

    async def _cascade(
        query: str,
        *,
        provider: SearchProvider = "serpapi",
        search_url: str = "http://localhost:8000/retrieve",
        page: int = 1,
        page_size: int = 5,
        timeout_seconds: int = 15,
    ) -> list[SearchPage]:
        del provider, search_url  # cascade owns provider selection
        serp_pages = await serpapi_fn(
            query, page=page, page_size=page_size, timeout_seconds=timeout_seconds
        )
        if _pages_are_usable(serp_pages):
            return serp_pages
        if browser_search_url:
            try:
                return await browser_fn(
                    query,
                    provider="retrieval",
                    search_url=browser_search_url,
                    page=page,
                    page_size=page_size,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("browser cascade leg failed for %r: %s", query, exc)
        return serp_pages

    return _cascade
```

(Confirm `logger` exists in `search.py`; if not, add `logger = logging.getLogger(__name__)` and `import logging`.)

- [ ] **Step 4: Run cascade tests to green**

Run: `pytest tests/unit/test_web_cascade_search.py -v`
Expected: 4 passed.

- [ ] **Step 5: Retarget the seed + a regression test**

In `src/tools/knowledge_base.py`, change the `web_search` seed. Replace:

```python
        MultiQueryWebSearchTool(
            provider="retrieval", search_url=search_url, page_size=top_k
        ),
```

with:

```python
        MultiQueryWebSearchTool(
            search_fn=make_web_cascade_search(
                browser_search_url=os.getenv("AGENTIC_SEARCH_BROWSER_SEARCH_URL")
            ),
            page_size=top_k,
        ),
```

Add the imports at the top of `knowledge_base.py`: `import os` and extend the existing `from .search import ...` line with `make_web_cascade_search`.

Append to `tests/unit/test_web_cascade_search.py`:

```python
def test_seeded_web_search_uses_cascade_not_retrieval():
    from src.tools.knowledge_base import tool_knowledge_base

    tools = {t.name: t for t in tool_knowledge_base(search_url="http://x/retrieve")}
    web = tools["web_search"]
    # The cascade search_fn is bound; the tool no longer routes to retrieval.
    assert web._search_fn is not None
    assert web._provider != "retrieval" or web._search_fn.__name__ == "_cascade"
```

- [ ] **Step 6: Run the seed regression + verify no accidental breakage**

Run: `pytest tests/unit/test_web_cascade_search.py -v && pytest tests/unit -k "knowledge_base or tool" -q`
Expected: all pass.

- [ ] **Step 7: Lint + commit**

```bash
ruff check . --fix && ruff format .
git add src/tools/search.py src/tools/knowledge_base.py tests/unit/test_web_cascade_search.py
git commit -m "feat(tools): web_search fetches the web via serpapi->browser cascade"
```

---

## Task 2: Chat backend (`POST /chat/send-chat-message`)

**Files:**
- Create: `src/internal/servers/web/plain_chat_runner.py`
- Modify: `src/internal/servers/query_and_chat/models.py` (chat models)
- Modify: `src/internal/servers/query_and_chat/chat_backend.py` (new endpoint)
- Test: `tests/unit/test_chat_backend.py`

**Interfaces:**
- Consumes: `PlainGenerationLoop`, `PlainGenerationLoopConfig` from `src.agents.generation.plain`; `NO_LOCAL_MODEL_MESSAGE` from `src.internal.servers.web.tool_agent_runner`; store methods `get_chat_session`, `create_chat_session`, `list_chat_messages`, `add_chat_message`; `resolve_request_user`.
- Produces: `async _run_plain_chat(message, *, manager, tokenizer, history, on_turn=None) -> str`; new route `POST /chat/send-chat-message` on the router returned by `create_chat_router`.

- [ ] **Step 1: Add the chat models**

Append to `src/internal/servers/query_and_chat/models.py`:

```python
class SendChatMessageRequest(BaseModel):
    session_id: str | None = None
    message: str
    stream: bool = True


class ChatMessageResponse(BaseModel):
    session_id: str
    answer: str
    error: str | None = None
```

- [ ] **Step 2: Create the plain-chat runner**

Create `src/internal/servers/web/plain_chat_runner.py`:

```python
"""Runner for PlainGenerationLoop — pure LLM generation, no retrieval or tools.

Neutral module (like tool_agent_runner.py) so query_and_chat routers can reuse
it without importing app.py.
"""
from __future__ import annotations

from src.agents.generation.plain import (
    PlainGenerationLoop,
    PlainGenerationLoopConfig,
)


async def _run_plain_chat(
    message: str,
    *,
    manager,
    tokenizer,
    history: list,
    on_turn=None,
) -> str:
    """Run one PlainGenerationLoop turn over history + the new user message."""
    loop = PlainGenerationLoop(
        tokenizer=tokenizer,
        server_manager=manager,
        config=PlainGenerationLoopConfig(),
    )
    messages = [{"role": m.role, "content": m.content} for m in history] + [
        {"role": "user", "content": message}
    ]
    output = await loop.run(
        messages,
        sampling_params={"temperature": 0.7, "max_tokens": 512},
        on_turn=on_turn,
    )
    return output.final_answer or ""
```

- [ ] **Step 3: Write the chat endpoint tests (failing)**

Create `tests/unit/test_chat_backend.py`:

```python
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.db import AgenticSearchStore
from src.internal.servers.query_and_chat.chat_backend import create_chat_router


def _make_app(*, with_model: bool) -> FastAPI:
    store = AgenticSearchStore(":memory:")
    app = FastAPI()
    app.include_router(create_chat_router(store))
    app.state.search_agent_manager = object() if with_model else None
    app.state.search_agent_tokenizer = object() if with_model else None
    return app


def test_send_chat_message_no_model_returns_400():
    client = TestClient(_make_app(with_model=False))
    resp = client.post("/chat/send-chat-message", json={"message": "hi", "stream": False})
    assert resp.status_code == 400
    assert "requires a local model" in resp.json()["detail"]


def test_send_chat_message_streams_answer_then_done(monkeypatch):
    from src.internal.servers.query_and_chat import chat_backend

    async def fake_run_plain_chat(message, *, on_turn=None, **kw):
        return f"echo: {message}"

    monkeypatch.setattr(chat_backend, "_run_plain_chat", fake_run_plain_chat)

    client = TestClient(_make_app(with_model=True))
    with client.stream(
        "POST", "/chat/send-chat-message", json={"message": "hello", "stream": True}
    ) as resp:
        assert resp.status_code == 200
        events = [
            json.loads(line[len("data:"):].strip())
            for line in resp.iter_lines()
            if line.startswith("data:")
        ]
    types = [e["type"] for e in events]
    assert types == ["answer", "done"]
    assert events[0]["text"] == "echo: hello"
    assert events[-1]["session_id"]
```

- [ ] **Step 4: Run to verify failure**

Run: `pytest tests/unit/test_chat_backend.py -v`
Expected: FAIL — 404 (route not defined) / AttributeError on `_run_plain_chat`.

- [ ] **Step 5: Add the endpoint to `create_chat_router`**

In `src/internal/servers/query_and_chat/chat_backend.py`, add imports at the top:

```python
import asyncio
import json as _json
import logging
from collections.abc import AsyncGenerator

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from src.internal.llm.interfaces import ChatMessage  # if this path is wrong, use src.context import ChatMessage
from src.internal.servers.query_and_chat.models import (
    ChatMessageResponse,
    SendChatMessageRequest,
)
from src.internal.servers.users.api import resolve_request_user

logger = logging.getLogger(__name__)
_MAX_HISTORY_MESSAGES = 40
```

Note: verify `ChatMessage` — use `from src.context import ChatMessage` (a frozen dataclass with `role`/`content`), matching `tool_backend.py`. If `create_chat_router` already imports these names, do not duplicate.

Inside `create_chat_router(store)`, before `return router`, add:

```python
    @router.post("/send-chat-message", response_model=None)
    async def send_chat_message(body: SendChatMessageRequest, http_request: Request):
        from src.internal.servers.web.plain_chat_runner import _run_plain_chat
        from src.internal.servers.web.tool_agent_runner import NO_LOCAL_MODEL_MESSAGE

        manager = getattr(http_request.app.state, "search_agent_manager", None)
        tokenizer = getattr(http_request.app.state, "search_agent_tokenizer", None)
        if manager is None or tokenizer is None:
            raise HTTPException(status_code=400, detail=NO_LOCAL_MODEL_MESSAGE)

        user = resolve_request_user(http_request)
        user_id = user.id if user and not user.is_anonymous else None
        if body.session_id and store.get_chat_session(body.session_id):
            session_id = body.session_id
        else:
            session_id = store.create_chat_session(
                user_id=user_id,
                title=body.message[:80],
                metadata={"source": "chat"},
                session_id=body.session_id,
            ).id
        history = [
            ChatMessage(role=m.role, content=m.content)
            for m in store.list_chat_messages(session_id)
        ][-_MAX_HISTORY_MESSAGES:]
        store.add_chat_message(session_id, role="user", content=body.message)

        if not body.stream:
            try:
                answer = await _run_plain_chat(
                    body.message, manager=manager, tokenizer=tokenizer, history=history
                )
                store.add_chat_message(session_id, role="assistant", content=answer)
                return ChatMessageResponse(session_id=session_id, answer=answer)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Chat failed for: %r", body.message)
                return ChatMessageResponse(session_id=session_id, answer="", error=str(exc))

        async def _gen() -> AsyncGenerator[str, None]:
            def _sse(data: dict) -> str:
                return f"data: {_json.dumps(data)}\n\n"

            try:
                answer = await _run_plain_chat(
                    body.message, manager=manager, tokenizer=tokenizer, history=history
                )
                store.add_chat_message(session_id, role="assistant", content=answer)
                yield _sse({"type": "answer", "text": answer})
                yield _sse({"type": "done", "session_id": session_id})
            except Exception as exc:  # noqa: BLE001
                logger.exception("Streaming chat failed for: %r", body.message)
                yield _sse({"type": "error", "detail": str(exc)})

        return StreamingResponse(_gen(), media_type="text/event-stream")
```

- [ ] **Step 6: Run chat tests to green**

Run: `pytest tests/unit/test_chat_backend.py -v`
Expected: 2 passed.

- [ ] **Step 7: Import smoke + full tool/chat suite**

Run: `python -c "from src.internal.servers.web.app import create_web_app; create_web_app()" && pytest tests/unit/test_chat_backend.py tests/unit/test_tool_backend.py -q`
Expected: no import error; all pass.

- [ ] **Step 8: Lint + commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/servers/web/plain_chat_runner.py src/internal/servers/query_and_chat/models.py src/internal/servers/query_and_chat/chat_backend.py tests/unit/test_chat_backend.py
git commit -m "feat: add /chat/send-chat-message (direct-LLM PlainGenerationLoop, streaming)"
```

---

## Task 3: Frontend API clients + types

**Files:**
- Modify: `web/src/types.ts`
- Modify: `web/src/api.ts`
- Test: `web/src/components/__tests__/directSurfaceApi.test.ts`

**Interfaces:**
- Consumes: existing `requestJson`, `SearchFullResponse` type (already in types.ts from the search backend — confirm its name; if absent, define `{ all_executed_queries: string[]; search_docs: SearchDoc[]; error?: string }`).
- Produces:
  - `type ChatStreamEvent = { type: "answer"; text: string } | { type: "done"; session_id: string } | { type: "error"; detail: string }`
  - `async function* sendChatMessage(body, init?): AsyncGenerator<ChatStreamEvent>` (SSE; throws `Error("NO_LOCAL_MODEL")` on 400)
  - `function sendSearchMessage(body): Promise<SearchFullResponse>`

- [ ] **Step 1: Add types**

Append to `web/src/types.ts`:

```typescript
export type ChatStreamEvent =
  | { type: "answer"; text: string }
  | { type: "done"; session_id: string }
  | { type: "error"; detail: string };

export interface SendChatMessageBody {
  session_id?: string;
  message: string;
  stream?: boolean;
}

export interface SendSearchMessageBody {
  search_query: string;
  num_hits?: number;
}
```

If `SearchFullResponse`/`SearchDoc` are not already exported in types.ts, add:

```typescript
export interface SearchDocView {
  document_id: string;
  title: string;
  content: string;
  url?: string | null;
  score?: number;
}
export interface SearchFullResponse {
  all_executed_queries: string[];
  search_docs: SearchDocView[];
  error?: string | null;
}
```

- [ ] **Step 2: Add the API functions**

Append to `web/src/api.ts` (mirror `sendToolMessage` for the SSE generator):

```typescript
export async function* sendChatMessage(
  body: SendChatMessageBody,
  init?: Pick<RequestInit, "signal">,
): AsyncGenerator<ChatStreamEvent> {
  const response = await fetch("/chat/send-chat-message", {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stream: true, ...body }),
    signal: init?.signal,
  });
  if (response.status === 400) throw new Error("NO_LOCAL_MODEL");
  if (!response.ok || !response.body) {
    throw new Error(`Chat stream failed: ${response.status}`);
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
        if (payload) yield JSON.parse(payload) as ChatStreamEvent;
      }
    }
  }
}

export function sendSearchMessage(
  body: SendSearchMessageBody,
): Promise<SearchFullResponse> {
  return requestJson<SearchFullResponse>("/search/send-search-message", {
    method: "POST",
    body: JSON.stringify({
      search_query: body.search_query,
      num_hits: body.num_hits ?? 8,
      run_query_expansion: false,
      stream: false,
    }),
  });
}
```

Add the new type names to the `import type { ... } from "./types"` line at the top of api.ts.

- [ ] **Step 3: Write parser tests**

Create `web/src/components/__tests__/directSurfaceApi.test.ts`:

```typescript
import { describe, expect, it, vi } from "vitest";
import { sendChatMessage, sendSearchMessage } from "../../api";

function sseStream(chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(c) {
      for (const s of chunks) c.enqueue(enc.encode(s));
      c.close();
    },
  });
}

describe("sendChatMessage", () => {
  it("parses answer then done", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true, status: 200,
      body: sseStream([
        'data: {"type":"answer","text":"hi"}\n\n',
        'data: {"type":"done","session_id":"s1"}\n\n',
      ]),
    }));
    const types: string[] = [];
    for await (const e of sendChatMessage({ message: "q" })) types.push(e.type);
    expect(types).toEqual(["answer", "done"]);
  });

  it("throws NO_LOCAL_MODEL on 400", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 400 }));
    await expect(async () => {
      for await (const _ of sendChatMessage({ message: "q" })) void _;
    }).rejects.toThrow("NO_LOCAL_MODEL");
  });
});

describe("sendSearchMessage", () => {
  it("posts and returns docs", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true, status: 200,
      json: async () => ({ all_executed_queries: ["q"], search_docs: [] }),
    }));
    const r = await sendSearchMessage({ search_query: "q" });
    expect(r.all_executed_queries).toEqual(["q"]);
  });
});
```

- [ ] **Step 4: Run test + typecheck**

Run: `cd web && npx vitest run src/components/__tests__/directSurfaceApi.test.ts && npm run typecheck`
Expected: PASS; no type errors.

- [ ] **Step 5: Commit**

```bash
git add web/src/types.ts web/src/api.ts web/src/components/__tests__/directSurfaceApi.test.ts
git commit -m "feat(web): sendChatMessage (SSE) + sendSearchMessage clients + types"
```

---

## Task 4: Search & Chat views + four-tab switcher

**Files:**
- Create: `web/src/components/ChatView.tsx`, `web/src/components/SearchView.tsx`
- Modify: `web/src/App.tsx`
- Test: `web/src/components/__tests__/ChatView.test.tsx`, `web/src/components/__tests__/SearchView.test.tsx`

**Interfaces:**
- Consumes: `sendChatMessage`, `sendSearchMessage` (Task 3); existing `SourceGrid` component (`web/src/components/SourceGrid.tsx`) for rendering docs.

- [ ] **Step 1: Implement ChatView**

Create `web/src/components/ChatView.tsx`:

```typescript
import { useState } from "react";
import { sendChatMessage } from "../api";

export function ChatView() {
  const [message, setMessage] = useState("");
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const [answer, setAnswer] = useState("");
  const [noModel, setNoModel] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    const text = message.trim();
    if (!text || busy) return;
    setBusy(true); setError(null); setNoModel(false); setAnswer("");
    try {
      for await (const e of sendChatMessage({ message: text, session_id: sessionId })) {
        if (e.type === "answer") setAnswer(e.text);
        else if (e.type === "done") setSessionId(e.session_id);
        else if (e.type === "error") setError(e.detail);
      }
      setMessage("");
    } catch (err) {
      if (err instanceof Error && err.message === "NO_LOCAL_MODEL") setNoModel(true);
      else setError(err instanceof Error ? err.message : "Chat failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="chat-view" aria-label="Chat">
      {noModel && (
        <div className="error-banner" role="alert">
          Chat needs a local model — set <code>SEARCH_AGENT_MODEL</code> (or{" "}
          <code>SEARCH_AGENT_SERVER_URL</code>) in <code>.env</code> and restart the backend.
        </div>
      )}
      <div className="chat-view__composer">
        <input
          aria-label="Chat message"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          placeholder="Message the model directly…"
          disabled={busy}
        />
        <button onClick={submit} disabled={busy}>{busy ? "…" : "Send"}</button>
      </div>
      {error && <div className="error-banner">{error}</div>}
      {answer && <div className="chat-view__answer">{answer}</div>}
    </section>
  );
}
```

- [ ] **Step 2: Implement SearchView**

Create `web/src/components/SearchView.tsx`:

```typescript
import { useState } from "react";
import { sendSearchMessage } from "../api";
import type { SearchDocView } from "../types";
import { SourceGrid } from "./SourceGrid";

export function SearchView() {
  const [query, setQuery] = useState("");
  const [docs, setDocs] = useState<SearchDocView[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    const q = query.trim();
    if (!q || busy) return;
    setBusy(true); setError(null); setDocs([]);
    try {
      const r = await sendSearchMessage({ search_query: q });
      if (r.error) setError(r.error);
      setDocs(r.search_docs ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setBusy(false);
    }
  }

  const documents = docs.map((d) => ({
    id: d.document_id,
    title: d.title,
    content: d.content,
    url: d.url ?? undefined,
    score: d.score ?? 0,
    citation: d.title,
    metadata: {},
  }));

  return (
    <section className="search-view" aria-label="Search">
      <div className="search-view__composer">
        <input
          aria-label="Search query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          placeholder="Search the corpus…"
          disabled={busy}
        />
        <button onClick={submit} disabled={busy}>{busy ? "…" : "Search"}</button>
      </div>
      {error && <div className="error-banner">{error}</div>}
      <SourceGrid documents={documents} />
    </section>
  );
}
```

Note: `SourceGrid` expects the app's document shape. Before writing this, open `web/src/components/SourceGrid.tsx` and `web/src/types.ts` to confirm the exact `documents` prop shape; adapt the `documents` mapping above to match the real fields (id/title/content/url/score/citation/metadata) rather than guessing. If `SourceGrid`'s type differs, map to its actual interface.

- [ ] **Step 3: Write the view tests**

Create `web/src/components/__tests__/ChatView.test.tsx`:

```typescript
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ChatView } from "../ChatView";
import * as api from "../../api";

describe("ChatView", () => {
  it("renders the streamed answer", async () => {
    async function* fake() {
      yield { type: "answer", text: "hello there" } as const;
      yield { type: "done", session_id: "s1" } as const;
    }
    vi.spyOn(api, "sendChatMessage").mockImplementation(fake as never);
    render(<ChatView />);
    fireEvent.change(screen.getByLabelText("Chat message"), { target: { value: "hi" } });
    fireEvent.click(screen.getByText("Send"));
    await waitFor(() => expect(screen.getByText("hello there")).toBeInTheDocument());
  });

  it("shows the no-model banner on NO_LOCAL_MODEL", async () => {
    vi.spyOn(api, "sendChatMessage").mockImplementation((() => {
      async function* g() { throw new Error("NO_LOCAL_MODEL"); yield undefined as never; }
      return g();
    }) as never);
    render(<ChatView />);
    fireEvent.change(screen.getByLabelText("Chat message"), { target: { value: "hi" } });
    fireEvent.click(screen.getByText("Send"));
    await waitFor(() => expect(screen.getByText(/needs a local model/)).toBeInTheDocument());
  });
});
```

Create `web/src/components/__tests__/SearchView.test.tsx`:

```typescript
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SearchView } from "../SearchView";
import * as api from "../../api";

describe("SearchView", () => {
  it("renders returned docs", async () => {
    vi.spyOn(api, "sendSearchMessage").mockResolvedValue({
      all_executed_queries: ["q"],
      search_docs: [
        { document_id: "D1", title: "FAISS overview", content: "…", url: null, score: 1 },
      ],
    } as never);
    render(<SearchView />);
    fireEvent.change(screen.getByLabelText("Search query"), { target: { value: "faiss" } });
    fireEvent.click(screen.getByText("Search"));
    await waitFor(() => expect(screen.getByText(/FAISS overview/)).toBeInTheDocument());
  });
});
```

- [ ] **Step 4: Run the view tests**

Run: `cd web && npx vitest run src/components/__tests__/ChatView.test.tsx src/components/__tests__/SearchView.test.tsx`
Expected: all pass. (If SearchView fails on the doc shape, fix the `documents` mapping to match `SourceGrid`'s real prop type, then re-run.)

- [ ] **Step 5: Wire the four-tab switcher into App.tsx**

In `web/src/App.tsx`:

1. Imports: `import { ChatView } from "./components/ChatView";` and `import { SearchView } from "./components/SearchView";`
2. Widen the surface state (it currently is `"assistant" | "tool"` from the prior task):
   `const [surface, setSurface] = useState<"assistant" | "search" | "chat" | "tool">("assistant");`
3. In the header switcher block, add two buttons between Assistant and Tool Agent, following the exact pattern of the existing switcher buttons:

```tsx
<button role="tab" aria-selected={surface === "search"}
  className={`icon-button${surface === "search" ? " active" : ""}`}
  onClick={() => setSurface("search")}>Search</button>
<button role="tab" aria-selected={surface === "chat"}
  className={`icon-button${surface === "chat" ? " active" : ""}`}
  onClick={() => setSurface("chat")}>Chat</button>
```

4. Extend the surface gates. The prior task used two gates: one around `SearchComposer` (+ assistant error banner) rendered when `surface === "assistant"`, and one around the results-layout rendered as `surface === "assistant" ? (<results/>) : (<ToolAgentView/>)`. Change them to:
   - Gate 1 (composer): keep `surface === "assistant" && (…)`.
   - Gate 2 (main body): replace the binary with a switch:

```tsx
{surface === "assistant" ? (
  <>{/* existing results-layout unchanged */}</>
) : surface === "search" ? (
  <SearchView />
) : surface === "chat" ? (
  <ChatView />
) : (
  <ToolAgentView />
)}
```

Keep the header and the Connectors/Tools/History/Console panels ungated (shown on all surfaces), exactly as before. Do not alter existing assistant state/handlers.

- [ ] **Step 6: Typecheck + full frontend suite**

Run: `cd web && npm run typecheck && npx vitest run`
Expected: no type errors; ALL tests pass (the pre-existing App tests + Tasks 3–4 additions).

- [ ] **Step 7: Commit**

```bash
git add web/src/components/ChatView.tsx web/src/components/SearchView.tsx web/src/components/__tests__/ChatView.test.tsx web/src/components/__tests__/SearchView.test.tsx web/src/App.tsx
git commit -m "feat(web): Search + Chat views and four-tab surface switcher"
```

---

## Task 5: Document the direct surfaces

**Files:**
- Modify: `docs/chat-engine.md`, `docs/tool-engine.md`

- [ ] **Step 1: Document the chat surface**

Append to `docs/chat-engine.md` a section:

```markdown
## Direct chat surface (`/chat/send-chat-message`)

Beyond the `/api/agent` auto-router, chat has a direct endpoint that calls the
local model with no retrieval and no tools:

- `POST /chat/send-chat-message` — runs `PlainGenerationLoop` over the session
  history + the new message, streaming SSE `answer` then `done` (`error` on
  failure). Requires a local model (`SEARCH_AGENT_MODEL` /
  `SEARCH_AGENT_SERVER_URL`); returns **400** otherwise. `stream:false` returns
  one JSON `{ session_id, answer }`. The runner is
  `src/internal/servers/web/plain_chat_runner.py`.

In the web UI, the **Chat** tab drives this endpoint.
```

- [ ] **Step 2: Update the tool-engine web note**

In `docs/tool-engine.md`, add a note that `web_search` now fetches the web via a
serpapi→browser cascade:

```markdown
### `web_search` fetches the real web

The seeded `web_search` tool uses a sequential cascade: SerpAPI first
(`SERP_API_KEY`), falling back to the browser search server
(`AGENTIC_SEARCH_BROWSER_SEARCH_URL`, a `/retrieve`-shaped playwright server)
when SerpAPI is empty or unavailable. With neither configured it returns no
results and the agent answers without web context. This applies to both the
Tool Agent tab and the `/api/agent` tool path.
```

- [ ] **Step 3: Commit**

```bash
git add docs/chat-engine.md docs/tool-engine.md
git commit -m "docs: document direct chat surface + web_search cascade"
```

---

## Self-Review

- **Spec coverage:** Search tab (frontend-only, Task 3/4) ✓; Chat direct endpoint + local-model + streaming + 400 (Task 2) ✓; Tool web cascade serpapi→browser + global retarget (Task 1) ✓; four-tab switcher (Task 4) ✓; docs (Task 5) ✓; no new DB tables (reuses sessions) ✓; Assistant untouched ✓.
- **Placeholder scan:** No TBD/TODO; every code step has runnable code and exact commands. Two steps flag a real verification the implementer must do (the `ChatMessage` import path; the `SourceGrid` prop shape) rather than guessing — these are verify-then-adapt instructions, not placeholders.
- **Type consistency:** `NO_LOCAL_MODEL_MESSAGE` reused from Task-2-era `tool_agent_runner` (added in PR #447). `sendChatMessage`/`ChatStreamEvent` names consistent across Tasks 3–4. `_run_plain_chat` signature identical in the runner, the endpoint call, and the monkeypatch target. `make_web_cascade_search` signature matches its test construction and the seed call.

### Known deviations from the spec (intentional)
- Chat streaming is `answer`+`done` (full answer), not token-by-token — `PlainGenerationLoop` returns a complete answer. Called out in the spec and Global Constraints.
- The browser cascade leg reuses `search_tool(provider="retrieval", search_url=browser_search_url)` because the browser server shares the `/retrieve` contract — simpler than a bespoke browser HTTP client, same wire behavior.
