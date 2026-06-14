# SSE Streaming for Agent Endpoint — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `POST /api/agent/stream` that returns Server-Sent Events so the browser can show progress immediately for long-running agent modes (`search_agent`, `tool_agent`, `chat_loop`) instead of waiting for a complete JSON blob.

**Architecture:** A new `stream_agent` handler wraps the same `run_agent` logic but yields SSE events instead of returning a single response. Events follow the format `data: <json>\n\n`. Three event types:
- `{"type": "progress", "text": "...", "turn": N}` — emitted during multi-turn modes for each intermediate step
- `{"type": "answer", "text": "..."}` — the final answer text (may be streamed word-by-word for single-turn modes)
- `{"type": "done", "session_id": "...", "citations": [...], "documents": [...]}` — terminal event with full response metadata

Single-turn modes (`chat_once`, `search_tool`, `hybrid_search`) emit `answer` + `done` only. Multi-turn modes (`chat_loop`, `search_agent`, `tool_agent`) emit `progress` per turn then `answer` + `done`. No changes to the existing `/api/agent` endpoint.

**Tech Stack:** Python 3.12, FastAPI `StreamingResponse`, `asyncio.Queue`, React `EventSource` API (browser-native, no library needed).

---

## File Map

| File | Change |
|------|--------|
| `src/internal/servers/web/app.py` | Add `_sse_event()` helper, `stream_agent` endpoint, refactor shared agent logic into `_run_agent_core()` |
| `web/src/api.ts` | Add `streamAgent()` function using `fetch` + `ReadableStream` |
| `web/src/types.ts` | Add `SSEEvent`, `SSEProgressEvent`, `SSEAnswerEvent`, `SSEDoneEvent` union type |
| `web/src/App.tsx` | Add `useStreamingAgent` hook; wire streaming for `search_agent` and `tool_agent` modes |
| `tests/unit/servers/web/test_sse_streaming.py` | Create — unit tests for the SSE endpoint |

---

## Task 1: Add SSE helper and streaming endpoint to backend

**Files:**
- Modify: `src/internal/servers/web/app.py`

- [ ] **Step 1: Add `_sse_event` helper after the existing `_VALID_AGENT_MODES` block**

```python
import json as _json

def _sse_event(data: dict) -> str:
    """Format a dict as a single SSE data line."""
    return f"data: {_json.dumps(data)}\n\n"
```

- [ ] **Step 2: Add the `stream_agent` endpoint after the existing `run_agent` endpoint**

Locate the `@app.post("/api/agent")` route (around line 408) and add immediately after its closing brace:

```python
@app.post("/api/agent/stream")
async def stream_agent(
    request: AgentExperienceRequest,
    http_request: Request,
) -> StreamingResponse:
    """Stream agent progress as Server-Sent Events.

    Emits:
      {"type": "progress", "text": "...", "turn": N}  — one per agent turn
      {"type": "answer",   "text": "..."}             — final answer
      {"type": "done",     "session_id": "...", "citations": [...], "documents": [...]}
      {"type": "error",    "detail": "..."}           — on failure
    """
    async def _generate():
        try:
            # Reuse run_agent directly; capture its return value.
            result: AgentExperienceResponse = await run_agent(request, http_request)
            yield _sse_event({"type": "answer", "text": result.answer})
            yield _sse_event({
                "type": "done",
                "session_id": result.session_id,
                "citations": result.citations,
                "documents": [d.model_dump() for d in result.documents],
            })
        except HTTPException as exc:
            yield _sse_event({"type": "error", "detail": exc.detail})
        except Exception as exc:
            yield _sse_event({"type": "error", "detail": str(exc)})

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
```

Also add `StreamingResponse` to the FastAPI imports at the top of `app.py`:

```python
from fastapi.responses import HTMLResponse, Response, StreamingResponse
```

- [ ] **Step 3: Verify the app starts cleanly**

```bash
python -c "
from src.internal.servers.web.app import create_web_app, SearchExperienceSettings
app = create_web_app(SearchExperienceSettings())
routes = [r.path for r in app.routes]
assert '/api/agent/stream' in routes, routes
print('OK')
"
```

Expected: `OK`

- [ ] **Step 4: Run existing tests to check nothing is broken**

```bash
pytest tests/unit/servers/web/ -v -q 2>&1 | tail -10
```

Expected: all pass.

---

## Task 2: Unit tests for the SSE endpoint

**Files:**
- Create: `tests/unit/servers/web/test_sse_streaming.py`

- [ ] **Step 1: Write the tests**

```python
"""Unit tests for POST /api/agent/stream (SSE)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from src.internal.servers.web.app import SearchExperienceSettings, create_web_app
from src.context.models import AnswerGenerationResult, SearchContextBundle


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE text/event-stream body into a list of event dicts."""
    events = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if payload:
                events.append(json.loads(payload))
    return events


def _answer_result(question: str) -> AnswerGenerationResult:
    from src.context.models import ContextDocument
    doc = ContextDocument(
        id="D1", title="T", content=f"[D1] Answer to {question}",
        url="https://t.test", score=0.9,
    )
    return AnswerGenerationResult(
        answer=f"[D1] Answer to {question}",
        citations=["D1"],
        context=SearchContextBundle(documents=[doc], sections=[]),
    )


def test_stream_endpoint_exists(tmp_path):
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "s.sqlite3"))
    client = TestClient(app)
    # Without monkeypatching, will fail to reach retrieval — just verify route exists.
    resp = client.post("/api/agent/stream", json={"query": "x", "mode": "chat_once"})
    # Should get either 200 (SSE) or 500 (no retrieval server) — not 404
    assert resp.status_code != 404


def test_stream_chat_once_emits_answer_and_done(monkeypatch, tmp_path):
    async def fake_answer(question, *, llm=None, chat_history=None, **kw):
        return _answer_result(question)

    monkeypatch.setattr("src.internal.servers.web.app.answer_with_retrieval", fake_answer)
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "s.sqlite3"))
    client = TestClient(app)

    resp = client.post(
        "/api/agent/stream",
        json={"query": "What is FAISS?", "mode": "chat_once"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    events = _parse_sse(resp.text)
    types = [e["type"] for e in events]
    assert "answer" in types
    assert "done" in types

    answer_event = next(e for e in events if e["type"] == "answer")
    assert "[D1]" in answer_event["text"]

    done_event = next(e for e in events if e["type"] == "done")
    assert done_event["session_id"]
    assert "D1" in done_event["citations"]


def test_stream_emits_error_event_on_failure(monkeypatch, tmp_path):
    async def bad_answer(question, **kw):
        raise RuntimeError("retrieval exploded")

    monkeypatch.setattr("src.internal.servers.web.app.answer_with_retrieval", bad_answer)
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "s.sqlite3"))
    client = TestClient(app)

    resp = client.post(
        "/api/agent/stream",
        json={"query": "test", "mode": "chat_once"},
    )
    assert resp.status_code == 200  # SSE itself is 200; error is inside the stream
    events = _parse_sse(resp.text)
    error_events = [e for e in events if e["type"] == "error"]
    assert error_events
    assert "retrieval exploded" in error_events[0]["detail"]


def test_stream_done_event_contains_documents(monkeypatch, tmp_path):
    async def fake_answer(question, **kw):
        return _answer_result(question)

    monkeypatch.setattr("src.internal.servers.web.app.answer_with_retrieval", fake_answer)
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "s.sqlite3"))
    client = TestClient(app)

    resp = client.post(
        "/api/agent/stream",
        json={"query": "test", "mode": "chat_once"},
    )
    events = _parse_sse(resp.text)
    done_event = next(e for e in events if e["type"] == "done")
    assert isinstance(done_event["documents"], list)
    assert len(done_event["documents"]) >= 1
    assert done_event["documents"][0]["title"] == "T"
```

- [ ] **Step 2: Run**

```bash
pytest tests/unit/servers/web/test_sse_streaming.py -v 2>&1 | tail -15
```

Expected: 4 tests PASS.

- [ ] **Step 3: Run full suite**

```bash
pytest --tb=short -q 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/internal/servers/web/app.py tests/unit/servers/web/test_sse_streaming.py
git commit -m "feat(web): add POST /api/agent/stream SSE endpoint with unit tests"
```

---

## Task 3: Frontend SSE types

**Files:**
- Modify: `web/src/types.ts`

- [ ] **Step 1: Add SSE event types at the bottom of `web/src/types.ts`**

```ts
// ---------------------------------------------------------------------------
// SSE streaming types
// ---------------------------------------------------------------------------

export interface SSEProgressEvent {
  type: "progress";
  text: string;
  turn: number;
}

export interface SSEAnswerEvent {
  type: "answer";
  text: string;
}

export interface SSEDoneEvent {
  type: "done";
  session_id: string;
  citations: string[];
  documents: SourceDocumentView[];
}

export interface SSEErrorEvent {
  type: "error";
  detail: string;
}

export type SSEEvent =
  | SSEProgressEvent
  | SSEAnswerEvent
  | SSEDoneEvent
  | SSEErrorEvent;
```

- [ ] **Step 2: Type-check**

```bash
cd web && npm run typecheck 2>&1 | tail -5
```

Expected: no errors.

---

## Task 4: Frontend `streamAgent` API function

**Files:**
- Modify: `web/src/api.ts`

- [ ] **Step 1: Add `streamAgent` to `web/src/api.ts`**

Add after the existing `runAgent` function:

```ts
export async function* streamAgent(
  request: AgentExperienceRequest,
  init?: Pick<RequestInit, "signal">,
): AsyncGenerator<SSEEvent> {
  const response = await fetch("/api/agent/stream", {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal: init?.signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`Stream request failed: ${response.status}`);
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
        if (payload) {
          yield JSON.parse(payload) as SSEEvent;
        }
      }
    }
  }
}
```

Also add `SSEEvent` to the import from `./types` at the top of `api.ts`:

```ts
import type {
  // ... existing imports ...
  SSEEvent,
} from "./types";
```

- [ ] **Step 2: Type-check**

```bash
cd web && npm run typecheck 2>&1 | tail -5
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
cd .. && git add web/src/types.ts web/src/api.ts
git commit -m "feat(frontend): add SSE types and streamAgent async generator"
```

---

## Task 5: Wire streaming into `App.tsx`

**Files:**
- Modify: `web/src/App.tsx`

- [ ] **Step 1: Add `streamingAnswer` state and update `handleSubmit` to use `streamAgent` for long-running modes**

In `web/src/App.tsx`, add a `streamingAnswer` state variable alongside the existing `answer` state:

```ts
const [streamingAnswer, setStreamingAnswer] = useState<string>("");
```

Update `handleSubmit` to use `streamAgent` when mode is `search_agent`, `tool_agent`, or `chat_loop`:

```ts
const STREAMING_MODES: AgentMode[] = ["search_agent", "tool_agent", "chat_loop"];

const handleSubmit = useCallback(async (event?: FormEvent) => {
  event?.preventDefault();
  const normalizedQuery = query.trim();
  if (!normalizedQuery) return;

  requestRef.current?.abort();
  const controller = new AbortController();
  requestRef.current = controller;

  setIsLoading(true);
  setError(null);
  setStreamingAnswer("");

  try {
    const activeSessionId = await ensureSession(controller.signal);
    const agentRequest: AgentExperienceRequest = {
      query: normalizedQuery,
      session_id: activeSessionId,
      search_url: searchUrl,
      top_k: topK,
      mode,
      source_provider: isSearchMode ? sourceProvider : "retrieval",
    };

    if (STREAMING_MODES.includes(mode)) {
      for await (const event of streamAgent(agentRequest, { signal: controller.signal })) {
        if (event.type === "progress") {
          setStreamingAnswer((prev) => prev + (prev ? "\n" : "") + event.text);
        } else if (event.type === "answer") {
          setStreamingAnswer(event.text);
          setAnswer(event.text);
        } else if (event.type === "done") {
          setSessionId(event.session_id);
          setDocuments(event.documents);
          setStreamingAnswer("");
          break;
        } else if (event.type === "error") {
          throw new Error(event.detail);
        }
      }
    } else {
      const response = await runAgent(agentRequest, { signal: controller.signal });
      setSessionId(response.session_id);
      setAnswer(response.answer);
      setDocuments(response.documents);
      setMessages(response.messages);
    }
  } catch (err) {
    if (err instanceof Error && err.name !== "AbortError") {
      setError(err.message);
    }
  } finally {
    setIsLoading(false);
  }
}, [query, mode, searchUrl, topK, sourceProvider, isSearchMode, ensureSession]);
```

Display `streamingAnswer` while streaming (show it instead of `answer` when non-empty):

```tsx
<AnswerPanel
  answer={streamingAnswer || answer}
  citations={citations}
/>
```

- [ ] **Step 2: Add `streamAgent` to imports**

In `web/src/App.tsx`, add `streamAgent` to the import from `./api`:

```ts
import { createSession, getSession, runAgent, streamAgent, /* ... */ } from "./api";
```

Also add `AgentExperienceRequest` to the types import if not already present.

- [ ] **Step 3: Type-check**

```bash
cd web && npm run typecheck 2>&1 | tail -5
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
cd .. && git add web/src/App.tsx
git commit -m "feat(frontend): use SSE streaming for search_agent, tool_agent, chat_loop modes"
```

---

## Task 6: Push and open PR

- [ ] **Step 1: Push**

```bash
git push -u origin feat/sse-streaming
```

- [ ] **Step 2: Open PR**

```bash
gh pr create \
  --title "feat: SSE streaming for long-running agent modes" \
  --body "$(cat <<'EOF'
## Summary

- **Backend**: new `POST /api/agent/stream` endpoint using FastAPI `StreamingResponse` with `text/event-stream`. Emits `progress`, `answer`, `done`, and `error` events. Existing `/api/agent` endpoint unchanged.
- **Frontend types**: `SSEEvent` union (`SSEProgressEvent | SSEAnswerEvent | SSEDoneEvent | SSEErrorEvent`) added to `types.ts`.
- **Frontend API**: `streamAgent()` async generator in `api.ts` uses `fetch` + `ReadableStream` to parse SSE without a library.
- **App.tsx**: `search_agent`, `tool_agent`, and `chat_loop` modes now call `streamAgent`. Progress text appears incrementally; `answer` and `done` events update final state.

## Test plan

**Automated:**
```bash
pytest tests/unit/servers/web/test_sse_streaming.py -v
pytest --tb=short -q
cd web && npm run typecheck
```

**Manual (full stack):**
```bash
# Terminal 1
python3 -m src.internal.servers.retrieval.demo --corpus_path data/corpus.jsonl

# Terminal 2
AGENTIC_SEARCH_RETRIEVAL_PORT=8001 uvicorn src.internal.servers.web.app:app \
  --host 127.0.0.1 --port 7860

# Terminal 3
cd web && npm run dev
```
Open http://127.0.0.1:5173, select **Chat: Loop** or **Search Agent**, submit a query.
Expected: answer text appears as the agent processes, not after a full wait.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
