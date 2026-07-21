# Design: First-class Tool-Agent surface (`/tool/*` API + Tool Agent tab)

Date: 2026-07-21
Status: Approved (brainstorming)

## Problem

The tool engine (`ToolAgentLoop`) is a second-class citizen. Search and chat
each own a dedicated backend router — `create_search_router` → `/search/*`
([search_backend.py](../../../src/internal/servers/query_and_chat/search_backend.py))
and `create_chat_router` → `/chat/*`
([chat_backend.py](../../../src/internal/servers/query_and_chat/chat_backend.py)).
The tool engine has no equivalent: it runs only through the unified
`/api/agent` (auto-routed or `mode=tool_agent`), and surfaces in the UI as a
`ToolCallTracePanel` bolted onto the shared answer column. There is no dedicated
tool-agent API and no dedicated frontend surface.

## Goal

Give the tool engine its **own API** and its **own frontend**, parallel to
search and chat:

- A `/tool/*` router exposing a conversational tool-agent endpoint (streaming
  live tool calls) plus a history endpoint.
- A dedicated **Tool Agent** frontend surface where the user converses with the
  `ToolAgentLoop` and watches multi-turn tool calls execute live.

Decisions locked during brainstorming:
- Purpose: **tool-agent chat surface** (conversational, watch multi-turn tool
  calls) — not a single-tool runner console.
- Frontend: **dedicated tab/page**, not an enhanced inline panel.
- Endpoint: **streams live** tool-call events.
- Backend: **Approach A** — thin router reusing the existing `_run_tool_agent`.

## Non-goals (YAGNI)

- No new tool-execution engine — reuse `ToolAgentLoop`.
- No changes to `/api/agent` routing behavior.
- No separate tool-history DB table — reuse chat sessions as history, exactly as
  `/search/search-history` does.
- No auth changes.
- No single-tool "runner console" (browse + invoke one tool with a form). That
  capability already exists at `POST /admin/tools/{name}/invoke`.

## Architecture

Mirror the search/chat backends. The one prerequisite is a **pure relocation**
to avoid a circular import (the new router lives in `query_and_chat/`, which
`app.py` imports; so the router cannot import back from `app.py`).

### The surgical move

Move three module-level helpers out of
[app.py](../../../src/internal/servers/web/app.py) into a new neutral module
`src/internal/servers/web/tool_agent_runner.py`, **verbatim**:

- `_run_tool_agent` (app.py:728) — runs `ToolAgentLoop`, accepts `on_turn` /
  `on_approval` hooks, returns the canonical
  `(answer, citations, documents, intent, extra)` tuple.
- `_extract_tool_calls_and_docs`
- `_infer_intent_from_output`

`app.py` imports these from the new module. Behavior is unchanged; the existing
`mode == "tool_agent"` path in `_run_agent_impl` keeps working. This is the only
change to existing behavior — a relocation with no logic edits.

### New router

`src/internal/servers/query_and_chat/tool_backend.py`:

```python
def create_tool_router(store: AgenticSearchStore, *, search_url: str = ...) -> APIRouter:
    router = APIRouter(prefix="/tool", tags=["tool"])
    ...
    return router
```

Registered in `app.py`'s `create_web_app()` next to the search/chat routers:

```python
app.include_router(create_tool_router(db, search_url=search_url))
```

## Backend endpoints (`/tool/*`)

### `POST /tool/send-tool-message`

Streaming NDJSON (`media_type="application/x-ndjson"`), paralleling
`/search/send-search-message`.

Request body:

```json
{
  "session_id": "optional; created if absent",
  "message": "user text",
  "run_search_tool": true,
  "stream": true
}
```

Behavior:

1. Read `manager` / `tokenizer` from
   `http_request.app.state.search_agent_manager` /
   `search_agent_tokenizer` (loaded at lifespan).
2. If either is `None` → **HTTP 400** with the same message the agent path uses:
   `"tool_agent mode requires a local model. Set SEARCH_AGENT_MODEL or
   SEARCH_AGENT_SERVER_URL in .env and restart."`
3. Resolve/create the session; load bounded session history from `store`.
4. Call
   `_run_tool_agent(message, manager=…, tokenizer=…, search_url=…, history=…,
   resolved=…, on_turn=…, on_approval=…, with_search_tool=run_search_tool)`.
5. Stream packets (see below) as the loop runs.
6. Persist the assistant turn to `store`, reusing the existing finalize/persist
   path so history, citations, and metadata match the main app.
7. When `stream=false`: run to completion and return a single JSON
   `{ answer, tool_calls, documents, num_turns, session_id }`.

Approvals: when a local authenticated user is present, wire `on_approval`
through `http_request.app.state.tool_approval_broker`, exactly as
`/api/agent/stream` does, so gated tools still prompt.

### `GET /tool/tool-history`

Params `limit` (1–1000, default 100), `filter_days` (optional, > 0). Returns
past sessions for the authenticated caller, same shape and semantics as
`/search/search-history` (anonymous → empty list; sessions used as the history
proxy).

## Streaming protocol

Reuse the packet vocabulary already emitted by `/api/agent/stream` so the
frontend NDJSON parser stays consistent. Emitted in order:

- `{"type":"progress","turn":N,"text":"<tool_name> · <doc_count> docs"}` — one
  per turn, from the `on_turn` hook (`"writing answer..."` when no tool).
- `{"type":"tool_call", ...ToolCallView}` — each completed tool call: name,
  arguments, result, status, latency.
- `{"type":"answer","text":"..."}` — the final answer.
- `{"type":"done","session_id":...,"tool_calls":[...],"documents":[...],"num_turns":N}`.
- `{"type":"error","detail":"..."}` — on failure.

## Frontend — Tool Agent tab

- Add a top-level view switcher in the header: **Assistant** (today's unified
  `/api/agent` surface, default, unchanged) and **Tool Agent** (new). The new
  tab is named "Tool Agent" to avoid colliding with the existing header
  **Tools** button (wrench), which opens the *Manage tools* admin panel.
- New `web/src/components/ToolAgentView.tsx`: its own composer plus a
  **trace-first** layout — a running conversation where `ToolCallTracePanel`
  entries appear live as `tool_call` packets stream in, with the final answer
  rendered below.
- New API functions in `web/src/api.ts`:
  - `sendToolMessage(...)` — POSTs `/tool/send-tool-message`, parses the NDJSON
    stream, dispatches typed packets to callbacks.
  - `getToolHistory(...)` — GET `/tool/tool-history`.
- Degraded state: on a 400 (no local model), the view shows a clear banner
  ("Tool Agent needs a local model — set `SEARCH_AGENT_MODEL`…") instead of a
  generic error.

## Error handling

- No local model → 400 (both endpoints that need it), surfaced as a banner.
- Loop exception mid-stream → `error` packet, and the stream closes cleanly
  (mirrors `_stream_generator` in `search_backend`).
- Anonymous caller on `tool-history` → empty list, not 401.
- Empty final answer → fall back to the last assistant message, reusing
  `_run_tool_agent`'s `_assistant_fallback` (same policy as explicit
  `mode=tool_agent`).

## Testing

Backend (`tests/unit/`):
- `send-tool-message` streaming happy path — fake manager/tokenizer + stubbed
  loop; assert ordered `progress` → `tool_call` → `answer` → `done` packets.
- No-model path returns 400 with the expected message.
- `tool-history` returns the session-proxy shape; anonymous → empty.
- Regression: `app.py`'s existing `mode=tool_agent` path still works after the
  helper relocation (import + one loop run).

Frontend (`web/src/components/__tests__/`):
- `ToolAgentView` renders streamed tool calls in order.
- View switcher toggles Assistant ↔ Tool Agent.
- 400 response renders the no-model banner.

## Files touched

New:
- `src/internal/servers/web/tool_agent_runner.py`
- `src/internal/servers/query_and_chat/tool_backend.py`
- `web/src/components/ToolAgentView.tsx`
- tests (backend + frontend)

Modified:
- `src/internal/servers/web/app.py` — import relocated helpers; register the
  tool router.
- `web/src/App.tsx` — view switcher + mount `ToolAgentView`.
- `web/src/api.ts`, `web/src/types.ts` — new API fns + packet types.
- `docs/tool-engine.md` — document the new `/tool/*` surface.
