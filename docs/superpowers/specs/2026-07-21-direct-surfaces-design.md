# Design: three direct surfaces (Search · Chat · Tool) alongside Assistant

Date: 2026-07-21
Status: Approved (brainstorming)
Builds on: `2026-07-21-tool-agent-surface-design.md` (PR #447, same branch)

## Problem

The goal is three cleanly separated surfaces, each with a distinct data source:
**search** fetches retrieval info, **chat** calls the LLM directly, **tool**
fetches web info. Current reality:

- **Search** — `/search/send-search-message` already does retrieval-only
  (`run_expanded_search`). ✅ matches the goal.
- **Chat** — `/chat/*` is only session CRUD. There is no chat-inference
  endpoint; chat only happens through the `/api/agent` auto-router. ❌
- **Tool** — `/tool/send-tool-message` works, but its `web_search` tool is
  seeded with `provider="retrieval"`, so it searches the corpus, not the web. ❌

## Goal

Add three explicit, direct surfaces in the UI alongside the existing
auto-routing Assistant tab, and make each hit its own endpoint with no intent
classification:

| Tab | Endpoint | Behavior |
|---|---|---|
| Assistant | `/api/agent` (existing) | auto-router — unchanged |
| Search | `POST /search/send-search-message` (exists) | retrieval only → docs, no LLM synthesis |
| Chat | `POST /chat/send-chat-message` (NEW) | `PlainGenerationLoop` over the local model → streamed answer, multi-turn, no retrieval/tools |
| Tool Agent | `POST /tool/send-tool-message` (exists; retarget) | `ToolAgentLoop` whose `web_search` uses a serpapi→browser cascade |

Decisions locked during brainstorming:
- UI: keep Assistant + add three explicit tabs (four total).
- Chat backend: **local model** via `PlainGenerationLoop` (not the OpenAI llm);
  400 if no local model, same pattern as the tool surface.
- Chat behavior: pure LLM, multi-turn (carries session history), streaming.
- Tool web source: **sequential cascade** — SerpAPI first; if it returns
  nothing, fall back to the browser search server; first non-empty wins, stop.
- The `web_search` retarget is **global** (affects the auto-router's tool path
  too, not only the Tool tab) — consistent with "tool = web" everywhere.

## Non-goals (YAGNI)

- No new DB tables — all three surfaces reuse chat sessions (as search already
  does).
- No change to the Assistant auto-router logic (its tool path merely inherits
  the new web `web_search`).
- No parallel racing of providers — sequential cascade only.
- No new web-search provider infrastructure — reuse existing serpapi + browser
  primitives.
- Chat does not fall back to the OpenAI llm — local model only.

## Architecture

### Chat backend (new endpoint)

Add `POST /chat/send-chat-message` to
`src/internal/servers/query_and_chat/chat_backend.py`, mirroring
`tool_backend.py`:

1. Read `manager`/`tokenizer` from `app.state.search_agent_manager` /
   `search_agent_tokenizer`. If either is `None` → **HTTP 400** with the shared
   `NO_LOCAL_MODEL_MESSAGE` (from `tool_agent_runner.py`).
2. Resolve/create the session; load bounded history (before persisting the new
   user turn); persist the user turn.
3. Call a new module-level `_run_plain_chat(message, *, manager, tokenizer,
   history, on_turn=None)` that builds `PlainGenerationLoop` and runs it over
   `history + [{"role":"user","content":message}]` with
   `sampling_params={"temperature":0.7,"max_tokens":512}`, returning
   `output.final_answer`.
4. Stream SSE (`text/event-stream`): `PlainGenerationLoop` has no tools, so the
   stream emits `answer` then `done` (and `error` on failure). `stream:false`
   returns `{ session_id, answer }`.
5. Persist the assistant turn on success (not on failure).

`_run_plain_chat` lives in a neutral module to avoid the same circular-import
issue the tool runner hit. Place it in a new
`src/internal/servers/web/plain_chat_runner.py` (parallel to
`tool_agent_runner.py`), importing `PlainGenerationLoop` from
`src.agents.generation`. `chat_backend.py` imports it function-locally inside
the endpoint (same pattern `tool_backend.py` uses).

Request/response models added to `query_and_chat/models.py`:

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

### Tool → real web cascade

Add `web_cascade_search` to `src/tools/search.py` (a `search_fn`-compatible
coroutine matching the `MultiQueryWebSearchTool` `search_fn` signature:
`(query, *, provider, search_url, page_size, timeout_seconds) -> list[SearchPage]`):

1. If `SERP_API_KEY` is set: `pages = await serpapi_search(query, …)`. If
   `pages` is non-empty → return them.
2. Else / if empty: if a browser search URL is configured, call the browser
   search server for `query` and return its pages.
3. Else: return `[]` (logged warning; the agent answers without web).

The browser leg reuses the existing browser-search HTTP path (the same
`browser_search_url` the web app's `_run_direct_search` uses). The cascade
function takes the browser URL and serp key from env/config at construction so
it stays testable (inject a fake fetcher in tests).

Seed the tool with the cascade: in `src/tools/knowledge_base.py`, change the
`MultiQueryWebSearchTool` seed from `provider="retrieval"` to
`search_fn=web_cascade_search` (provider becomes irrelevant when a `search_fn`
is supplied). Everything else in `tool_knowledge_base` stays as-is. This makes
`web_search` genuinely fetch the web for both `/tool/send-tool-message` and the
`/api/agent` tool path.

`_run_tool_agent`'s injected `build_search_tool(provider="retrieval")` (the
`search` tool) is left as retrieval — `search` is the corpus tool; `web_search`
is the web tool. Only `web_search` is retargeted.

### Search tab (no backend change)

`/search/send-search-message` already returns docs. The frontend adds a client
+ view; nothing changes server-side.

## Frontend

- Extend the header switcher from two options to four: `Assistant | Search |
  Chat | Tool Agent`. `surface` state becomes
  `"assistant" | "search" | "chat" | "tool"`, default `"assistant"`. Each
  explicit surface renders its own view in place of the assistant
  composer/results block; the Assistant surface is unchanged.
- New `web/src/components/SearchView.tsx` — own composer, calls
  `sendSearchMessage`, renders returned docs via the existing `SourceGrid`
  (no answer panel).
- New `web/src/components/ChatView.tsx` — own composer, consumes the
  `sendChatMessage` SSE stream, renders the streamed answer + a running
  transcript; no-model banner on a 400 (`NO_LOCAL_MODEL`), like `ToolAgentView`.
- New API clients in `web/src/api.ts`:
  - `sendSearchMessage(body): Promise<SearchFullResponse>` (JSON; posts
    `{search_query, num_hits, run_query_expansion:false, stream:false}`).
  - `sendChatMessage(body, init?): AsyncGenerator<ChatStreamEvent>` (SSE,
    mirrors `sendToolMessage`; throws `Error("NO_LOCAL_MODEL")` on 400).
- New types in `web/src/types.ts`: `ChatStreamEvent` (`answer` | `done` |
  `error`), `SendChatMessageBody`, `SendSearchMessageBody`, and reuse of the
  existing search doc types.

## Error handling

- No local model → 400 on `/chat/send-chat-message` (search is model-free;
  tool already handles it). Frontend shows a no-model banner.
- Chat/search/tool loop exception mid-stream → `error` SSE event, stream closes
  cleanly, assistant turn not persisted.
- Web cascade: serpapi error → treated as empty → browser fallback; browser
  error or unconfigured → empty results, logged, agent continues.

## Testing

Backend (`tests/unit/`):
- `test_chat_backend.py`: `send-chat-message` streaming happy path (fake
  manager/tokenizer + monkeypatched `_run_plain_chat`) asserting `answer` →
  `done`; no-model 400 path; `stream:false` JSON shape.
- `test_web_cascade_search` (in the tools tests): serpapi-hit returns without
  calling browser; serpapi-empty falls back to browser; neither configured →
  `[]`. Inject fake serp/browser fetchers.
- Regression: `web_search` tool, when seeded via `tool_knowledge_base`, now
  carries the cascade `search_fn` (assert it no longer routes to retrieval).

Frontend (`web/src/components/__tests__/`):
- `ChatView` renders a streamed answer and shows the no-model banner on
  `NO_LOCAL_MODEL`.
- `SearchView` renders returned docs from a mocked `sendSearchMessage`.
- Switcher toggles all four surfaces; each renders its own view.

## Files touched

New:
- `src/internal/servers/web/plain_chat_runner.py`
- `web/src/components/SearchView.tsx`, `web/src/components/ChatView.tsx`
- tests (backend + frontend)

Modified:
- `src/internal/servers/query_and_chat/chat_backend.py` — new endpoint.
- `src/internal/servers/query_and_chat/models.py` — chat request/response models.
- `src/tools/search.py` — `web_cascade_search`.
- `src/tools/knowledge_base.py` — seed `web_search` with the cascade.
- `web/src/App.tsx` — four-tab switcher + mount Search/Chat views.
- `web/src/api.ts`, `web/src/types.ts` — new clients + types.
- `docs/tool-engine.md` / `docs/chat-engine.md` — document the direct surfaces.
