# Agentic Framework — Full Stack Polish

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Polish every feature area of the Agentic Search platform so each works reliably end-to-end — from retrieval and indexing through agent loops, tool use, MCP, chat orchestration, and admin observability.

**Architecture:** The existing agentic workflow is the foundation — `SearchAgentLoop` (XML ReAct: think/search/fetch/answer), `ToolAgentLoop` (function calling: Hermes/Llama3/JSON), and `AgenticRAGLoop` (multi-hop RAG). This spec does not change that architecture; it wires, verifies, and polishes each layer so the full stack runs as a coherent agentic framework.

**Tech Stack:** Python 3.12, FastAPI, SQLite (AgenticSearchStore), React 19 + Vite, aiohttp, transformers, mlx-lm, fastmcp, FAISS, pyserini, SerpAPI.

---

## Feature Areas

### 1. Search Workflow (`search_tool`, `hybrid_search`)

**Status:** Backend implemented. Source provider routing and response formatting need verification.

**Files:**
- `src/internal/servers/web/app.py` — `_run_direct_search`, `_run_hybrid_search`, `_search_only_answer`
- `src/internal/servers/web_search/serp.py`, `google.py`, `browser.py`
- `tests/unit/servers/test_agent_endpoint.py` (create)

**Changes:**
- Verify `source_provider` routing works for `retrieval`, `serpapi`, `browser`, `all`
- `_search_only_answer` returns a flat string — add `executed_queries` to the response metadata
- Add smoke test: POST `mode=search_tool` to `/api/agent` against demo retrieval server, assert non-empty `documents`
- Add smoke test: POST `mode=hybrid_search`, assert `executed_queries` in DB metadata

---

### 2. Chat Workflow (`chat_once`, `chat_loop`)

**Status:** Backend implemented. `rounds_used` persisted in DB metadata but not surfaced in UI.

**Files:**
- `src/internal/servers/web/app.py` — `chat_once` branch, `chat_loop` branch
- `web/src/components/SessionTimeline.tsx`
- `web/src/types.ts` — `ChatMessageView`
- `tests/unit/servers/test_agent_endpoint.py`

**Changes:**
- `chat_loop`: expose `rounds_used` from DB metadata in `ChatMessageView` response so UI can display it
- `SessionTimeline.tsx`: render a small badge "N rounds" next to `chat_loop` assistant messages
- Add unit test: `chat_loop` with 2-turn history returns `rounds_used >= 1`
- Add unit test: `chat_once` does not carry `rounds_used`

---

### 3. Search Agent (`search_agent`)

**Status:** Fixed in this branch (transformers 5.x `apply_chat_template` compat, aiohttp session cleanup). Needs unit test coverage.

**Files:**
- `src/agents/base.py` — `_build_prompt_ids_sync`
- `tests/unit/test_agent_loop.py`

**Changes:**
- Add unit test for `_build_prompt_ids_sync` covering the `tokenize=False` + explicit encode path
- Add unit test: `aclose()` is called after `OpenAIServerManager.generate()` completes
- Verify end-to-end in web UI: set `SEARCH_AGENT_MODEL`, POST `mode=search_agent`, assert answer returned

---

### 4. Tool Agent (`tool_agent`) — new mode

**Status:** `ToolAgentLoop` fully implemented but not wired into the web endpoint. No frontend mode entry.

**Files:**
- `web/src/types.ts`
- `web/src/components/SearchComposer.tsx`
- `src/internal/configs/app_configs.py`
- `src/internal/servers/web/app.py`
- `tests/unit/test_configs.py`
- `tests/unit/servers/test_agent_endpoint.py`

**Changes:**

`web/src/types.ts` — add `"tool_agent"` to `AgentMode`:
```ts
export type AgentMode = "search_tool" | "hybrid_search" | "chat_once" | "chat_loop" | "search_agent" | "tool_agent";
```

`web/src/components/SearchComposer.tsx` — add to `MODE_OPTIONS`:
```ts
{ value: "tool_agent", label: "Tool Agent (Function Calling)" },
```

`src/internal/configs/app_configs.py` — add field:
```python
tool_agent_parser: str = "json"   # "json" | "hermes" | "llama3"
```
Read from env: `get_env_str(source, "TOOL_AGENT_PARSER", "json")`

`src/internal/servers/web/app.py` — add branch in `run_agent()` after `search_agent` block:
```python
if mode == "tool_agent":
    from src.agents.tool_calling import ToolAgentLoop, ToolAgentLoopConfig
    from src.tools import build_search_tool
    from src.tools.registry import tool_registry

    manager = getattr(http_request.app.state, "search_agent_manager", None)
    tokenizer = getattr(http_request.app.state, "search_agent_tokenizer", None)
    if manager is None or tokenizer is None:
        raise HTTPException(
            status_code=400,
            detail="tool_agent mode requires SEARCH_AGENT_MODEL or SEARCH_AGENT_SERVER_URL",
        )
    tools = [build_search_tool(search_url=search_url)] + tool_registry.list_tools()
    loop = ToolAgentLoop(
        tokenizer=tokenizer,
        server_manager=manager,
        tools=tools,
        config=ToolAgentLoopConfig(tool_parser_format=resolved.tool_agent_parser),
    )
    output = await loop.run(
        [{"role": "user", "content": query}],
        sampling_params={"temperature": 0.0, "max_tokens": 512},
    )
    answer = output.final_answer or next(
        (m["content"] for m in reversed(output.trajectory_messages) if m.get("role") == "assistant"),
        "",
    )
    db.add_chat_message(
        session_id, role="assistant", content=answer,
        metadata={"mode": mode, "hooks": hook_metadata, "num_turns": output.num_turns},
    )
    messages = [ChatMessageView(role=m.role, content=m.content) for m in db.list_chat_messages(session_id)]
    return AgentExperienceResponse(
        session_id=session_id, answer=answer,
        citations=[], documents=[], messages=messages, hook_metadata=hook_metadata,
    )
```

**Extensibility:** Any tool added via `@tool`, OpenAPI import via ToolPanel, or `tool_registry.register()` is automatically available in `tool_agent` mode — no code change required.

---

### 5. Web Search Servers

**Status:** Three servers exist (`serp.py`, `google.py`, `browser.py`). Routing in `_run_direct_search` needs verification.

**Files:**
- `src/internal/servers/web_search/serp.py`
- `src/internal/servers/web_search/google.py`
- `src/internal/servers/web_search/browser.py`
- `bin/run_web_stack.sh`

**Changes:**
- Verify `SERP_API_KEY` is read and forwarded correctly from `.env` in `bin/run_web_stack.sh`
- Verify `google.py` returns results when `GOOGLE_API_KEY` + `GOOGLE_CSE_ID` are set
- Verify `browser.py` starts correctly with playwright installed
- Add health check assertion to each server's startup

---

### 6. Document Indexing & Retrieval

**Status:** Index builders and retrieval servers exist. Demo server (`demo.py`) works locally. Background workers need verification.

**Files:**
- `src/internal/document_index/index_builder.py`
- `src/internal/servers/retrieval/demo.py`, `retrieval_server.py`, `hybrid_rerank.py`
- `src/internal/servers/backgroundworker/`

**Changes:**
- Verify `demo.py` returns results for a sample corpus (`data/corpus.jsonl`)
- Verify `index_builder.py` produces a loadable FAISS index end-to-end
- Document the exact commands in CLAUDE.md (already there — verify accuracy)
- No new code unless a specific breakage is found during verification

---

### 7. Connectors

**Status:** Connector CRUD endpoints exist. `WebConnector`, `RSSConnector`, `LocalFileConnector` implemented.

**Files:**
- `src/internal/connectors/`
- `src/internal/servers/connectors/`
- `web/src/components/ConnectorPanel.tsx`

**Changes:**
- Verify ConnectorPanel can create and list connectors via the admin API
- Verify `WebConnector` fetches and indexes a public URL
- Add a connector smoke test: create → trigger ingest → assert document indexed

---

### 8. Tool Use & Tool Registry

**Status:** `tool_registry` singleton exists. `ToolPanel` can list, register (OpenAPI), and invoke tools. `sync_tool_to_mcp` bridges to MCP.

**Files:**
- `src/tools/registry.py`
- `src/internal/servers/tools/api.py`
- `web/src/components/ToolPanel.tsx`

**Changes:**
- Verify `POST /admin/tools/openapi` registers tools from a sample OpenAPI spec
- Verify `POST /admin/tools/{name}/invoke` executes a registered tool
- Verify `sync_tool_to_mcp` correctly mirrors a new tool to the MCP server
- These are verification tasks — no new code unless breakage found

---

### 9. Chat Orchestration

**Status:** `build_chat_turn`, `run_llm_loop`, `DynamicCitationProcessor`, `compress_chat_history` all exist.

**Files:**
- `src/internal/chat/process_message.py`
- `src/internal/chat/llm_loop.py`
- `src/internal/chat/citation_processor.py`

**Changes:**
- Verify citation markers are extracted and returned in `AgentExperienceResponse.citations`
- Verify `compress_chat_history` triggers when history exceeds token budget
- No new code unless a specific breakage is found

---

### 10. MCP Server

**Status:** `fastmcp` server exists with search/chat/research/dynamic tools. Not always started with the web stack.

**Files:**
- `src/internal/mcp_server/api.py`
- `src/internal/mcp_server/tools/search.py`, `chat.py`, `research.py`, `dynamic.py`
- `bin/run_web_stack.sh`

**Changes:**
- Add optional MCP server startup to `bin/run_web_stack.sh` (gated on `MCP_ENABLED=1`)
- Verify `ask_agentic_search` tool calls `SearchAgentLoop` end-to-end
- Verify `sync_tool_to_mcp` works after a tool is registered via ToolPanel
- Add a README note: run `pip install -e ".[mcp]"` before starting MCP server

---

### 11. Admin & Observability

**Status:** Health, analytics, hooks, rate limits, license endpoints all exist.

**Files:**
- `src/internal/observability/`
- `src/internal/servers/analytics/`

**Changes:**
- Verify `GET /health` returns `{"status": "ok"}` with all three processes running
- Verify `GET /analytics/query` returns data after a few agent calls
- Verify `build_admin_surface_summary` reflects `search_agent` and `tool_agent` mode usage
- No new code unless a specific breakage is found

---

## Build Order

Execute in this order — each area is a prerequisite for the ones below it:

```
1. Search workflow (smoke tests)
2. Chat workflow (rounds_used UI)
3. Search agent (unit tests)
4. Tool agent (new mode wiring)        ← primary new code
5. Web search servers (verification)
6. Document indexing (verification)
7. Connectors (verification)
8. Tool use & registry (verification)
9. Chat orchestration (verification)
10. MCP server (optional startup)
11. Admin & observability (verification)
```

Areas 5–11 are mostly verification tasks — run the thing, observe output, fix what's broken. The only net-new code is in areas 1–4.

---

## Configuration Reference

New env vars added by this spec:

| Var | Default | Description |
|-----|---------|-------------|
| `TOOL_AGENT_PARSER` | `json` | Tool call parser for `tool_agent` mode: `json`, `hermes`, `llama3` |

Existing vars required to enable agent modes:

| Var | Required for |
|-----|-------------|
| `SEARCH_AGENT_MODEL` | `search_agent` and `tool_agent` modes (local inference) |
| `SEARCH_AGENT_SERVER_URL` | `search_agent` and `tool_agent` modes (OpenAI-compatible server) |
| `SEARCH_AGENT_DEVICE` | `search_agent` and `tool_agent` local inference device (default: `mps`) |
| `SERP_API_KEY` | `serpapi` source provider and `bin/run_bamboogle_eval.sh` |

---

## Out of Scope

- Merging `SearchAgentLoop` and `ToolAgentLoop` into a single unified loop
- Streaming token-by-token output for agent modes
- Multi-agent orchestration (agents calling agents)
- New retrieval backends beyond what already exists
- RL training pipeline changes (GRPO/PPO remain as-is)
