# Search Agent UI Integration — Design Spec

**Goal:** Wire `SearchAgentLoop` into the existing web UI so a user can type a question in the browser and get a grounded answer backed by local MPS inference — the same agent used in training and eval.

**Status:** Approved, pending implementation plan.

---

## Problem

`SearchAgentLoop` (the agent trained via GRPO and evaluated on Bamboogle) is completely disconnected from the web UI. The UI uses `AgenticRAGLoop` and four separate modes (`chat_once`, `chat_loop`, `search_tool`, `hybrid_search`). Training and serving are two parallel worlds.

---

## Approach

Add `"search_agent"` as a fifth mode to the existing `POST /api/agent` endpoint. The model loads once at app startup via `LocalServerManager` and is stored in `app.state`. Requests with `mode=search_agent` await `SearchAgentLoop.run()` directly — no thread executor needed since the method is already async. All other modes are untouched.

---

## Architecture

```
Browser
  │  POST /api/agent { mode: "search_agent", query: "..." }
  ▼
app.py  ──── (startup) ────►  LocalServerManager
  │                           loads SEARCH_AGENT_MODEL on SEARCH_AGENT_DEVICE once
  │
  ├─ chat_once / chat_loop / search_tool / hybrid_search  (unchanged)
  │
  └─ search_agent (new)
      → SearchAgentLoop.run()   (async, awaited directly)
      → SerpAPI retrieval server (port 8000, existing)
      → AgentLoopOutput { final_answer, trajectory_messages, context }
      → mapped to AgentExperienceResponse { answer, citations, documents, messages }
      → browser renders answer + source cards (existing components, unchanged)
```

---

## Files Changed

| File | Change |
|------|--------|
| `src/internal/servers/web/app.py` | Extend lifespan to init `LocalServerManager`; add `"search_agent"` branch in `run_agent()` |
| `src/internal/configs/__init__.py` | Add `search_agent_model: str \| None` and `search_agent_device: str` to `AppSettings` |
| `web/src/types.ts` | Add `"search_agent"` to `AgentMode` union |
| `web/src/App.tsx` | Add "Search Agent" option to mode selector |
| `web/src/components/SearchComposer.tsx` | Show retrieval URL field when mode is `"search_agent"` |
| `bin/run_web_stack.sh` | New script: starts SerpAPI server + web backend + Vite dev server |

No new routers, DB tables, or UI components.

---

## Output Mapping

`AgentLoopOutput` → `AgentExperienceResponse`:

| `AgentLoopOutput` | `AgentExperienceResponse` |
|---|---|
| `final_answer` | `answer` |
| `context.documents[*].citation` | `citations` |
| `context.documents` | `documents` (via existing `_document_view()`) |
| `trajectory_messages` | `messages` |

`AgentLoopOutput.context` already holds `ContextDocument` objects — the same type all other modes use — so `_document_view()` and `_dedupe_documents()` work without changes.

---

## Configuration

Two new optional env vars (add to `.env`):

```
SEARCH_AGENT_MODEL=Qwen/Qwen2.5-0.5B-Instruct   # 8 GB RAM; use 1.5B-Instruct on 16 GB+
SEARCH_AGENT_DEVICE=mps                           # default: mps
```

Read into `AppSettings`:

```python
search_agent_model: str | None = None   # None = mode disabled, returns 400
search_agent_device: str = "mps"
```

If `SEARCH_AGENT_MODEL` is unset, `mode=search_agent` returns HTTP 400 with a clear message. All other modes work regardless.

---

## Startup Behaviour

The existing lifespan context manager (line 273 of `app.py`) is extended:

```python
if settings.search_agent_model:
    app.state.search_agent = LocalServerManager(
        model_path=settings.search_agent_model,
        device=settings.search_agent_device,
        allow_unsafe_mps=True,
        local_files_only=True,
    )
else:
    app.state.search_agent = None
```

Model load takes ~10–30 s on first startup. Subsequent requests reuse the loaded model with no reload cost.

---

## Dev Startup Script

`bin/run_web_stack.sh` replaces the current 3-terminal workflow from CLAUDE.md:

```bash
#!/usr/bin/env bash
# Reads SERP_API_KEY and SEARCH_AGENT_MODEL from .env
# Usage: bin/run_web_stack.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }

python3 -m src.internal.servers.web_search.serp --port 8000 &
SERP_PID=$!
trap 'kill $SERP_PID 2>/dev/null' EXIT

SEARCH_AGENT_MODEL="${SEARCH_AGENT_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}" \
SEARCH_AGENT_DEVICE="${SEARCH_AGENT_DEVICE:-mps}" \
PYTHONPATH="$ROOT/src" \
uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860 &
WEB_PID=$!
trap 'kill $SERP_PID $WEB_PID 2>/dev/null' EXIT

cd "$ROOT/web" && npm run dev
```

Open `http://127.0.0.1:5173`, select **Search Agent** mode, ask a question.

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| `SEARCH_AGENT_MODEL` not set | HTTP 400: "Search agent model not configured — set SEARCH_AGENT_MODEL in .env" |
| Model file not found locally | Startup fails with clear error (propagated from `LocalServerManager`) |
| SerpAPI key missing | Retrieval server fails to start; web backend logs connection error per request |
| Other modes while model is loading | Unaffected — `app.state.search_agent` not touched by other mode branches |

---

## Out of Scope

- vLLM server-backed mode (future)
- Training data flywheel / feedback collection (future)
- Streaming token-by-token output (future — current modes return complete responses)
- Multiple concurrent search-agent requests (MPS serializes; acceptable for demo)
