# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Project: Agentic Search

### Setup

```bash
pip install -e .          # one-time; makes src importable as a package
pip install -r requirements.txt
```

### Running the 3-process local stack

```bash
# Terminal 1 — retrieval server (demo TF-IDF, port 8001)
python3 -m src.internal.servers.retrieval.demo --corpus_path data/corpus.jsonl

# Terminal 1 alt — hybrid: RRF-fused dense e5 + sparse TF-IDF, drop-in for demo
python3 -m src.internal.servers.retrieval.hybrid --corpus_path data/corpus.jsonl
# add --no-dense to force TF-IDF only (skips the e5 model download)
# dense setup / "Dense leg unavailable" troubleshooting: docs/hybrid-dense-setup.md

# Optional — cross-encoder reranker (Terminal 1b). Then set the env on the web
# backend and restart it so retrieved docs are reranked before display:
python3 -m src.internal.servers.retrieval.rerank --port 8002
# web backend env: AGENTIC_SEARCH_RERANK_URL=http://localhost:8002/rerank

# Terminal 2 — web backend (port 7860)
PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860

# Terminal 3 — frontend dev server (port 5173)
cd web && npm install && npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api/*` to port 7860 during development.
For production, `npm run build` produces `web/dist`; the FastAPI app serves it automatically when the bundle exists.

### Tests

```bash
pytest                           # unit + regression (default)
pytest tests/unit/test_agent_loop.py -v   # single test file
pytest tests/integration/        # integration tests — requires live Postgres/Weaviate/Redis stack
```

### Linting

```bash
ruff check . --fix && ruff format .
```

Frontend type-check: `cd web && npm run typecheck`

### Agent CLI

```bash
# Local inference (Apple Silicon — MPS is ~50x faster than CPU)
python3 -m examples.run_agentic_search \
  --mode single --question "What is FAISS?" \
  --model Qwen/Qwen2.5-1.5B-Instruct --local --device mps --allow_unsafe_mps \
  --allow_remote_model_downloads

# Server-backed search mode
python3 -m examples.run_agentic_search \
  --mode search --question "Compare dense and sparse retrieval" \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --vllm_url http://localhost:8080 --search_url http://localhost:8001/retrieve
```

Modes: `single` (PlainGenerationLoop), `search` (SearchAgentLoop), `tool` (ToolAgentLoop).

---

### Architecture

The system has three layers that run as separate processes:

**1. Retrieval servers**
Multiple interchangeable backends behind the same `/retrieve` API.

Retrieval (`src/internal/servers/retrieval/`):
- `demo.py` — TF-IDF over a local corpus.jsonl, no Java required
- `hybrid.py` — RRF-fused dense (e5) + sparse TF-IDF; Java-free, FAISS-free
- `server.py` — full `RetrievalService` (BM25 / dense via FAISS / hybrid, env-configured via `RETRIEVAL_BACKEND`) with per-mode + admin endpoints
- `rerank.py` — standalone cross-encoder reranker

Web search (`src/internal/servers/web_search/`):
- `google.py` — Google Custom Search API proxy (requires `GOOGLE_API_KEY` + `GOOGLE_CSE_ID`)
- `serp.py` — SerpAPI proxy (requires `SERP_API_KEY`)
- `browser.py` — playwright-cli browser automation; no API key needed, slower (~5–10s/query)

**2. Web backend** (`src/internal/servers/web/app.py`)
FastAPI app that exposes `POST /api/agent`. On each request it:
1. Calls `answer_with_retrieval` from `src/context/` which fetches from the retrieval server
2. Runs the agent loop from `src/agents/`
3. Persists chat state to `AgenticSearchStore` (SQLite via `src/internal/db/`)
4. Returns streaming JSON with citations and source cards

The backend also mounts a large set of admin/enterprise routers (auth, SCIM, billing, connectors, OAuth, etc.) all registered in `create_web_app()`.

**3. Frontend** (`web/`)
React 19 + Vite + TypeScript. No component library — custom components only. Proxies `/api/*` to the FastAPI server on port 7860 in dev mode.

**Agent loops** (`src/agents/`, grouped into `core/` `generation/` `search/` `tool/`)
- `core/base.py` — `AgentLoopBase` with shared state/tool dispatch
- `generation/plain.py` — `PlainGenerationLoop` (no retrieval)
- `search/search.py` — `SearchAgentLoop` (retrieval-grounded, multi-turn)
- `search/agentic_rag.py` — `AgenticRAGLoop` (iterative hybrid retrieval + sufficiency check)
- `tool/tool_calling.py` — `ToolAgentLoop` (generic function calling)
- Agent loops are selected via the registry (`get_registered_agent_loop` + `resolve_agent_name` in `src/agents/core/base.py`).

**Context pipeline** (`src/context/`)
`answer_with_retrieval` wires retrieval → context-building → prompting → LLM call. `preprocessing/` holds access filters applied before retrieval.

**Retrieval internals** (`src/internal/document_index/`, `src/internal/retrieval/`)
Low-level chunking, embedding, FAISS index building, sparse BM25, and hybrid retrieval. Used both by the retrieval servers and the indexing pipeline.

**Indexing pipeline** (`src/internal/servers/backgroundworker/`)
Async workers: `light_worker` (polling/scheduling), `heavy_worker` (embedding + indexing), `beat_worker` (cron), `monitoring_worker`. Connectors (`src/internal/connectors/`) feed documents into this pipeline.

**Configuration** (`src/internal/configs/`)
Typed dataclasses loaded from environment variables. Key env vars:
- `AGENTIC_SEARCH_RETRIEVAL_PORT` (default 8000)
- `AGENTIC_SEARCH_WEB_PORT` (default 8080 in config; run on 7860 by convention)
- `AGENTIC_SEARCH_WEB_DB_PATH` (default `:memory:`)

**Training** (`src/training/`)
SFT data builders, PPO/GRPO reward helpers — standalone scripts for fine-tuning, not part of the serving stack.
