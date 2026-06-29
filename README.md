# Agentic Search

A retrieval-backed agent platform for multi-turn search, RAG, and RL training. Built around a FastAPI backend, interchangeable retrieval servers, and an async agent loop that supports dense/sparse hybrid retrieval, tool calling, and streaming chat.

🔍 **Agentic RAG** — Multi-hop retrieval with query decomposition, HyDE, hybrid reranking, and citation-grounded synthesis via `AgenticRAGLoop`.

🤖 **Custom Agents** — Compose agents from instructions, knowledge sources, tools, and memory; backed by `SearchAgentLoop` or `ToolAgentLoop`.

🌍 **Web Search** — Live retrieval via Google Custom Search, SerpAPI, and playwright-cli browser automation — all behind the same `/retrieve` API.

📚 **Document Indexing** — Chunk, embed, and index documents into FAISS or BM25; async background workers handle ingestion at scale.

🔗 **Connectors** — Pull content from local files, Google Drive, Slack, Confluence, GitHub, Jira, SharePoint, Salesforce, Zendesk, and Notion.

🛠️ **Tool Use** — Register Python callables or OpenAPI 3.x schemas as tools; `ToolAgentLoop` handles dispatch and structured output.

💬 **Chat Orchestration** — Streaming multi-turn chat with citation extraction, tool dispatch, context compression, and persisted sessions.

🧭 **Intent Routing** — Auto-classifies every query as `search`, `chat`, or `tool`; dispatches to the right agent loop with no configuration; RAG-Fusion multi-source aggregation in tool mode.

🖥️ **React Frontend** — Streaming chat UI with live SSE progress log, Markdown rendering, `[D1]`-format citation anchor links, per-card source expand/collapse, tool call trace panel, and intent-adaptive layout.

🧠 **RL Training** — GRPO/PPO training with composite shaped rewards; `SearchAgentGRPOTrainer` runs real agent-loop rollouts so all reward components fire during training.

📐 **Bamboogle Evaluation** — Benchmark `SearchAgentLoop` on two-hop QA with exact-match, contains-match, and shaped reward metrics; Apple Silicon (`--device mps`) supported out of the box.

🔌 **MCP Server** — Expose search, retrieval, and RAG as Model Context Protocol tools so any MCP-compatible LLM client (Claude Desktop, etc.) can query your knowledge base directly.

📊 **Admin & Observability** — Health, analytics, rate limits, hooks, billing, SCIM provisioning, and license state via the FastAPI admin API.


[![Architecture](agentic-search-grpo-architecture.png)](https://htmlpreview.github.io/?https://github.com/lingduoduo/Agentic-Search-GRPO/blob/main/agentic-search-grpo-architecture.html)

*Click to open the interactive version.*


| Feature | Key modules |
|---------|-------------|
| 🔍 Agentic RAG | `src/agents/agentic_rag.py`, `src/context/query_enhancer.py`, `src/internal/servers/retrieval/hybrid_rerank.py` |
| 🤖 Custom Agents | `src/agents/search.py`, `src/agents/components/loop_controller.py`, `src/agents/tool_calling.py`, `src/agents/base.py` |
| 🌍 Web Search | `src/internal/servers/web_search/google.py`, `src/internal/servers/web_search/serp.py`, `src/internal/servers/web_search/browser.py` |
| 📚 Document Indexing | `src/internal/document_index/`, `src/internal/servers/backgroundworker/` |
| 🔗 Connectors | `src/internal/connectors/`, `src/internal/servers/connectors/`, `src/internal/servers/oauth/` |
| 🛠️ Tool Use | `src/tools/base.py`, `src/tools/api.py`, `src/tools/search.py`, `src/agents/tool_calling.py` |
| 💬 Chat Orchestration | `src/internal/chat/process_message.py`, `src/internal/chat/llm_loop.py`, `src/internal/chat/citation_processor.py`, `src/internal/chat/compression.py` |
| 🧭 Intent Routing | `src/internal/servers/web/app.py` (`_run_auto_routed`), `src/context/` |
| 🖥️ React Frontend | `web/src/App.tsx`, `web/src/components/`, `web/src/styles.css` |
| 🧠 RL Training | `src/training/reward.py`, `src/training/grpo.py`, `src/training/ppo/search_agent_grpo_trainer.py` |
| 📐 Bamboogle Evaluation | `src/training/eval/bamboogle.py`, `examples/run_bamboogle_eval.py`, `bin/run_bamboogle_eval.sh` |
| 🔌 MCP Server | `src/internal/mcp_server/tools/`, `src/internal/mcp_server/resources/` |
| 📊 Admin & Observability | `src/internal/observability/`, `src/internal/servers/analytics/`, `src/internal/servers/reporting/`, `src/internal/servers/license/` |
| ⚡ Retrieval Optimization | `src/internal/retrieval/query_optimizer.py`, `src/internal/retrieval/bm25_tuner.py`, `src/internal/retrieval/index_optimizer.py`, `src/internal/retrieval/fusion_learner.py`, `src/internal/retrieval/result_cache.py` |
| 🏆 Reranking Optimization | `src/internal/retrieval/async_reranker.py`, `src/internal/retrieval/cached_reranker.py`, `src/internal/retrieval/two_stage_reranker.py`, `src/internal/retrieval/onnx_reranker.py`, `src/internal/retrieval/reranker_benchmark.py` |


## Contents

- [Repository Structure](#repository-structure)
- [Install](#install) · [Quick Start](#quick-start) · [Frontend](#frontend) · [Examples](#examples)
- [Intent Routing](#intent-routing) · [Features](#features) · [Agent Framework & Control Flow](#agent-framework--control-flow) · [Agentic RAG](#agentic-rag)
- **Retrieval:** [Retrieval Setup](#retrieval-setup) · [Neural Reranking](#neural-reranking) · [Retrieval Optimization](#retrieval-optimization) · [Query Transformation Optimization](#query-transformation-optimization) · [Routing & Query Construction](#routing--query-construction)
- **HTTP APIs:** [Retrieval Server API](#retrieval-server-api) · [Web Backend API](#web-backend-api) · [Chat & Session API](#chat--session-api)
- **Training & eval:** [Training](#training) · [Evaluation](#evaluation)
- **Ops:** [MCP Server](#mcp-server) · [API Health Checks](#api-health-checks) · [Configuration](#configuration) · [Tests](#tests) · [Notes](#notes)

---

## Repository Structure

```
src/
├── agents/                      # Agent loops (SearchAgentLoop, ToolAgentLoop, AgenticRAGLoop, …)
├── cli/                         # CLI query interface
├── context/                     # Retrieval-grounded context & prompt builders
├── model/                       # LLM generation, intent classifier, tensor helpers
├── shared_configs/              # Shared configuration dataclasses
├── tools/                       # Tool schemas, search tools, OpenAPI tool registry
├── training/
│   ├── eval/                    # Benchmark evaluation (Bamboogle, …)
│   ├── ppo/                     # PPO core, LLMGRPOTrainer, SearchAgentGRPOTrainer
│   ├── data.py                  # Training dataset builders
│   ├── grpo.py                  # GRPO advantage helpers
│   ├── reward.py                # SearchRewardFunction
│   └── sft.py                   # SFT data pipeline
└── internal/
    ├── access/                  # Access control & ACL helpers
    ├── auth/                    # Authentication & authorization
    ├── cache/                   # In-memory cache backend (chat session state)
    ├── chat/                    # Chat pipeline (loop, steps, citations, compression)
    ├── configs/                 # Environment-based configuration (AppSettings)
    ├── connectors/              # Data source connectors
    ├── context/                 # Internal retrieval context helpers
    ├── db/                      # SQLite store (AgenticSearchStore)
    ├── document_index/          # Document index (FAISS / BM25)
    ├── feature_flags/           # Feature-flag providers (env, PostHog, composite)
    ├── file_store/              # In-memory chat file handling
    ├── hooks/                   # Outbound webhook execution
    ├── llm/                     # LLM provider integrations
    ├── mcp_server/              # MCP server (tools, resources, auth)
    ├── metrics/                 # Metrics collection helpers
    ├── natural_language_processing/  # NLP utilities
    ├── observability/           # Admin surface summary & health score
    ├── prompts/                 # Prompt templates
    ├── retrieval/               # Retrieval core: service, fusion, query transforms, routers
    ├── routing/                 # Routing layer: per-query router + 6 query constructors
    ├── search/                  # Search-vs-chat flow classification
    ├── tools/                   # Internal tool registry
    ├── utils/                   # License, encryption, telemetry utilities
    └── servers/
        ├── admin_surface/       # Admin summary endpoint
        ├── analytics/           # Usage analytics API
        ├── backgroundworker/    # Async workers (beat, docfetching, light, heavy, monitoring)
        ├── billing/             # Stripe billing proxy
        ├── connectors/          # Connector management endpoints
        ├── documents/           # Connector-credential pair management
        ├── enterprise_settings/ # Enterprise configuration endpoints
        ├── evals/               # Evaluation endpoints
        ├── features/            # Feature-flag endpoints
        ├── indexing/            # Indexing status & control endpoints
        ├── license/             # License validation & seat management
        ├── limits/              # Usage limit enforcement
        ├── middleware/          # License enforcement, tier gate, tenant tracking
        ├── oauth/               # OAuth 2.0 connector authorization
        ├── query_and_chat/      # Search and chat endpoints
        ├── query_history/       # Query history & export
        ├── reporting/           # Usage report ZIP generation
        ├── retrieval/           # Dense/sparse/rerank server entry points
        ├── scim/                # SCIM 2.0 user & group provisioning
        ├── settings/            # Settings endpoints
        ├── tenants/             # Multi-tenant provisioning & management
        ├── token_rate_limits/   # Per-user token rate limiting
        ├── user_group/          # Group management
        ├── users/               # User management
        ├── web/                 # FastAPI app assembly
        └── web_search/          # Web search servers (Google, SerpAPI, browser)
bin/                             # Shell helpers (eval, training data generation)
tests/                           # Unit and integration test suites
examples/                        # Runnable CLI examples
```

The FastAPI app is assembled in `src/internal/servers/web/app.py`. Every feature area is a self-contained router factory. `AgenticSearchStore` (SQLite) is the single persistence layer — no Postgres, Redis, or Celery required locally.


## Install

Requires Python 3.10+.

```bash
pip install -e .               # makes src importable as a package
pip install -r requirements.txt
```

For MCP server support:

```bash
pip install -e ".[mcp]"
```

For BM25 (pyserini), Java must be available on `PATH`. Set `JAVA_HOME` if needed.

Env vars — copy `.env.example` to `.env` (loaded automatically via `python-dotenv`):

```bash
# LLM provider (required for agent loops)
GEN_AI_MODEL_PROVIDER=openai       # openai | anthropic | ollama | litellm
GEN_AI_MODEL_VERSION=gpt-4o-mini
GEN_AI_API_KEY=...
GEN_AI_API_BASE=...                # optional override (e.g. http://localhost:11434/v1)

# Web search (pick one or more)
GOOGLE_API_KEY=...
GOOGLE_CSE_ID=...
SERP_API_KEY=...

# Optional
JAVA_HOME=/path/to/java            # for BM25 / pyserini
```


## Quick Start

Three processes, each in its own terminal:

**Retrieval service** — `http://localhost:8001`
```bash
python3 -m src.internal.servers.retrieval.demo --corpus_path data/corpus.jsonl
```

**Web API** — `http://localhost:7860`
```bash
PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

**Frontend** — `http://localhost:5173`
```bash
cd web && npm install && npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api/*` to the web API on port 7860.
For production, `npm run build` produces `web/dist`; the FastAPI app serves it automatically.

**Search Agent mode** (optional — local MPS inference)

The UI has a fifth mode "Search Agent (Local Model)" that runs `SearchAgentLoop` in-process.
To enable it, set `SEARCH_AGENT_MODEL` before starting the web API:

```bash
# 8 GB RAM
SEARCH_AGENT_MODEL=Qwen/Qwen2.5-0.5B-Instruct PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860

# 16 GB RAM (better quality)
SEARCH_AGENT_MODEL=Qwen/Qwen2.5-1.5B-Instruct PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

Or use `bin/run_web_stack.sh` which reads `SEARCH_AGENT_MODEL` from `.env` and starts all three processes in one command (~30–60s first response on MPS).


## Frontend

The `web/` directory contains a React 19 + Vite + TypeScript single-page app. It runs against the FastAPI backend at port 7860 and proxies `/api/*` through Vite in development.

```bash
cd web && npm install && npm run dev   # dev server at http://127.0.0.1:5173
cd web && npm run build                # production bundle → web/dist/ (served by FastAPI)
cd web && npm run typecheck            # TypeScript check
cd web && npm run test -- --run        # Vitest unit tests
```

### UI features

**Streaming answers** (`AnswerPanel.tsx` → `ProgressLog`) — every query streams over SSE; `streamAgent` (`web/src/api.ts`) drives the UI from the `progress` / `answer` / `done` events (full schema in the [SSE event table](#intent-routing)). While the agent runs, a live **Agent reasoning** log renders one row per turn (`⟳ Turn N · writing answer…` active, `✓ Turn N · <tool> · N docs` completed) and answer tokens stream in as markdown; on `done` the log collapses to a one-line summary (`✓ 3 turns`) with a **show reasoning ▸** toggle that re-expands the full trace. Backend side, each turn fires the `on_turn` callback (`OnTurnCallback`) → a `progress` event, while token / tool-call / citation packets originate from `AgentQueueManager` → `Emitter`. The **New** button (`handleNewSession`) aborts any in-flight request and clears answer / citations / documents / messages / intent; an in-flight turn is cancellable via the stop-signal fence.

**Markdown rendering** — Answers render via `react-markdown`: headings, bold/italic, inline code, code blocks, and ordered/unordered lists. Citation markers (`[D1]`, `[D2]`, …) become anchor links that scroll the page to the matching source card.

**Chat history** — Session timeline renders as a chat bubble layout: user messages right-aligned, assistant messages left-aligned. System messages are filtered out. Keys are stable against message prepend/removal.

**Source cards** (`SourceGrid.tsx`) — `SourceGrid` is a thin mapper over a controlled `SourceCard` (memoised, per-document, owning its own `expanded` / `copied` state). Each card renders one `SourceDocumentView` (`{ id, citation, title, content, url, score, metadata }`) and:
- collapses content to 3 lines by default (`source-content--clamped`); **show more ▾** / **show less ▴** toggles per card.
- a **⎘ copy** button copies the full content and flips to "copied ✓" for 1.5 s.
- carries `id="source-{citation}"` so `[D1]`-style anchor links from the answer scroll to it.
- color-codes the relevance score via `scoreColor()` (green ≥ 0.7, amber ≥ 0.4, orange > 0, grey for 0).
- tags the source provider with a colored pill via `SOURCE_COLORS` (Browser Retrieval, SerpAPI, Local Retrieval, All Active Sources; grey fallback).

Source cards are frontend-only (no dedicated backend endpoint): they are populated from the `documents` array of the `POST /api/agent` response (see [Web Backend API](#web-backend-api)); the retrieval server returns the same fields as `results[]` from `POST /search`. Inspect that backing data with:
```bash
curl -s -X POST http://localhost:7860/api/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "What is FAISS?", "top_k": 3}' \
  | python -c "import sys, json; [print(d['citation'], round(d['score'],2), d['title']) for d in json.load(sys.stdin)['documents']]"
# → [D1] 0.81 FAISS: A Library for Efficient Similarity Search ...
```

**Tool Call Trace Panel** — When the agent runs in `tool` mode, a panel below the answer shows every tool call: name, status (✓ / ✗), arguments as JSON, result summary (first 200 chars or "N items" for lists), and latency in ms. Failed calls render with a red border and the error message.

**Intent-adaptive layout** — `App.tsx` reads `response.intent` (set from the `done` SSE event via `setIntent`) and applies `intent-${intent}` to the `.results-layout` container; when `intent` is undefined no class is added and the layout falls back to the default single-column stack. The behaviour is **CSS-only** — `styles.css` rules consume the class to reflow the existing panels (no extra components), keyed off stable hooks `.answer-column`, `.sources-panel`, `.session-panel`, and `.tool-trace-panel`:

| Intent | `.results-layout` class | Layout |
|--------|--------|--------|
| `search` | `intent-search` | Single column; `.sources-panel` gets a highlighted border; `.session-panel` dimmed |
| `chat` | `intent-chat` | `.answer-column` + `.session-panel` side-by-side (≥720 px); `.sources-panel` full-width below |
| `tool` | `intent-tool` | `.tool-trace-panel` full-width hero; `.sources-panel` and `.session-panel` side-by-side below |
| narrow (≤720 px) | — | All intents fall back to a single-column grid stack |

The intent itself comes from the backend's routing decision — see the `response.intent` contract under [Web Backend API](#web-backend-api). No new endpoints back this feature; the layout is a pure function of that one field.

**Intent badge** (`AnswerPanel.tsx`) — a pill under the answer summarising what ran, derived from `response.intent` + counts: `Searched · 5 sources`, `Answered · 3 citations`, or `Used tools`. Hidden when the answer is empty or the intent is undefined.

**Example-query chips** (`SearchComposer.tsx`) — three chips under the search box, one per routing intent, that populate and run a representative query in a single click so the intent router can be exercised without knowing what triggers each path: 🔍 `find the onboarding checklist` (search), 💬 `explain how FAISS indexing works` (chat), 🛠 `summarize the latest sales figures and chart them` (tool). The chips are hidden while a request is in flight.

**Components** (`web/src/components/`) — each panel is a focused, independently tested unit:

| Component | What it does |
|-----------|--------------|
| `SearchComposer` | Single input box (no mode selector), per-intent example-query chips, source-provider / retrieval-URL / top-K controls, Cmd+Enter submit |
| `AnswerPanel` | Streamed markdown answer + intent badge + `[D1]` citation anchor links |
| `SourceGrid` | Expand/collapse source cards with copy-to-clipboard and citation `id` anchors |
| `SessionTimeline` | Chat-bubble history (user right, assistant left; system filtered) |
| `ToolCallTracePanel` | Per-tool-call trace (name, ✓/✗ status, JSON args, result summary, latency) for `tool` intent |
| `AdminOverview` | Single-call health snapshot — connectors, indexing, users, auth, models, tools, analytics with a composite health score |
| `AnalyticsDashboard` | Usage breakdowns by LLM, persona, and flow (`getAnalyticsBy*`) |
| `ConnectorPanel` | Lists configured connectors and their sync/index status |
| `QueryHistoryPanel` | Per-user query history with CSV export (`getQueryHistory`) |
| `ToolPanel` | Admin view of MCP/OpenAPI tools registered via `tool_registry` |

API client functions live in `web/src/api.ts`: `runAgent` / `streamAgent` (SSE), `createSession` / `getSession`, `getAdminSummary`, `getAnalyticsByLLM` / `getAnalyticsByPersona` / `getAnalyticsByFlow`, `getQueryHistory`, `getAuditSummary`, `submitFeedback`.

**Feedback loop (UI → fine-tuning)** — `submitFeedback(chatMessageId, isPositive, feedbackText?)` posts per-message like/dislike to `POST /chat/create-chat-message-feedback`, and session thumbs go to `POST /api/feedback`; `QueryHistoryPanel` can filter sessions by `feedback_type` (`like` / `dislike`). These ratings are exactly what `load_feedback_examples` reads back into [feedback-driven GRPO](#rl-training) — the human-feedback signal that fine-tunes the policy.


## Intent Routing

The backend auto-classifies every query and dispatches to the right agent without any configuration:

| Intent | Agent loop | Trigger |
|--------|-----------|---------|
| `search` | `SearchAgentLoop` | Query needs external retrieval (web or indexed docs) |
| `chat` | `PlainGenerationLoop` | Conversational follow-ups, definitions, open-ended questions |
| `tool` | `ToolAgentLoop` | Explicit tool use (`search_routing_tool`, custom tools) |

The router is `_run_auto_routed` in `src/internal/servers/web/app.py`. It runs an LLM-backed classifier (`classify_is_search_flow`) and falls back to `chat` on ambiguous input.

**RAG-Fusion in tool mode** — `search_routing_tool` aggregates results from all configured retrieval sources (local index, Google, SerpAPI) in a single call, deduplicates by URL, and returns a ranked list with `[D1]`/`[D2]` citation labels.

**SSE streaming with progress events** — All three agent paths emit SSE events:

| Event type | When emitted | Payload |
|------------|-------------|---------|
| `progress` | Each agent turn | `{type, turn, text}` |
| `answer` | Answer token chunks | `{type, text}` |
| `done` | Stream complete | `{type, session_id, citations, documents, intent, tool_calls}` |
| `error` | Unhandled exception | `{type, detail}` |

The `on_turn` callback (`OnTurnCallback` in `src/agents/base.py`) is the hook that feeds per-turn events into the SSE queue from inside the agent loop.


## Examples

**Agent CLI**

| Mode | Loop | Needs retrieval server | Use it for |
|------|------|------------------------|------------|
| `single` | `PlainGenerationLoop` | No | Local generation smoke tests |
| `search` | `SearchAgentLoop` | Yes | Multi-turn RAG, SFT, and RL traces |
| `tool` | `ToolAgentLoop` | Yes | Structured tool-calling experiments |

```bash
# single — no retrieval server needed (plain generation)
# Apple Silicon: use --device mps --allow_unsafe_mps for ~50x faster inference
python3 -m examples.run_agentic_search \
  --mode single --question "What is FAISS?" \
  --model Qwen/Qwen2.5-1.5B-Instruct --local --device mps --allow_unsafe_mps \
  --allow_remote_model_downloads

# single with retrieval server — small models (≤3B) use --mode single; search/tool require 7B+ to emit structured tags
python3 -m examples.run_agentic_search \
  --mode single --question "What is FAISS?" \
  --model Qwen/Qwen2.5-1.5B-Instruct --local --device mps --allow_unsafe_mps \
  --search_url http://localhost:8001/retrieve --allow_remote_model_downloads

# search — 3B is the Mac sweet spot (~6 GB unified memory); 7B needs 16 GB+ and will swap
python3 -m examples.run_agentic_search \
  --mode search --question "What is RAG?" \
  --model Qwen/Qwen2.5-3B-Instruct --local --device mps --allow_unsafe_mps \
  --search_url http://localhost:8001/retrieve --allow_remote_model_downloads

# search — server-backed, requires vLLM on :8080 and retrieval on :8001
python3 -m examples.run_agentic_search \
  --mode search --question "Compare dense and sparse retrieval" \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --vllm_url http://localhost:8080 --search_url http://localhost:8001/retrieve
```

**Bamboogle evaluation** (always requires retrieval server on :8001)

```bash
# Smoke test — local model, 1 example, full trace printed
python3 -m examples.run_bamboogle_eval \
  --model Qwen/Qwen2.5-3B-Instruct --local --device mps --allow_unsafe_mps \
  --search_url http://localhost:8001/retrieve --limit 1 --print_trace \
  --allow_remote_model_downloads

# Full benchmark — Apple Silicon, requires SERP_API_KEY in .env
bin/run_bamboogle_eval.sh --limit 125
```

**PPO/GRPO reward**

```bash
python3 -m examples.run_grpo_training_pipeline         # end-to-end reward + GRPO (no GPU)
```

**Dataset preparation**

```bash
# Search-QA parquet
python3 -m examples.prepare_search_qa_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets --dataset_config nq --local_dir data/nq_search

# Preview before writing
python3 -m examples.prepare_search_qa_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets --dataset_config nq \
  --splits test --max_examples 20 --preview --preview_rows 5

# RAG parquet from cached retrieval results
python3 -m examples.prepare_search_rag_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets --dataset_config nq \
  --corpus_path data/wiki-18.jsonl \
  --train_retrieval_cache data/nq_train_retrieval_cache.json \
  --test_retrieval_cache data/nq_test_retrieval_cache.json \
  --topk 3 --local_dir data/nq_rag
```

**Search pipeline with access filters** (no live model or retrieval server required)

```bash
python3 -m examples.run_search_pipeline
```


## Features

**Retrieval, Indexing & Search**
- **Hybrid + rerank** — dense (FAISS/E5) + sparse (BM25) RRF fusion with cross-encoder reranking in a single `/retrieve` endpoint
- **`QueryEnhancer`** (`src/context/query_enhancer.py`) — base query-transformation primitives: `decompose()` (2–4 sub-queries), `hyde()` (hypothetical answer), `step_back()` (broader reformulation), and `enhance()` which runs all three into a `QueryBundle`. Every method is fallback-safe — it returns the original query / `None` when no LLM is configured
- **`expand_keywords`** (`src/internal/servers/secondary_llm_flows/query_expansion.py`) — LLM keyword/synonym expansion for the BM25 leg; the `QT_KEYWORDS` branch of `QueryTransformPipeline`
- **`Reranker`** (`src/internal/retrieval/reranker.py`) — unified neural reranker supporting local cross-encoders (`BAAI/bge-reranker-v2-m3`, `cross-encoder/ms-marco-*`) and Cohere v3/v4 API; built via `Reranker.from_env()`; injected into `RetrievalService`; skipped when `RERANKER_PROVIDER` is unset; appends `+reranked` to `retrieval_mode`
- **`AsyncReranker`** (`src/internal/retrieval/async_reranker.py`) — wraps any reranker in a `ThreadPoolExecutor`; raises `RerankerTimeoutError` when `RERANKER_TIMEOUT_MS` is exceeded; exposes `arerank()` for async callers
- **`CachedReranker`** (`src/internal/retrieval/cached_reranker.py`) — Redis-backed score cache keyed on `sha256(query:sorted_doc_ids:k=top_k)`; `stats()` returns hits/misses/hit_rate; `from_env()` returns base unchanged when `RERANKER_CACHE_REDIS_URL` is unset
- **`TwoStageReranker`** (`src/internal/retrieval/two_stage_reranker.py`) — fast pre-filter over all N candidates, heavy scorer over top M; both legs independently wrapped; enabled via `RERANKER_TWO_STAGE=true`
- **`ONNXReranker`** (`src/internal/retrieval/onnx_reranker.py`) — drop-in replacement using `optimum.onnxruntime`; falls back to PyTorch `Reranker` on `ImportError`; enabled via `RERANKER_USE_ONNX=true`
- **`PassageTruncator`** (`src/internal/retrieval/passage_truncator.py`) — whitespace-token truncation applied before scoring; zero-dependency; configurable via `RERANKER_MAX_TOKENS` (0 = disabled)
- **`RerankerBenchmark`** (`src/internal/retrieval/reranker_benchmark.py`) — offline CLI grid search over model × batch_size × max_tokens; writes JSONL output and prints a ranked table sorted by NDCG@k
- **`QueryTransformPipeline`** (`src/context/query_transform.py`) — composes decompose, HyDE, step-back, keyword expansion, and filter extraction behind one interface, producing a `TransformedQueryBundle`; `bundle.retrieval_variants(max_variants)` deduplicates the variants and always keeps the original query last, and `RetrievalService` retrieves each variant in parallel then fuses with `rrf_fuse`; all `QT_*` env vars default to `false` (zero overhead when disabled); appends `+rag_fusion` to `retrieval_mode`. Refactored to expose `_build_jobs`/`_assemble` and a per-query `config_override`, plus the module helper `config_signature()`, so the wrappers below can compose on top of the unchanged leaf
- **`AsyncQueryTransformPipeline`** (`src/internal/retrieval/async_query_transform.py`) — runs the leaf's transform calls (decompose, HyDE, step-back, keywords, filter construction) concurrently in a `ThreadPoolExecutor`; a transform that exceeds `QT_TRANSFORM_TIMEOUT_MS` or raises degrades to its empty default rather than failing the request; enabled via `QT_ASYNC=true`
- **`CachedQueryTransformPipeline`** (`src/internal/retrieval/cached_query_transform.py`) — Redis-backed bundle cache keyed on `sha256(query|config_signature)`; caches the **filter-free** bundle and re-merges caller filters per call (no cross-caller leakage); `stats()` returns hits/misses/hit_rate; `from_env()` returns base unchanged when `QT_CACHE_REDIS_URL` is unset
- **`MultiQueryGenerator`** (`src/internal/retrieval/multi_query.py`) — true Multi-Query retrieval: one LLM call produces N paraphrased reformulations (distinct from decompose's sub-questions); surfaced as the `multi_query` field on `TransformedQueryBundle`; enabled via `QT_MULTI_QUERY=true` (`QT_MULTI_QUERY_N` controls N)
- **`variant_weighted_rrf_fuse` / `dedup_variants`** (`src/internal/retrieval/fusion.py`) — weighted RAG-Fusion across N variant result sets (original query weighted highest) gated by `QT_FUSION_WEIGHTED`; embedding-cosine dedup that drops near-duplicate variants before retrieval gated by `QT_SEMANTIC_DEDUP` (dormant until a dense backend exposes a batch `embed()`)
- **`QueryRouter` / `RoutedQueryTransformPipeline`** (`src/internal/retrieval/query_router.py`, `routed_query_transform.py`) — per-query learned routing: predicts which transforms to enable from a serialized scikit-learn artifact (`QT_ROUTER_MODEL_PATH`) with a rule-based heuristic fallback when no artifact is present; the wrapper threads the predicted config down the chain as `config_override`; enabled via `QT_ROUTER=true`. Train the artifact offline with `src/training/train_query_router.py`
- **`build_query_transform_pipeline_from_env`** (`src/internal/retrieval/query_transform_factory.py`) — composes the active layers `RoutedQueryTransformPipeline → CachedQueryTransformPipeline → AsyncQueryTransformPipeline → QueryTransformPipeline`, skipping any whose flag is unset; returns `None` (single-query path, zero overhead) when no `QT_*` flag is set
- **`QueryTransformBenchmark`** (`src/internal/retrieval/query_transform_benchmark.py`) — offline grid over technique-combination configs × a labeled dataset; `run_query_transform_benchmark()` reports recall@k / NDCG@k (reusing `eval_metrics`) plus mean transform latency per config, sorted by recall
- **`qt_slo_exceeded`** (`src/internal/retrieval/eval_runner.py`) — P99 transform-latency gate; `eval_runner --qt-slo-ms N` records per-query `qt_latency_ms` and exits non-zero when P99 exceeds the budget
- **`QueryConstructor`** (`src/internal/retrieval/query_constructor.py`) — NL → metadata filter extraction; with `QT_CONSTRUCT_OPERATORS=true` it additionally emits numeric comparison filters (`rating_gte`, `rating_lte`) beyond equality and date ranges
- **`QueryOptimizer`** (`src/internal/retrieval/query_optimizer.py`) — acronym expansion (`data/query/acronyms.json`), WordNet synonym injection, and `symspellpy` spell correction applied to the BM25 leg only; enabled via `QUERY_EXPANSION_ENABLED` / `SPELL_CORRECTION_ENABLED`
- **`BM25Tuner`** (`src/internal/retrieval/bm25_tuner.py`) — grid search over `(k1, b)` against labeled QA pairs; results written to `data/eval/bm25_params.json`; BM25+ variant (`δ=1.0`) enabled via `BM25_VARIANT=bm25plus`
- **`FAISSIndexBuilder`** (`src/internal/retrieval/index_optimizer.py`) — builds IVF-PQ indexes (`nlist=4096, m=96, nbits=8, nprobe=64`) cutting memory from ~30 GB to ≤ 4 GB at 10 M docs; `HNSWTuner` finds minimum `ef_search` meeting a recall target; `EmbeddingBatcher` coalesces concurrent embed calls within a 5ms window
- **`FusionLearner`** (`src/internal/retrieval/fusion_learner.py`) — fits per-source RRF weights `(w_sparse, w_dense)` offline; loaded at startup from `FUSION_WEIGHTS_PATH`; falls back to uniform weights when absent; `adaptive_mmr_lambda` selects `λ` by query length when `ADAPTIVE_MMR=true`
- **`ResultCache`** (`src/internal/retrieval/result_cache.py`) — Redis-backed full `SearchResponse` cache keyed on canonicalized query + filters + top_k; TTL via `RESULT_CACHE_TTL`; hit/miss stats surfaced via `GET /api/admin/retrieval/stats`
- **`graph_rag_search`** (`src/internal/retrieval/graph_rag.py`) — GraphRAG retrieval: `extract_entities` + `build_entity_graph` build an `EntityGraph` over the top retrieved passages, then re-rank by entity connectivity; served by `POST /internal/search/graph` (`retrieval_mode: "graph"`)
- **`CachedEmbedder` / `EmbeddingBatcher`** (`src/internal/retrieval/embedding_cache.py`) — query-embedding cache keyed on `sha256(query)` plus a batcher that coalesces concurrent embed calls within a short window to cut redundant encoder passes
- **`RetrievalService`** (`src/internal/retrieval/service.py`) — the retrieval core behind the HTTP server: composes sparse + dense backends → RRF fusion → MMR → optional reranker → optional query-transform pipeline; `from_env()` wires every optimization layer from `QT_*` / `RERANKER_*` / cache env vars; exposes `search(query, top_k, filters)` returning `(results, retrieval_mode)`
- **Offline evaluation** — `run_beir_eval` (`beir_eval.py`) scores BM25/dense/hybrid against BEIR datasets (NDCG/MRR/Recall); `run_ragas_eval` (`ragas_eval.py`) scores end-to-end RAG answers (faithfulness, answer/context relevancy) via `build_ragas_dataset`; `eval_runner.py` is the CI gate (NDCG/MRR/MAP + latency SLO)
- Local dense retrieval with FAISS-compatible indexes (E5, BGE, custom embedders)
- Local sparse retrieval with BM25/Pyserini
- Web search via Google Custom Search, SerpAPI, and playwright-cli
- FAISS and BM25 index builders from a JSONL corpus (`src/internal/document_index/index_builder.py`)
- Background indexing pipeline — async workers fetch, parse, chunk, enrich, embed, and index; supports mini-chunks, vector-write retries, and document prefiltering
- **Connectors** (`src/internal/connectors/`) — collect documents from multiple sources:
  - `LocalFileConnector` / `LocalFilePollConnector` — UTF-8 files from paths, directories, or globs
  - `SearchConnector` — search results as documents via retrieval, Google, or SerpAPI
  - `WebConnector` / `RSSConnector` — web page scraping and RSS feed ingestion
  - `InMemoryConnector` — Python objects for testing and prototyping
  - `OAuthConnector` — base class for authorization-code OAuth flows (Google Drive, Slack, Confluence, GitHub, Jira, SharePoint, Salesforce, Zendesk, Notion)
  - `PollConnector` / `CheckpointedConnector` / `SlimConnector` — base classes for incremental sync with time-window, checkpoint, and permission-metadata variants

**Agent Loops**
- **Agentic RAG** (`AgenticRAGLoop`) — multi-hop query decomposition, HyDE, iterative retrieval with evidence sufficiency gating, and grounded synthesis with citations
- Multi-turn `SearchAgentLoop` traces with `<think>`, `<search>`, `<information>`, and `<answer>` actions
- `ToolAgentLoop` — generic tool-calling loop usable from both search and chat flows; emits `action_trace` (newline-delimited JSON of every `ToolExecutionResult`) for downstream parsing and display
- `OnTurnCallback` — async hook called after each agent turn with `(turn, tool_name, doc_count)`; wired through `SearchAgentLoop`, `ToolAgentLoop`, and `PlainGenerationLoop`; used by the web backend to forward live progress events over SSE
- `BaseAgent` (`src/agents/graph_base.py`) — Pydantic-based agent base class; lightweight alternative to LangGraph for custom agent workflows with `invoke()`-compatible interface

**LLM Backends**
- `OpenAICompatibleLLM` — single client for OpenAI, Azure OpenAI, Anthropic, Ollama, LiteLLM, and vLLM (`src/internal/llm/providers.py`)
- Server-backed inference via any OpenAI-compatible endpoint (`--vllm_url`)
- In-process HuggingFace models on CPU, CUDA, or MPS (`--local --device`)
- Configured via `GEN_AI_MODEL_PROVIDER`, `GEN_AI_MODEL_VERSION`, `GEN_AI_API_KEY`, `GEN_AI_API_BASE`

**Tool Use**
- Hermes, Llama-3, and JSON tool-call parsers
- `ApiToolRegistry` — load and execute tools from any OpenAPI 3.x schema at runtime
- `FunctionTool` — wrap any Python callable with auto-generated JSON schema
- `build_search_tool` — ready-made tool dispatching to retrieval, Google, or SerpAPI
- `ToolCallView` (`src/internal/servers/web/app.py`) — response model for each tool call: `tool_name`, `status`, `arguments` (dict), `result_summary` (first 200 chars or "N items"), `latency_ms`, `error`; returned as `AgentExperienceResponse.tool_calls` for `intent == "tool"` requests

**Chat Processing**
- `build_chat_turn` — top-level orchestrator: resolves persona, tools, files, and LLM; dispatches to `run_llm_loop`; persists via `save_chat_turn` (`src/internal/chat/process_message.py`)
- `run_llm_loop` — multi-turn loop: message history, tool dispatch, context injection, token streaming
- `run_llm_step` — single LLM step: prompt → stream → extract tool calls → `LlmStepResult`
- `DynamicCitationProcessor` — streams tokens and extracts citation markers in REMOVE / KEEP / HYPERLINK modes
- `compress_chat_history` — summarises older turns when context exceeds the token budget; branch-aware
- `Emitter` — routes packets (tokens, tool calls, citations) from worker threads to the HTTP stream
- `build_system_prompt` — assembles system prompt from persona, tools, knowledge, and memory context
- `AgentQueueManager` (`src/internal/chat/queue_manager.py`) — thread-safe queue that funnels `AgentThought` packets (token deltas, tool calls, citations, `QueueEvent` markers) from worker threads to the SSE stream; the backbone of streamed chat
- `ChatStateContainer` / `ChatTurnSetup` / `AvailableFiles` (`src/internal/chat/chat_state.py`) — per-turn chat state: resolved persona, tools, uploaded files, and message history assembled once per turn
- `maybe_emit_argument_delta` + `Parser` (`src/internal/chat/tool_call_args_streaming.py`) — incrementally parses and streams tool-call **argument** deltas so tool inputs render live as the model emits them
- Stop / cancel signalling (`src/internal/chat/stop_signal_checker.py`) — `set_fence` / `is_connected` / `reset_cancel_status` use a Redis fence keyed by session to abort an in-flight turn when the client disconnects or hits Stop
- `compress_chat_history` token-budget policy is documented in `src/internal/chat/COMPRESSION.md`

**Cache & Persistence**
- `AgenticSearchStore` (SQLite) — connectors, documents, permissions, chat sessions, indexing attempts, usage reports, rate limits, SCIM tokens, standard answers (`src/internal/db/store.py`)
- Search history per user (`GET /search/search-history`) and query history with CSV export (`GET /admin/query-history/export`)
- `InMemoryCache` — in-flight chat session state (processing flag, stop signal, cancel) during streaming
- `ChunkBatchStore` — temp disk buffer decoupling embedding from index insertion for large jobs (`src/internal/servers/indexing/chunk_batch_store.py`)
- `InMemoryChatFile` — uploaded files (images, PDFs, text) held in memory for one chat turn

**Prompts**
- Chat prompt constants — citation reminders, system prompt defaults, file/image/tool templates (`src/internal/prompts/chat_prompts.py`)
- `KEYWORD_EXPANSION_PROMPT` / `QUERY_TYPE_PROMPT` — broaden sparse queries and classify intent for retrieval tuning
- Binary search/chat classification prompt with labelled examples and strict single-word output
- Agentic RAG prompts — decompose (2–4 sub-questions) and HyDE (hypothetical ideal answer) for `QueryEnhancer`
- `build_search_agent_instruction` — assembles the ReAct-style system prompt for `SearchAgentLoop` (`src/agents/search.py`)

**RL Training**
- Composite reward shaping (`SearchRewardFunction`) — correctness, format compliance, citation support, unnecessary-fetch penalty, and fetch-usefulness reward components
- Group-relative advantage helpers for PPO, GRPO, and REINFORCE-style experiments
- PPO core: clipped policy loss, value loss, entropy, KL penalty, adaptive and fixed KL controllers
- `LLMGRPOTrainer` — online GRPO for any HuggingFace causal-LM; rolls out G completions per prompt, scores with `judge_fn` + `SearchRewardFunction`, and updates with PPO-clip + KL penalty (`src/training/ppo/llm_grpo_trainer.py`)
- `SearchAgentGRPOTrainer` — extends `LLMGRPOTrainer` with real `SearchAgentLoop` rollouts to unlock the full shaped-reward signal (citations, search quality, fetch usefulness) (`src/training/ppo/search_agent_grpo_trainer.py`)
- **Feedback-driven GRPO** — `load_feedback_examples(db_path, min_ratings=10)` (`src/training/data.py`) reads thumbs-up/down sessions from `AgenticSearchStore` (the `retrieval_feedback` table fed by `POST /api/feedback`) into `PromptTrainingExample`s with `metadata["human_signal"] = +1.0 / -1.0`. `SearchRewardFunction` adds a `human_feedback` reward component weighted by `SearchRewardConfig.human_feedback_weight` (default `0.0` → zero regression on existing presets); `SearchAgentGRPOTrainer` threads `human_signal` from batch metadata into the score. Closes the loop: user feedback → reward signal → policy update
- **SFT warm-start** (`src/training/sft.py`) — `SFTTrainer` / `SFTConfig` (`epochs=3`, `lr=2e-5`) supervised-fine-tune a base model on agent traces before GRPO, so RL starts from a competent policy rather than cold-exploring. `load_sft_examples(db_path, jsonl_path=None, min_ratings=1)` (`src/training/data.py`) merges **thumbs-up** sessions from `AgenticSearchStore` with optional JSONL pairs (`{"question", "response"}`) into `list[SFTExample]` (built via `build_search_sft_example`). Loss is cross-entropy on **assistant tokens only** — system / user / tool-result tokens are masked to `-100` so the model imitates only the agent's own actions. Two-phase via `examples/run_sft_grpo.py`: Phase 1 SFT → intermediate checkpoint (`data/checkpoints/sft_warmstart/`) → Phase 2 GRPO loads it with `SearchAgentGRPOTrainer.from_pretrained(...)`; `--sft_epochs 0` skips straight to GRPO with no code-path change
- Training data builders for search-QA and RAG parquet datasets (`src/training/data.py`)
- `bin/generate_training_data.sh` — one-command parquet generation for Bamboogle, NQ, TriviaQA, and HotpotQA; `--preview` mode prints sample records without writing

**Intent Routing & Query Transformations**
- **Auto-routing** (`_run_auto_routed`) — single entry point that classifies every query as `search`, `chat`, or `tool` and dispatches to the right agent loop; no per-query configuration needed
- **RAG-Fusion** — `search_routing_tool` in tool mode aggregates across all configured retrieval sources, deduplicates by URL, and returns `[D1]`/`[D2]`-labelled results
- **Query decomposition** (`QueryEnhancer.decompose`) — splits complex questions into 2–4 independent sub-queries for parallel retrieval
- **HyDE** (`QueryEnhancer.hyde`) — generates a hypothetical ideal answer to expand sparse queries before retrieval
- **Step-back prompting** — reformulates narrow questions into broader conceptual queries
- **Multi-Query retrieval** (`MultiQueryGenerator`) — one LLM call yields N paraphrased reformulations retrieved in parallel and fused, distinct from decomposition's sub-questions
- **Weighted RAG-Fusion** (`variant_weighted_rrf_fuse`) — RRF across all variant result sets with the original query weighted highest; optional pre-retrieval semantic dedup of near-duplicate variants
- **Canonical query rewrite** (`QueryEnhancer.rewrite`, `QT_REWRITE`) — one normalized rewrite that fixes typos and strips verbosity while preserving meaning, distinct from step-back's broadening; threaded through the bundle/router as the 7th transform label
- **Learned query routing** (`QueryRouter`) — predicts the per-query transform set (7 labels: decompose, hyde, step_back, keywords, construct_filters, multi_query, rewrite) from a scikit-learn artifact with a rule-based heuristic fallback, so cheap/keyword queries skip expensive transforms
- **Keyword extraction** — strips conversational noise from queries before BM25 retrieval
- **Search vs chat** (`classify_is_search_flow`) — LLM-backed binary router; defaults to chat on ambiguous input (`src/internal/servers/secondary_llm_flows/search_flow_classification.py`)
- **Intent classifier** (`IntentPipeline`) — trainable feedforward ML model classifying `purchase` / `navigate` / `qa` / `recommendation`; selects fast / balanced / reasoning model tier (`src/model/intent_classifier.py`)

**Routing Layer & Query Construction** (`src/internal/routing/`, default-off behind `ROUTING_ENABLED`)
- **Per-query Router** (`router.py`) — routes each query to a domain → source(s) → retriever target, emitting a `RouteDecision`. Heuristic strategy by default (no LLM); optional **logical** (LLM structured-classification) and **semantic** (embedding-similarity over route descriptions) strategies, each falling back to the heuristic. Backed by a config-driven `RouteRegistry` (`ROUTING_REGISTRY_PATH`) so domains aren't hardcoded
- **Six query constructors** behind one `construct(query, route) -> ConstructedQuery` interface (`construction/`): **Metadata Filter** (wraps `QueryConstructor`), **Vector Search** params, **Hybrid** fusion config (reuses `adaptive_mmr_lambda`); plus net-new **SQL** (schema-aware Text-to-SQL, SELECT-only + table allowlist), **Knowledge Graph** (read-only Cypher templating, word-boundary write-clause rejection), and **API Request** (NL → allowlisted request params)
- **Construct-only safety** — the three net-new constructors build and *validate* a query but never execute it (no SQL/KG/API backend); `RetrievalService` short-circuits those targets to empty results, so routing to them never touches a live system. Every `route()`/`construct()` is fallback-safe (degrades, never raises)
- **Routing-accuracy gate** (`routing_accuracy`, `eval_runner --routing-eval`) — scores the router's top-1 retriever pick against a labeled `data/eval/routing_labels.jsonl`

**Observability & Feature Flags**
- `build_admin_surface_summary` — single-call health snapshot: connectors, indexing, users, auth, models, tools, analytics, enterprise controls with a composite health score
- `MonitoringWorker` — background poller for process memory (RSS), index queue depth, connector count; ships JSON snapshots to a cloud data-plane URL
- `event_telemetry` / `identify_user` — PostHog event capture helpers; no-ops when PostHog is not configured
- Feature flags — composable chain: `EnvFeatureFlagProvider` → `PostHogFeatureFlagProvider`; `StaticFeatureFlagProvider` for tests; single call-site via `is_feature_enabled`


## Agent Framework & Control Flow

The agent layer (`src/agents/`) behind every loop the [Web Backend API](#web-backend-api) and [CLI](#examples) drive.

### Agent taxonomy — two families

The repo has **two parallel agent designs**; "the agent framework" is the first, and the registry covers only it.

| Family | Members | `run()` contract | LLM access | Registry? | GRPO-trainable? |
|---|---|---|---|---|---|
| **Framework loops** (`AgentLoopBase`) | `plain_generation`, `single_turn_agent`, `search_agent` (the search/tool agents), `tool_agent` | `run(messages, sampling_params, *, on_turn) → AgentLoopOutput` | injected `server_manager` (token-level) | ✅ | ✅ |
| **RAG pipeline** | `AgenticRAGLoop` (web `chat_loop`) | `run(question, *, chat_history) → AgenticRAGResult` | `LLMClient` (chat-level) | ❌ | ❌ |
| **Retrieval pipelines** | `search_tool`, `hybrid_search`, `chat_once` | retrieve → answer functions | — | ❌ | ❌ |

**Tool agents and search agents are members of the framework** — siblings under `AgentLoopBase`, sharing the registry, the `LoopController` + components, and the `server_manager` model boundary. **Agentic RAG sits *beside* the framework, not inside it:** its constructor, `run()` signature, and return type diverge from `AgentLoopBase`, so registering it would break the `dict[str, type[AgentLoopBase]]` contract — it stays a deliberate non-registry loop. A dispatch layer (registry + `resolve_agent_name` + the web intent router) picks one target per request, treating all three families as interchangeable.

**Why they're kept separate (by design).** The framework loops are *token-level* because they're built for GRPO RL training (policy gradients need `prompt_ids`/`response_ids`). `AgenticRAGLoop` is a lighter *chat-level* serving pipeline (`LLMClient`, no tokenizer/`server_manager`) doing query decomposition + HyDE + grounded synthesis. Two simple designs for two purposes beat one contract forced onto both; the boundary is enforced (the registry rejects non-conforming loops) and documented ([`docs/agent-invocation-surface.md`](docs/agent-invocation-surface.md)) so the families don't quietly drift together. Consolidation *is* feasible — `SearchAgentLoop` already does most of what `AgenticRAGLoop` does (sub-questions, iterative retrieval, evidence gating, citations); express agentic-RAG as a `SearchAgentLoop` config + a HyDE query-transform, bridged via the `ServerManager` protocol, and retire `AgenticRAGLoop`. It's deferred architectural-debt work, not a feature, so the families stay separate for now.

**Loop registry — one source of truth.** Agent loops register by name (`@register`) and are resolved through `get_registered_agent_loop(name)`; `resolve_agent_name` maps CLI/web aliases to the canonical loop. The registry covers the four `AgentLoopBase` loops below. `AgenticRAGLoop` (constructor + `run()` signature diverge from `AgentLoopBase`) and the retrieval pipelines (`search_tool` / `hybrid_search` / `chat_once`) are a distinct, non-registry category — see [`docs/agent-invocation-surface.md`](docs/agent-invocation-surface.md).

| Canonical loop | CLI `--mode` | Web `mode` | Purpose |
|---|---|---|---|
| `plain_generation` | `single` | — | one-shot generation, no retrieval |
| `single_turn_agent` | — | — | one-shot RAG |
| `search_agent` | `search` | `search_agent` | multi-turn retrieval QA |
| `tool_agent` | `tool` | `tool_agent` | generic function calling |

```bash
python -c "from src import list_registered_agent_loops, resolve_agent_name; \
print(list_registered_agent_loops()); print(resolve_agent_name('search'))"
# → ['plain_generation', 'search_agent', 'single_turn_agent', 'tool_agent']
# → search_agent
```

**LoopController — the search loop's two decisions.** `SearchAgentLoop` consults a stateless `LoopController` (`src/agents/components/loop_controller.py`) for *keep searching?* and *how to answer?*. Four **default-on** behaviors (tunable via `SearchAgentLoopConfig`):

- **Adaptive search budget** — `effective_search_limit` scales rounds by subquestion count: `max_search_limit + search_budget_per_subquestion·(n−1)`, capped at `max_search_limit_cap` (default `10`); single-subquestion runs are unchanged.
- **Plateau early-stop** — stops searching when a round's evidence gain `< evidence_plateau_min_gain` (default `0.05`) **and** evidence is already sufficient (`plateau_requires_sufficient`); never forces a thin answer.
- **Graceful dead-end answer** — at a dead-end / budget-exhaust with evidence collected, one bounded turn yields a best-effort answer instead of returning nothing (`force_answer_on_deadend`); never fabricates when no evidence exists.
- **Smarter answer-gating** — accept / reject (with targeted per-subquestion feedback) / force, decided by the controller.

Each surfaces an additive `metrics` key — `effective_search_limit`, `adaptive_budget_bonus`, `plateau_early_stop`, `forced_final_answer` — the last priced by `SearchRewardConfig.forced_final_answer_penalty` (`-0.05`, mutually exclusive with `answer_when_evidence_insufficient`). Existing reward presets stay byte-stable.

**`run()` control flow** is a linear, append-only turn loop. Each turn: `_generate_turn` (build prompt → generate → decode → parse actions) → action dispatch → `_apply_answer_gate` / `_handle_no_action`, which return a `TurnControl` directive (`CONTINUE` / `BREAK`) the loop acts on → the observation is appended as a `user` message. `_finalize_run_metrics` computes the derived/reward metrics once after the loop.

**Model backend.** Every loop receives an injected `server_manager` satisfying the `ServerManager` protocol (`src/model/serving.py`); `build_server_manager(tokenizer, server_url=…, model=…)` selects the OpenAI-compatible (remote) or in-process HuggingFace (local) backend — shared by the CLI and the web app.

**Drive it from the CLI** (control-flow knobs in `examples/run_agentic_search.py`):
```bash
python -m examples.run_agentic_search --mode search \
  --question "Compare dense and sparse retrieval" \
  --model meta-llama/Llama-3.1-8B-Instruct --vllm_url http://localhost:8080 \
  --search_url http://localhost:8001/retrieve \
  --max_search_limit 5 --max_turns 8 --max_answer_rejections 3
# --no_evidence_gate disables the require-sufficient-evidence answer gate
```

**Drive it over the API / UI.** `POST /api/agent` picks the loop by `mode`; `POST /api/agent/stream` emits a `progress` SSE event after each turn via the `OnTurnCallback`, so plateau early-stops and forced dead-end answers appear live in the UI progress trace:
```bash
curl -sN -X POST http://localhost:7860/api/agent/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare dense and sparse retrieval", "mode": "search_agent", "top_k": 5}'
# data: {"type": "progress", "turn": 1, "text": "search_routing_tool · 5 docs"}
# data: {"type": "progress", "turn": 2, "text": "writing answer…"}
# data: {"type": "answer", "text": "..."}
# data: {"type": "done", "intent": "search", "citations": ["[D1]"], "documents": [...]}
```
See [Web Backend API](#web-backend-api) for the full request/response schema.

## Agentic RAG

`chat_loop` is the web API name for `AgenticRAGLoop` — web modes are named by session behavior, not retrieval strategy. Valid modes: `search_tool`, `hybrid_search`, `chat_once`, `chat_loop`.

```bash
curl -X POST http://localhost:7860/api/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "What is FAISS?", "mode": "chat_loop", "top_k": 5}'
```

Loop flow:

1. **Query enhancement** — decompose into sub-queries; generate HyDE hypothetical answer
2. **Hybrid+rerank retrieval** — retrieve per enhanced query; accumulate unique documents
3. **Sufficiency check** — LLM judges if context is enough; break or continue
4. **Follow-up generation** — LLM proposes targeted follow-up queries if insufficient
5. **Grounded synthesis** — answer from all accumulated evidence with inline citations


## Retrieval Setup

`src.internal.document_index` is the single indexing entry point — filtering, chunking, embedding, retry-isolated writes, and failure reporting. Query-time retrievers and the retrieval HTTP client live in `src.context`. Reranker utilities live in `src.internal.servers.retrieval`.

**Retrieval servers** (`src/internal/servers/retrieval/`):

| Module | Description |
|--------|-------------|
| `demo.py` | TF-IDF over corpus.jsonl — no Java required |
| `retrieval_server.py` | BM25 or dense (E5/BGE via FAISS) |
| `retrieval_rerank.py` | Retrieval + cross-encoder reranker |
| `rerank.py` | Standalone cross-encoder reranker (no retrieval) |
| `hybrid_rerank.py` | Dense + BM25 RRF fusion + rerank (recommended for `AgenticRAGLoop`) |

**Web search servers** (`src/internal/servers/web_search/`):

| Module | Description |
|--------|-------------|
| `google.py` | Google Custom Search proxy |
| `serp.py` | SerpAPI proxy |
| `browser.py` | playwright-cli browser automation; no API key, ~5–10s/query |

**Start a retrieval server:**

```bash
# Dense (E5)
python3 -m src.internal.servers.retrieval.retrieval_server \
  --model_path intfloat/e5-base-v2 --index_path data/indexes/e5_Flat.index \
  --corpus_path data/corpus.jsonl --retrieval_method e5 --device cpu --topk 5

# Sparse BM25
python3 -m src.internal.servers.retrieval.retrieval_server \
  --index_path data/indexes/bm25 --corpus_path data/corpus.jsonl --retrieval_method bm25
```

**Build indexes:**

```bash
python3 -m src.internal.document_index.index_builder \
  --retrieval_method e5 --model_path intfloat/e5-base-v2 \
  --corpus_path data/corpus.jsonl --faiss_type Flat --save_dir data/indexes/

python3 -m src.internal.document_index.index_builder \
  --retrieval_method bm25 --corpus_path data/corpus.jsonl --save_dir data/indexes/
```

**Hybrid + rerank:**

```bash
python3 -m src.internal.servers.retrieval.hybrid_rerank \
  --dense_model intfloat/e5-base-v2 --index_path data/indexes/e5_Flat.index \
  --corpus_path data/corpus.jsonl \
  --sparse_index_path data/indexes/bm25 --hybrid_alpha 0.5 \
  --retrieval_topk 10 --rerank_topk 5
```

**Web search servers:**

```bash
python3 -m src.internal.servers.web_search.serp \
  --search_url "https://serpapi.com/search" --topk 3 --serp_api_key "$SERP_API_KEY"

python3 -m src.internal.servers.web_search.google \
  --api_key "$GOOGLE_API_KEY" --topk 5 --cse_id "$GOOGLE_CSE_ID" --snippet_only
```

**Health check:**

```bash
curl -i -sS http://127.0.0.1:8001/health
curl -i -sS -X POST http://127.0.0.1:8001/retrieve \
  -H "Content-Type: application/json" -d '{"query":"What is FAISS?","topk":5}'
```


## Neural Reranking

`RetrievalService` optionally reranks hybrid-fused results via a layered wrapper chain. Set `RERANKER_PROVIDER` to enable; all wrappers are opt-in via env vars and compose on top of the unchanged `Reranker` leaf.

**Wrapper chain** (outermost → innermost):
```
TwoStageReranker → CachedReranker → AsyncReranker → Reranker (leaf)
```

**Enable local BGE reranking:**
```bash
RERANKER_PROVIDER=local RERANKER_MODEL=BAAI/bge-reranker-v2-m3 \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

**Enable Cohere reranking:**
```bash
RERANKER_PROVIDER=cohere RERANKER_MODEL=rerank-english-v3.0 COHERE_API_KEY=... \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

**Enable async + Redis cache wrapper:**
```bash
RERANKER_PROVIDER=local RERANKER_ASYNC=true \
  RERANKER_TIMEOUT_MS=500 RERANKER_CACHE_REDIS_URL=redis://localhost:6379 \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

**Enable two-stage pipeline** (fast pre-filter → heavy scorer):
```bash
RERANKER_PROVIDER=local RERANKER_TWO_STAGE=true \
  RERANKER_FAST_MODEL=BAAI/bge-reranker-base \
  RERANKER_PRE_FILTER_TOP_N=50 RERANKER_OVER_FETCH_MULTIPLIER=2.0 \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

**ONNX runtime** (lower latency than PyTorch, requires `pip install optimum[onnxruntime]`):
```bash
RERANKER_PROVIDER=local RERANKER_USE_ONNX=true RERANKER_MODEL=BAAI/bge-reranker-base \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

**Evaluate reranker quality and latency:**
```bash
# Baseline vs reranked NDCG/MRR + per-query latency
python -m src.internal.retrieval.eval_runner \
  --dataset data/eval/qa_pairs.jsonl --top_k 10 \
  --reranker local --reranker_model BAAI/bge-reranker-v2-m3 \
  --compare-baseline --slo-ms 200

# Output JSON:
# { "retrieval":  {"ndcg@10": 0.48, "mrr": 0.63},
#   "reranked":   {"ndcg@10": 0.55, "mrr": 0.71, "map@10": 0.52},
#   "latency_ms": {"mean": 312, "p50": 290, "p99": 680, "n": 50},
#   "reranker_improvement_ratio": 0.145 }
```

**Benchmark model configurations offline:**
```bash
python -m src.internal.retrieval.reranker_benchmark \
  --qa-pairs data/eval/qa_pairs.jsonl \
  --models BAAI/bge-reranker-base BAAI/bge-reranker-v2-m3 \
  --batch-sizes 8 16 32 \
  --max-tokens 256 512 \
  --output results/reranker_bench.jsonl
# Prints ranked table sorted by NDCG@10
```


## Retrieval Server API

The retrieval server (`src/internal/servers/retrieval/server.py`, examples use `:8001`) exposes the retrieval core over HTTP. The demo server (`demo.py`, TF-IDF) only serves `POST /retrieve`; the full server below adds per-mode and admin endpoints.

**Health:**
```bash
curl -s http://localhost:8001/health
# → {"status": "ok", "backend": "local"}
```

**Hybrid search with metadata filters** (`POST /search` — sparse+dense → RRF → MMR → optional rerank):
```bash
curl -s -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what is FAISS?", "top_k": 5, "filters": {"source": "arxiv"}}'
# → {"results": [{"doc_id": "...", "title": "...", "text": "...", "score": 0.71, ...}],
#    "retrieval_mode": "hybrid", "executed_queries": ["what is FAISS?"], "latency_ms": 41.2}
```

**Per-mode retrieval** (`/internal/search/*` — isolate one retrieval strategy, e.g. for evals):
```bash
# Sparse (BM25) only
curl -s -X POST http://localhost:8001/internal/search/sparse \
  -H "Content-Type: application/json" -d '{"query": "vector database", "top_k": 5}'
# → retrieval_mode: "sparse"

# Dense (embeddings) only
curl -s -X POST http://localhost:8001/internal/search/dense \
  -H "Content-Type: application/json" -d '{"query": "vector database", "top_k": 5}'
# → retrieval_mode: "dense"

# Hybrid with explicit fusion/MMR knobs
curl -s -X POST http://localhost:8001/internal/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{"query": "vector database", "top_k": 5, "over_fetch": 4, "mmr_lambda": 0.5}'
# → retrieval_mode: "hybrid"

# GraphRAG (entity-graph re-ranking)
curl -s -X POST http://localhost:8001/internal/search/graph \
  -H "Content-Type: application/json" -d '{"query": "who founded OpenAI", "top_k": 5}'
# → retrieval_mode: "graph"
```

**Demo server** (`demo.py`, TF-IDF, no Java/embeddings — note `topk`):
```bash
curl -s -X POST http://localhost:8001/retrieve \
  -H "Content-Type: application/json" -d '{"query": "what is FAISS?", "topk": 5}'
```

**Standalone reranker** (`rerank.py` — batch interface: `queries` + per-query `documents` lists):
```bash
curl -s -X POST http://localhost:8001/rerank \
  -H "Content-Type: application/json" \
  -d '{"queries": ["what is FAISS?"],
       "documents": [[{"title": "FAISS", "content": "FAISS is a similarity search library"},
                      {"title": "Cats", "content": "Cats are mammals"}]],
       "rerank_topk": 2}'
```

**Inspect / hot-reload retrieval config** (admin):
```bash
curl -s http://localhost:8001/api/admin/retrieval/stats
curl -s -X PATCH http://localhost:8001/api/admin/retrieval/config \
  -H "Content-Type: application/json" \
  -d '{"rrf_k": 80, "mmr_lambda": 0.4, "nprobe": 96, "result_cache_ttl": 600}'
```


## Web Backend API

The FastAPI web backend (`src/internal/servers/web/app.py`, `:7860`) drives the UI and agent loops.

**Run the intent-routed agent** (`POST /api/agent`) — auto-routes search / chat / tool; `response.intent` reflects the chosen path:
```bash
curl -s -X POST http://localhost:7860/api/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare dense and sparse retrieval", "mode": "chat_loop", "top_k": 5}'
# → {"answer": "...", "intent": "chat", "citations": ["[D1]"], "documents": [...], "session_id": "..."}
```

`response.intent` is `"search" | "chat" | "tool"` and is the single field that drives the [intent-adaptive layout](#ui-features) (`App.tsx` maps it to a `.results-layout` class). Read just that field:
```bash
curl -s -X POST http://localhost:7860/api/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "find the onboarding checklist", "top_k": 5}' \
  | python -c "import sys, json; print(json.load(sys.stdin)['intent'])"
# → search
```

**Stream the same over SSE** (`POST /api/agent/stream`) — emits one `progress` event after each agent turn (via the `on_turn` callback), then `answer`, then `done` (which carries `intent`, `citations`, and `documents`; the frontend feeds `intent` to `setIntent`). The non-streaming `/api/agent` is unchanged:
```bash
curl -sN -X POST http://localhost:7860/api/agent/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare dense and sparse retrieval", "top_k": 5}'
# Server-Sent Events (one JSON object per `data:` line):
# data: {"type": "progress", "turn": 1, "text": "search_routing_tool · 5 docs"}
# data: {"type": "progress", "turn": 2, "text": "writing answer…"}
# data: {"type": "answer",   "text": "Dense retrieval embeds the query …"}
# data: {"type": "done",     "session_id": "...", "intent": "chat", "citations": ["[D1]"], "documents": [...]}
```
On failure the stream yields `data: {"type": "error", "detail": "..."}` instead of `done`, which `streamAgent` surfaces as the error banner.

**Sessions:**
```bash
curl -s -X POST http://localhost:7860/api/sessions \
  -H "Content-Type: application/json" -d '{"title": "Search session"}'
curl -s http://localhost:7860/api/sessions/{session_id}
```

**Submit retrieval feedback** (`POST /api/feedback` — drives the feedback-GRPO training signal):
```bash
curl -s -X POST http://localhost:7860/api/feedback \
  -H "Content-Type: application/json" \
  -d '{"session_id": "sess-123", "signal": "thumbs_up"}'
# → {"ok": true}
```

### Request paths & dispatch

Both `POST /api/agent` and `/api/agent/stream` run the same dispatcher
(`_run_agent_impl`). The optional `mode` field selects the path; **when `mode` is
omitted (the default, and what the bundled UI always sends) the request goes
through the 4-way auto-router.** Every path returns the same shape
`(answer, citations, documents, intent, …)` with `intent ∈ {search, chat, tool}`.

| `mode` | Path / loop | `intent` | Requires |
|---|---|---|---|
| _(omitted)_ | auto-router → one of the four below | varies | — |
| `search_agent` | `SearchAgentLoop` (multi-turn search) | `search` | local model (else `400`) |
| `tool_agent` | `ToolAgentLoop` (OpenAPI/MCP tools) | `tool`/`search`/`chat` | local model (else `400`) |
| `chat_loop` | `AgenticRAGLoop` (decompose + HyDE) | `chat` | LLM client |
| `hybrid_search` | internal + web fan-out, MMR-merged | `search` | — |
| `search_tool` | raw retrieval, no synthesis | `search` | — |

**Auto-router** (`route_query`, `src/internal/servers/web/intent_routing.py`):
explicit non-`auto` source → `search_agent`; otherwise an LLM 4-way classifier
(`classify_route`) when an LLM is present, else a rule-based route. The chosen
strategy (`direct_llm` / `agentic_rag` / `search_agent` / `tool_agent`) dispatches
the matching loop and **degrades** when its backend is absent (e.g.
`search_agent`→hybrid pipeline with no local model; `agentic_rag`→pipeline with no
LLM). `extra["route"]` / `extra["route_degraded"]` record the decision. The three
routing axes — strategy, web-vs-internal (`source_provider`), and internal backend
(the server-side M10 router) — are described in [docs/routing-axes.md](docs/routing-axes.md).

**`source_provider`** (web vs internal corpus): `auto` (default — internal
`retrieval` ∥ web `serpapi`, merged via MMR), `retrieval`, `serpapi`, `google`,
`browser`, `all`. The bundled UI sends `source_provider` and `search_url` only in
dev builds; in production it sends neither, so the backend uses `auto`.

### Web reachability of auto-routed queries (known gap)

Whether an auto-routed UI query actually reaches the **web** is config-dependent
and currently narrower than the `source_provider=auto` default suggests:

- **With a local model configured** (`SEARCH_AGENT_MODEL`): the auto path runs
  either `SearchAgentLoop` (search) or `AgenticRAGLoop` (chat) — **both are
  internal-corpus only**. `AgenticRAGLoop` retrieves only from `retrieval_url`;
  `SearchAgentLoop`'s web retriever (`web_search_url`) is never wired in the web
  backend, so `<search retriever="web">` silently degrades to the internal corpus.
- **The web is reached only via the retrieval-first pipeline**
  (`_auto_search_pipeline` → `_run_hybrid_search`, `source_provider=auto`), which
  the auto path uses only as a **degradation** — `search_agent` with no local
  model, or `agentic_rag` with no LLM — or via explicit `mode=hybrid_search`.
- Even then, the web leg needs `SERP_API_KEY`/`SERPAPI_API_KEY` (SerpAPI) or a
  configured `browser_search_url`; without them it returns error docs and the
  result is effectively internal-only.
- `direct_llm` performs no retrieval at all.

Net: in the common single-machine setup (local model set, no SerpAPI key), every
UI query stays on the internal corpus. Routing the multi-turn loop to the web via
the existing `source_provider` infrastructure is a possible future change.


## Chat & Session API

Chat session management and search-flow routing live on the web backend (`:7860`) under the `/chat`, `/search`, and `/query` routers (`src/internal/servers/query_and_chat/`). The streamed send-message flow itself is `POST /api/agent` / `/api/agent/stream` above; these endpoints manage the sessions and feedback around it.

**Chat sessions** (`/chat`):
```bash
# Create a session
curl -s -X POST http://localhost:7860/chat/create-chat-session \
  -H "Content-Type: application/json" -d '{"title": "Onboarding questions"}'
# → {"chat_session_id": "..."}

# List the user's sessions / fetch one with its messages
curl -s http://localhost:7860/chat/get-user-chat-sessions
curl -s http://localhost:7860/chat/get-chat-session/{session_id}

# Rename / delete
curl -s -X PUT http://localhost:7860/chat/rename-chat-session \
  -H "Content-Type: application/json" \
  -d '{"chat_session_id": "...", "name": "Renamed"}'
curl -s -X DELETE http://localhost:7860/chat/delete-chat-session/{session_id}
```

**Per-message feedback** (`POST /chat/create-chat-message-feedback`):
```bash
curl -s -X POST http://localhost:7860/chat/create-chat-message-feedback \
  -H "Content-Type: application/json" \
  -d '{"chat_message_id": "...", "is_positive": true, "feedback_text": "spot on"}'
```

**Search-flow classification** (`POST /search/search-flow-classification` — keyword-search vs chat routing):
```bash
curl -s -X POST http://localhost:7860/search/search-flow-classification \
  -H "Content-Type: application/json" -d '{"user_query": "find the Q3 onboarding deck"}'
# → {"is_search_flow": true}
```

**Direct search message** (`POST /search/send-search-message` — optional query expansion, streamable):
```bash
curl -s -X POST http://localhost:7860/search/send-search-message \
  -H "Content-Type: application/json" \
  -d '{"search_query": "vector database benchmarks", "run_query_expansion": true, "num_hits": 10, "stream": false}'
```

**Search history** (`GET /search/search-history`):
```bash
curl -s http://localhost:7860/search/search-history
```

`GET /query/standard-answer` exists but is an Enterprise-gated stub — it returns `501` ("Standard Answers is an Enterprise feature … not available in this deployment") in the open-source build.


## Retrieval Optimization

All optimization components are opt-in; unset env vars = unchanged M1–M4 behavior.

**Tune BM25 parameters against your QA pairs:**
```bash
curl -s -X POST http://localhost:8001/internal/optimize/bm25-tune \
  -H "Content-Type: application/json" \
  -d '{"qa_pairs_path": "data/eval/qa_pairs.jsonl", "k1_range": [0.6, 0.9, 1.2], "b_range": [0.5, 0.75]}' \
  -H "Authorization: Bearer $TOKEN"
# → {"k1": 0.9, "b": 0.6, "score": 0.86}
```

**Learn fusion weights (sparse vs dense RRF weights):**
```bash
curl -s -X POST http://localhost:8001/internal/optimize/fusion-weights \
  -H "Content-Type: application/json" \
  -d '{"qa_pairs_path": "data/eval/qa_pairs.jsonl"}' \
  -H "Authorization: Bearer $TOKEN"
# → {"w_sparse": 0.38, "w_dense": 0.62}
```

**Tune HNSW ef_search for a recall target:**
```bash
curl -s -X POST http://localhost:8001/internal/optimize/hnsw-tune \
  -H "Content-Type: application/json" \
  -d '{"target_recall": 0.82}' \
  -H "Authorization: Bearer $TOKEN"
# → {"ef_search": 96, "measured_recall": 0.831}
```

**Retrieval stats (cache hit rate, latency, throughput):**
```bash
curl -s http://localhost:7860/api/admin/retrieval/stats \
  -H "Authorization: Bearer $TOKEN"
# → {"result_cache_hit_rate": 0.42, "p99_latency_ms": 112, "throughput_qps": 87, ...}
```

**Hot-reload tunable parameters without restart:**
```bash
curl -s -X PATCH http://localhost:7860/api/admin/retrieval/config \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"rrf_k": 80, "mmr_lambda": 0.4, "nprobe": 96, "result_cache_ttl": 600}'
# → {"applied": ["rrf_k", "mmr_lambda", "nprobe", "result_cache_ttl"]}
```

**Enable query expansion and result caching:**
```bash
QUERY_EXPANSION_ENABLED=true SPELL_CORRECTION_ENABLED=true EXPANSION_MAX_TERMS=3 \
  BM25_VARIANT=bm25plus \
  RESULT_CACHE_REDIS_URL=redis://localhost:6379 RESULT_CACHE_TTL=300 \
  ADAPTIVE_MMR=true \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

**Build an IVF-PQ FAISS index** (cuts memory from ~30 GB to ≤ 4 GB at 10 M docs):
```python
from src.internal.retrieval.index_optimizer import FAISSIndexBuilder
import numpy as np

builder = FAISSIndexBuilder()
index = builder.build_ivfpq(embeddings, nlist=4096, m=96, nbits=8, nprobe=64)
# Save alongside existing index; load via FAISS_INDEX_TYPE=ivfpq
```


## Query Transformation Optimization

A layered-wrapper optimization stack over `QueryTransformPipeline`, parallel to Neural Reranking. Every layer is opt-in; with all `QT_*` unset, `RetrievalService` runs the single-query path unchanged (`build_query_transform_pipeline_from_env` returns `None`).

**Wrapper chain** (outermost → innermost):
```
RoutedQueryTransformPipeline → CachedQueryTransformPipeline → AsyncQueryTransformPipeline → QueryTransformPipeline (leaf)
```

**Enable parallel transforms + Redis bundle cache:**
```bash
QT_DECOMPOSE=true QT_HYDE=true QT_STEP_BACK=true \
  QT_ASYNC=true QT_TRANSFORM_TIMEOUT_MS=400 \
  QT_CACHE_REDIS_URL=redis://localhost:6379 QT_CACHE_TTL_SECONDS=600 \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

**Enable Multi-Query + weighted RAG-Fusion:**
```bash
QT_MULTI_QUERY=true QT_MULTI_QUERY_N=3 QT_FUSION_WEIGHTED=true \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

**Enable per-query learned routing** (heuristic until an artifact exists):
```bash
QT_ROUTER=true QT_ROUTER_MODEL_PATH=data/query_router.joblib \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```
`QT_ROUTER` and `QT_MULTI_QUERY` each activate the pipeline on their own — no other `QT_*` flag is required.

Query transformation is **backend-only** — there is no dedicated HTTP endpoint and no query-transform-specific UI. The pipeline runs inside `RetrievalService.from_env()`, so it applies to **both** the retrieval server's `/search` and the web backend's `/api/agent`. Its observable effect is the `+rag_fusion` suffix on `retrieval_mode`.

**Test it on the retrieval server** (`POST /search` — `retrieval_mode` reflects the transform):
```bash
# Start the retrieval server with QT flags enabled, then:
curl -s -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare dense and sparse retrieval", "top_k": 5}' \
  | python -c "import sys, json; print(json.load(sys.stdin)['retrieval_mode'])"
# → hybrid+rag_fusion
```

**Test it on the web backend** (`POST /api/agent`):
```bash
curl -s -X POST http://localhost:7860/api/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare dense and sparse retrieval", "mode": "chat_loop", "top_k": 5}' \
  | python -m json.tool | grep -i retrieval_mode
# → "retrieval_mode": "hybrid+rag_fusion"   (or "hybrid+rag_fusion+reranked" with a reranker)
```

**Extract metadata filters from natural language** (numeric operators behind `QT_CONSTRUCT_OPERATORS`):
```bash
QT_CONSTRUCT_FILTERS=true QT_CONSTRUCT_OPERATORS=true \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
# "arxiv papers after 2023 rated above 4" → filters {date_after: "2023-...", rating_gte: 4}
curl -s -X POST http://localhost:7860/api/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "arxiv papers after 2023 rated above 4 on retrieval", "mode": "chat_loop", "top_k": 5}'
```

**Train the learned router offline:**
```bash
python -m src.training.train_query_router --out data/query_router.joblib
# → wrote data/query_router.joblib
# Predicts 7 transform labels: decompose, hyde, step_back, keywords, construct_filters, multi_query, rewrite
```

**Gate transform latency in CI:**
```bash
python -m src.internal.retrieval.eval_runner \
  --dataset data/eval/qa_pairs.jsonl --top_k 10 --qt-slo-ms 300
# Records per-query "qt_latency_ms"; exits non-zero when P99 transform latency > 300ms
```

**Benchmark technique combinations offline** (Python API; the `--dataset` CLI ships a stub `retrieve_fn` to wire to your retriever):
```python
from src.context.query_transform import QueryTransformConfig
from src.internal.retrieval.query_transform_benchmark import run_query_transform_benchmark

dataset = [("what is FAISS", {"doc-1"}), ("compare BM25 and dense", {"doc-2"})]

def retrieve(query, config):
    # build a pipeline from `config`, run RetrievalService.search, return ranked doc_ids
    ...

rows = run_query_transform_benchmark(dataset, retrieve, [
    QueryTransformConfig(),
    QueryTransformConfig(multi_query=True),
    QueryTransformConfig(decompose=True, hyde=True),
], k=10)
# → [{"config_signature": "...", "recall": 0.91, "ndcg": 0.78, "mean_latency_ms": 142.0}, ...]
```


## Routing & Query Construction

The RAG **Routing → Query Construction** stage (`src/internal/routing/`). It decides **where** a query should go (domain → source → retriever) and **how** to express it for the chosen backend. Distinct from [Intent Routing](#intent-routing) (web-level `search`/`chat`/`tool`) and from `QueryRouter` (which picks *transforms*): this layer picks the *retriever/construction target* per query.

**Backend-only and default-off.** With no `ROUTING_*` env set, `build_router_from_env()` returns `None`, `RetrievalService.search` skips the routing branch entirely, and behavior is byte-identical to today — zero overhead, no frontend change. There is no dedicated HTTP endpoint or UI; routing runs inside `RetrievalService.from_env()`.

**Pipeline:**
```
query → Router.route() → RouteDecision(domain, sources, retriever, construction_target)
      → QueryConstructor.construct() → ConstructedQuery(target, payload, text)
```

**Router strategies** (heuristic default; LLM strategies fall back to it on any failure):

| Strategy | Env | How it routes |
|----------|-----|---------------|
| Heuristic | (default) | Rule-based cue matching → SQL / GRAPH / API / default HYBRID. No LLM; the path the accuracy gate runs against |
| Logical | `ROUTING_LOGICAL=true` | LLM structured-classification into a registered route by name |
| Semantic | `ROUTING_SEMANTIC=true` | Embedding cosine between the query and each route's description |

Routes come from a config-driven registry (`ROUTING_REGISTRY_PATH` → JSON of `{name, description, sources, retriever}`; a built-in default mirrors the local corpus). `RetrieverTarget` ∈ `sparse · dense · hybrid · metadata · sql · graph · api`.

**Six query constructors** (`construction/`, one `construct(query, route) -> ConstructedQuery` interface):

| Constructor | Target | Backing | Output |
|-------------|--------|---------|--------|
| Metadata Filter | `metadata` | wraps `QueryConstructor` | NL → `{filters}` + cleaned query |
| Vector Search | `dense` | params | `{top_k, namespace, filters}` |
| Hybrid Retrieval | `hybrid` | reuses `adaptive_mmr_lambda` | `{rrf_k, w_sparse, w_dense, mmr_lambda}` |
| SQL Generation | `sql` | net-new (no exec) | schema-aware Text-to-SQL, SELECT-only + table allowlist + multi-statement reject |
| Knowledge Graph | `graph` | net-new (no exec) | read-only Cypher (`MATCH…RETURN`), word-boundary write-clause rejection |
| API Request | `api` | net-new (no exec) | `{endpoint, params}` filtered to an `ApiSpec` allowlist |

The three net-new constructors **build and validate but never execute** a query — there is no live SQL/KG/API backend, so `RetrievalService` short-circuits the `sql`/`graph`/`api` targets to `([], "routed:<target>")`. When a real backend is wired later, only the executor changes. Every `route()`/`construct()` degrades to a safe empty/None payload rather than raising.

**Enable per-query routing:**
```bash
ROUTING_ENABLED=true \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
# Optional LLM strategies + a custom route registry:
ROUTING_ENABLED=true ROUTING_LOGICAL=true ROUTING_SEMANTIC=true \
  ROUTING_REGISTRY_PATH=data/routes.json  uvicorn ...
```

**Score routing accuracy** (heuristic router; no LLM needed):
```bash
python -m src.internal.retrieval.eval_runner \
  --routing-eval --dataset data/eval/routing_labels.jsonl
# → {"routing_accuracy": 1.0, "num_queries": 12}
```


## Training

The training pipeline is modular: generate trajectories → score with rewards → compute advantages → optimize.

| Task | Entry point |
|------|-------------|
| QA parquet preparation | `python3 -m examples.prepare_search_qa_dataset` |
| Training data (shell) | `bin/generate_training_data.sh` |
| Reward/GRPO smoke test | `python3 -m examples.run_grpo_training_pipeline` |
| Bamboogle benchmark eval | `python3 -m examples.run_bamboogle_eval` / `bin/run_bamboogle_eval.sh` |
| Reward function | `src/training/reward.py` |
| GRPO helpers | `src/training/grpo.py` |
| Online GRPO for HF LMs | `src/training/ppo/llm_grpo_trainer.py` |
| Agent-loop GRPO (full reward) | `src/training/ppo/search_agent_grpo_trainer.py` |
| PPO core | `src/training/ppo/core_algos.py` |
| Generation and policy loss | `src/model/generation.py` |
| Feedback-driven GRPO | `python3 -m examples.run_feedback_grpo` |
| SFT warm-start + GRPO | `python3 -m examples.run_sft_grpo` |

**Fine-tune from user feedback** — train directly on thumbs-up/down sessions collected via `POST /api/feedback` (no GPU required for the smoke path; `--device mps` on Apple Silicon):
```bash
# Feedback-driven GRPO: load rated sessions from the web DB → reward with human_signal → update
python3 -m examples.run_feedback_grpo \
  --db_path data/feedback.sqlite3 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --min_ratings 10 --human_feedback_weight 0.5 \
  --num_rollouts 4 --search_url http://localhost:8001/retrieve --device mps \
  --output_dir data/checkpoints/feedback_grpo/

# SFT warm-start (Phase 1, assistant-token-only CE on thumbs-up traces) then GRPO (Phase 2);
# --sft_epochs 0 skips Phase 1 and runs pure GRPO from the base model
python3 -m examples.run_sft_grpo \
  --db_path data/feedback.sqlite3 --model Qwen/Qwen2.5-1.5B-Instruct \
  --jsonl_path data/sft_pairs.jsonl \
  --sft_epochs 3 --sft_lr 2e-5 --sft_output_dir data/checkpoints/sft_warmstart/ \
  --grpo_output_dir data/checkpoints/sft_grpo/ --device mps
```
`load_feedback_examples` raises if fewer than `--min_ratings` rated sessions exist, so collect feedback first (thumbs in the UI, or `POST /api/feedback`). There is **no HTTP training endpoint** — fine-tuning is offline by design; the only backend endpoint in this loop is `POST /api/feedback` (see [Web Backend API](#web-backend-api)).

**Reward components** (`SearchRewardFunction`):

| Component | Config field | What it measures |
|-----------|-------------|-----------------|
| Correctness | `correctness_weight` | Judge score against gold answer (EM / contains-match) |
| Citation support | `citation_support_weight` | Fraction of retrieved docs cited in the final answer |
| Subquestion coverage | `subquestion_coverage_weight` | Fraction of sub-questions with sufficient evidence |
| Search quality | `search_quality_weight` | Evaluator verdict + per-query search quality |
| Unnecessary search | `unnecessary_search_penalty` | Penalty per search round beyond the first |
| Unnecessary fetch | `unnecessary_fetch_penalty` | Penalty per fetched page not cited in the answer |
| Fetch usefulness | `fetch_usefulness_reward` | Bonus when fetched pages are cited in the final answer |
| Format compliance | `format_reward_weight` | Structural compliance in the final answer |
| Human feedback | `human_feedback_weight` | `human_signal` (±1.0) from thumbs-up/down sessions; `0.0` by default (off) |

Reward preset names: `sparse_final_only` | `simple_sparse_with_search_penalty` | `second_pass` | `third_pass_with_format` (see `SearchRewardConfig` in `src/training/reward.py`).

**GRPO** — `score_prompt_group` scores G rollouts for one prompt and normalises within-group advantages. `compute_grpo_outcome_advantage` computes `reward_i - mean(group)` for a flat rewards list. See `src/training/grpo.py`.

**PPO** — `compute_ppo_policy_loss_core` returns `(pg_loss, pg_clipfrac, ppo_kl, surrogate)`; `compute_value_loss` returns `(vf_loss, vf_clipfrac)`. Both require an `eos_mask` tensor. See `src/training/ppo/core_algos.py`.

**Smoke test** (end-to-end reward + GRPO, no GPU):

```bash
python3 -m examples.run_grpo_training_pipeline
```

**XML search protocol** — the ReAct-style trace format used by `SearchAgentLoop`:

Model-output tags:

```xml
<think>decide whether to answer or search</think>
<search>one precise query when external evidence is needed</search>
<fetch>comma- or newline-separated URLs when snippets are insufficient</fetch>
<answer>final grounded answer with citation labels</answer>
```

Optional model-output tags for multi-hop tasks:

```xml
<search_decision>answer</search_decision>   <!-- skip search when internal knowledge suffices -->
<subquestions>one research subquestion per line</subquestions>
<searches>parallel independent queries, one per line</searches>
```

Environment-only tags (injected by the loop — never output by the model):

```xml
<information>search results with citation labels</information>
<search_evaluation>sufficiency verdict and weak-query hints</search_evaluation>
<subquestions_feedback>per-subquestion coverage status</subquestions_feedback>
<full_page>fetched page content</full_page>
```

Mask all environment-only tags from policy/SFT action loss.


## MCP Server

The MCP server exposes Agentic Search capabilities as [Model Context Protocol](https://modelcontextprotocol.io/) tools, letting any MCP-compatible client (Claude Desktop, Cursor, etc.) query your knowledge base directly.

**Start the server** (requires the `mcp` extra):

```bash
pip install -e ".[mcp]"
uvicorn src.internal.mcp_server.api:mcp_app --port 8090
```

**Connect Claude Desktop** — add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "agentic-search": {
      "type": "http",
      "url": "http://localhost:8090/",
      "headers": { "Authorization": "Bearer YOUR_TOKEN_HERE" }
    }
  }
}
```

**Tools available to the LLM client:**

| Tool | What it does |
|------|-------------|
| `search_indexed_documents` | Search the private knowledge base with optional source filter |
| `search_web` | Web search via Google Custom Search or SerpAPI |
| `open_urls` | Fetch full page text from a list of URLs |
| `ask_agentic_search` | Full `SearchAgentLoop` answer with citations |
| `retrieve_documents` | Raw retrieval — returns full document content and relevance scores |
| `expand_query` | Query decomposition and HyDE expansion |

Dynamic tools registered via `FunctionTool` / `ApiToolRegistry` can be mirrored to MCP by calling `sync_tool_to_mcp(name)` after registration (`src/internal/mcp_server/tools/dynamic.py`).

**Resources:**

| Resource | What it exposes |
|----------|----------------|
| `indexed_sources` | Available retrieval source types based on configured API keys |
| `document_sets` | Document sets scoped for search |

**Debug with MCP Inspector:**

```bash
npx @modelcontextprotocol/inspector http://localhost:8090/
```

**MCP environment variables:**

| Var | Default | Description |
|-----|---------|-------------|
| `MCP_SERVER_CORS_ORIGINS` | — | Comma-separated allowed origins for CORS |
| `API_SERVER_HOST` | `127.0.0.1` | Host of the web backend |
| `API_SERVER_PROTOCOL` | `http` | Protocol for the web backend URL |
| `API_SERVER_URL_OVERRIDE_FOR_HTTP_REQUESTS` | — | Override the full web backend URL |


## Evaluation

### Bamboogle

Bamboogle is a two-hop QA benchmark that requires chaining retrieval across multiple hops — a strong signal for `SearchAgentLoop` quality.

**CLI (local CPU):**

```bash
python3 -m examples.run_bamboogle_eval \
  --model Qwen/Qwen2.5-1.5B-Instruct --local --limit 5 --print_trace
```

**CLI (server-backed):**

```bash
python3 -m examples.run_bamboogle_eval \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --vllm_url http://localhost:8080 \
  --search_url http://localhost:8001/retrieve \
  --reward_preset second_pass --limit 125
```

Reward presets: `sparse_final_only` | `simple_sparse` | `second_pass` | `third_pass`

**Apple Silicon shell script** (auto-starts SerpAPI retrieval server, reads `SERP_API_KEY` from `.env`):

```bash
bin/run_bamboogle_eval.sh                              # 5 examples, mps device
bin/run_bamboogle_eval.sh --smoke                      # 1 example, quick sanity check
bin/run_bamboogle_eval.sh --limit 125                  # full benchmark
bin/run_bamboogle_eval.sh --device cpu --limit 10
bin/run_bamboogle_eval.sh --limit 125 --concurrency 8  # ~6-8x faster via parallel SerpAPI calls
bin/run_bamboogle_eval.sh --limit 125 --concurrency 8 --resume  # resume an interrupted run
```

The dataset is cached locally after the first download (`~/.cache/agentic_search/bamboogle_test.jsonl`), so subsequent runs skip the network fetch. `--resume` reads the existing output file and skips already-evaluated questions, appending new results.

**Training data generation:**

```bash
bin/generate_training_data.sh                         # Bamboogle → data/bamboogle_train/
bin/generate_training_data.sh --preview               # print 5 sample rows, no write
bin/generate_training_data.sh --dataset nq            # Natural Questions
bin/generate_training_data.sh --dataset trivia_qa     # TriviaQA
bin/generate_training_data.sh --dataset hotpotqa --max_examples 500
```

Each run writes `data/<dataset>_train/train.parquet` and `data/<dataset>_train/test.parquet` ready for `LLMGRPOTrainer` or SFT.


## API Health Checks

Web backend: `http://localhost:7860` · Retrieval server: `http://localhost:8001`

**Generate a dev JWT** (required for admin endpoints):

```bash
export TOKEN=$(bin/gen_dev_token.sh)   # or: source bin/gen_dev_token.sh
```

**Core**

```bash
curl -s http://localhost:7860/health                  # web server
curl -s http://localhost:8001/health                  # retrieval server
curl -s http://localhost:7860/settings                # tier / license status (no auth)
```

**Search & chat**

```bash
curl -s -X POST http://localhost:7860/api/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "What is FAISS?", "mode": "search_tool"}'

curl -s http://localhost:7860/api/sessions/SESSION_ID -H "Authorization: Bearer $TOKEN"

curl -s -X POST http://localhost:8001/retrieve \
  -H "Content-Type: application/json" -d '{"query": "dense retrieval", "topk": 3}'
```

**Admin — analytics, billing, reporting**

```bash
curl -s "http://localhost:7860/analytics/query?start=2024-01-01&end=2025-12-31" \
  -H "Authorization: Bearer $TOKEN"
curl -s http://localhost:7860/admin/billing/billing-information -H "Authorization: Bearer $TOKEN"
curl -s http://localhost:7860/admin/usage-report                -H "Authorization: Bearer $TOKEN"
```

**Admin — hooks, rate limits, web search**

```bash
curl -s http://localhost:7860/admin/hooks/specs              -H "Authorization: Bearer $TOKEN"
curl -s http://localhost:7860/admin/hooks                    -H "Authorization: Bearer $TOKEN"
curl -s http://localhost:7860/admin/token-rate-limits/users  -H "Authorization: Bearer $TOKEN"
curl -s http://localhost:7860/admin/web-search/search-providers -H "Authorization: Bearer $TOKEN"
```

**Admin — license**

```bash
curl -s http://localhost:7860/license       -H "Authorization: Bearer $TOKEN"
curl -s http://localhost:7860/license/seats -H "Authorization: Bearer $TOKEN"
```

**SCIM** (uses SCIM bearer token, not a JWT)

```bash
curl -s http://localhost:7860/scim/v2/ServiceProviderConfig  # no auth
curl -s http://localhost:7860/scim/v2/Users  -H "Authorization: Bearer $SCIM_TOKEN"
curl -s http://localhost:7860/scim/v2/Groups -H "Authorization: Bearer $SCIM_TOKEN"
```


## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `AGENTIC_SEARCH_AUTH_SECRET` | `agentic-search-dev-secret` | JWT signing secret |
| `AGENTIC_SEARCH_SUPER_USERS` | `[]` | JSON list of admin user IDs or emails |
| `AGENTIC_SEARCH_WEB_DB_PATH` | `:memory:` | SQLite path (`:memory:` for ephemeral) |
| `AGENTIC_SEARCH_RETRIEVAL_URL` | `http://localhost:8001/retrieve` | Retrieval server URL |
| `AGENTIC_SEARCH_CLOUD_DATA_PLANE_URL` | — | Cloud data plane for billing proxy |
| `AGENTIC_SEARCH_LICENSE_ENFORCEMENT_ENABLED` | `false` | Enable license gating |
| `AGENTIC_SEARCH_DATA_DIR` | `~/.local/share/agentic_search` | License file directory |
| `WEB_DOMAIN` | `http://localhost:8080` | External URL for OAuth redirects |
| `GEN_AI_MODEL_PROVIDER` | `openai` | LLM provider (openai, anthropic, ollama, etc.) |
| `GEN_AI_MODEL_VERSION` | `gpt-4o-mini` | Model name / version |
| `GEN_AI_API_KEY` | — | Provider API key |
| `GEN_AI_API_BASE` | — | Override base URL (e.g. `http://localhost:11434/v1`) |
| `OAUTH_SLACK_CLIENT_ID` | — | Slack OAuth app client ID |
| `OAUTH_CONFLUENCE_CLOUD_CLIENT_ID` | — | Confluence OAuth app client ID |
| `OAUTH_GOOGLE_DRIVE_CLIENT_ID` | — | Google Drive OAuth app client ID |
| `RERANKER_PROVIDER` | — | `local` or `cohere`; omit to disable neural reranking in `RetrievalService` |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder model for local reranking |
| `RERANKER_BATCH_SIZE` | `32` | Batch size for local cross-encoder |
| `RERANKER_DEVICE` | `cpu` | Device for local reranker (`cpu`, `mps`, `cuda`) |
| `RERANKER_TOP_K` | same as search `top_k` | Cap returned results after reranking |
| `COHERE_API_KEY` | — | Cohere API key (required when `RERANKER_PROVIDER=cohere`) |
| `RERANKER_ASYNC` | `false` | Wrap reranker in `AsyncReranker` (thread-pool offload) |
| `RERANKER_TIMEOUT_MS` | `500` | Per-query scorer timeout for `AsyncReranker` |
| `RERANKER_MAX_WORKERS` | `4` | Thread pool size for `AsyncReranker` |
| `RERANKER_CACHE_REDIS_URL` | — | Enable `CachedReranker`; set to a Redis URL |
| `RERANKER_CACHE_TTL_SECONDS` | `300` | TTL for cached reranker scores |
| `RERANKER_MAX_TOKENS` | `512` | `PassageTruncator` token limit before scoring (0 = disabled) |
| `RERANKER_USE_ONNX` | `false` | Load reranker via ONNX runtime (`ONNXReranker`) |
| `RERANKER_TWO_STAGE` | `false` | Enable `TwoStageReranker` (fast pre-filter → heavy scorer) |
| `RERANKER_PRE_FILTER_TOP_N` | `50` | Candidates passed to the heavy scorer in two-stage mode |
| `RERANKER_FAST_MODEL` | inherits `RERANKER_MODEL` | Fast-stage model name in two-stage mode |
| `RERANKER_OVER_FETCH_MULTIPLIER` | `2.0` | Retrieval over-fetch ratio when a reranker is active |
| `QUERY_EXPANSION_ENABLED` | `false` | Enable acronym + WordNet synonym expansion in BM25 leg |
| `SPELL_CORRECTION_ENABLED` | `false` | Enable `symspellpy` spell correction in BM25 leg |
| `EXPANSION_MAX_TERMS` | `3` | Max added terms per query to prevent BM25 query bloat |
| `BM25_VARIANT` | — | Set to `bm25plus` to enable BM25+ lower-bound floor (`δ=1.0`) |
| `FAISS_INDEX_TYPE` | `hnsw` | `ivfpq` for IVF-PQ quantized index; `hnsw` for original |
| `EF_SEARCH` | — | HNSW `ef_search` override (higher = more recall, slower) |
| `ADAPTIVE_MMR` | `false` | Select MMR `λ` by query length (short → 0.8, long → 0.3) |
| `FUSION_WEIGHTS_PATH` | `data/eval/fusion_weights.json` | Learned per-source RRF weights; falls back to uniform if absent |
| `RESULT_CACHE_REDIS_URL` | — | Enable `ResultCache`; set to a Redis URL |
| `RESULT_CACHE_TTL` | `300` | TTL in seconds for cached full search responses |
| `LATENCY_SLO_MS` | `120` | CI SLO gate: P99 above this exits non-zero in `eval_runner` |
| `QT_DECOMPOSE` | `false` | Enable query decomposition in `QueryTransformPipeline` |
| `QT_HYDE` | `false` | Enable HyDE (hypothetical document embedding) |
| `QT_STEP_BACK` | `false` | Enable step-back query rephrasing |
| `QT_KEYWORDS` | `false` | Enable keyword expansion for BM25 variants |
| `QT_CONSTRUCT_FILTERS` | `false` | Enable NL → metadata filter extraction |
| `QT_REWRITE` | `false` | Enable canonical query rewrite (`QueryEnhancer.rewrite`); 7th router label |
| `QT_MAX_VARIANTS` | `5` | Max parallel retrieval variants when any `QT_*` is enabled |
| `QT_ASYNC` | `false` | Run the leaf's transform LLM calls in parallel (`AsyncQueryTransformPipeline`) |
| `QT_TRANSFORM_TIMEOUT_MS` | `400` | Per-transform timeout; on exceed that field degrades to its default |
| `QT_MAX_WORKERS` | `5` | Thread-pool size for `AsyncQueryTransformPipeline` |
| `QT_CACHE_REDIS_URL` | — | Enable `CachedQueryTransformPipeline`; set to a Redis URL |
| `QT_CACHE_TTL_SECONDS` | `600` | TTL for cached transform bundles |
| `QT_MULTI_QUERY` | `false` | Enable `MultiQueryGenerator` (N paraphrased query variants) |
| `QT_MULTI_QUERY_N` | `3` | Number of paraphrases generated per query |
| `QT_FUSION_WEIGHTED` | `false` | Use `variant_weighted_rrf_fuse` (original query weighted highest) |
| `QT_SEMANTIC_DEDUP` | `false` | Drop near-duplicate variants before retrieval (needs a backend `embed()`) |
| `QT_SEMANTIC_DEDUP_THRESHOLD` | `0.95` | Cosine cutoff for variant dedup |
| `QT_ROUTER` | `false` | Per-query routing of transforms (`QueryRouter` + heuristic fallback) |
| `QT_ROUTER_MODEL_PATH` | — | Serialized scikit-learn router artifact; heuristic used when unset/missing |
| `QT_CONSTRUCT_OPERATORS` | `false` | Extract numeric range/comparison filters (`rating_gte`/`rating_lte`) |
| `ROUTING_ENABLED` | `false` | Enable the per-query routing layer in `RetrievalService` (domain/source/retriever + query construction); zero overhead when unset |
| `ROUTING_LOGICAL` | `false` | Add the LLM structured-classification router strategy (falls back to heuristic) |
| `ROUTING_SEMANTIC` | `false` | Add the embedding-similarity router strategy (falls back to heuristic) |
| `ROUTING_REGISTRY_PATH` | — | JSON route registry (`{name, description, sources, retriever}`); built-in default used when unset |


## Tests

```bash
pytest                           # full suite
pytest tests/unit/ -v            # unit only
pytest tests/unit/servers/ -v    # server-focused
pytest tests/unit/test_reward.py tests/unit/test_grpo.py tests/unit/test_llm_agent_generation.py -v

# Integration (requires live server, default http://localhost:8080)
pytest tests/integration/ -v
API_SERVER_HOST=localhost API_SERVER_PORT=8080 pytest tests/integration/
```

| Test area | What is tested |
|-----------|----------------|
| `server/billing/` | Circuit breaker state, endpoint responses, HTTP mocks |
| `server/features/hooks/` | SSRF safety, endpoint validation, `HookValidateStatus` |
| `server/license/` | PEM stripping, `_strip_pem` boundary cases |
| `server/middleware/` | Path allowlist, license enforcement, tier gating |
| `server/settings/` | `_load_license_status`, `/settings` endpoint |
| `server/web/test_tool_trace.py` | `ToolCallView` trace parsing, latency rounding, list/string summarisation, error forwarding |
| `utils/test_license_utils.py` | RSA signature verification with real key pairs |
| `utils/test_license_expiry.py` | 18 parametrized `ExpiryWarningStage` boundary points |
| `utils/test_tier.py` | `get_tier` + `tier_at_least` matrix |

Frontend tests (`web/src/components/__tests__/`):

| Test file | What is tested |
|-----------|----------------|
| `App.test.tsx` | SSE streaming flow, intent class applied per response, reset on new session |
| `AnswerPanel.test.tsx` | Markdown rendering, `[D1]` citation link generation, `ReactNode[]` children handling |
| `SessionTimeline.test.tsx` | Chat bubble layout, system message filtering, stable React keys |
| `SourceGrid.test.tsx` | Card expand/collapse, copy button 1.5 s feedback, `id` anchor attribute |
| `ToolCallTracePanel.test.tsx` | Empty→null, completed/failed card classes, latency display, JSON arguments |


## Notes

- Dense retrieval defaults to CPU; set `--device cuda` on a dedicated retrieval node or `--device mps` on Apple Silicon.
- MPS acceleration is available for local inference (`--device mps`); add `--allow_unsafe_mps` to suppress PyTorch MPS safety warnings.
- BM25 serving requires Java because Pyserini uses Lucene.
- Empty or invalid queries return empty result lists.
- Some web pages block scraping or return little usable text.
- Google Custom Search and SerpAPI are subject to their own quota and billing rules.
- If `prepare_search_qa_dataset` fails with a `pyarrow` extension error, run `pip install -r requirements.txt`.
