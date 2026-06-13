# Agentic Search

A retrieval-backed agent platform for multi-turn search, RAG, and RL training. Built around a FastAPI backend, interchangeable retrieval servers, and an async agent loop that supports dense/sparse hybrid retrieval, tool calling, and streaming chat.

🔍 **Agentic RAG** — Multi-hop retrieval with query decomposition, HyDE, hybrid reranking, and citation-grounded synthesis via `AgenticRAGLoop`.

🤖 **Custom Agents** — Compose agents from instructions, knowledge sources, tools, and memory; backed by `SearchAgentLoop` or `ToolAgentLoop`.

🌍 **Web Search** — Live retrieval via Google Custom Search, SerpAPI, and playwright-cli browser automation — all behind the same `/retrieve` API.

📚 **Document Indexing** — Chunk, embed, and index documents into FAISS or BM25; async background workers handle ingestion at scale.

🔗 **Connectors** — Pull content from local files, Google Drive, Slack, Confluence, GitHub, Jira, SharePoint, Salesforce, Zendesk, and Notion.

🛠️ **Tool Use** — Register Python callables or OpenAPI 3.x schemas as tools; `ToolAgentLoop` handles dispatch and structured output.

💬 **Chat Orchestration** — Streaming multi-turn chat with citation extraction, tool dispatch, context compression, and persisted sessions.

🧠 **RL Training** — GRPO/PPO training with composite shaped rewards; `SearchAgentGRPOTrainer` runs real agent-loop rollouts so all reward components fire during training.

📐 **Bamboogle Evaluation** — Benchmark `SearchAgentLoop` on two-hop QA with exact-match, contains-match, and shaped reward metrics; Apple Silicon (`--device mps`) supported out of the box.

🔌 **MCP Server** — Expose search, retrieval, and RAG as Model Context Protocol tools so any MCP-compatible LLM client (Claude Desktop, etc.) can query your knowledge base directly.

📊 **Admin & Observability** — Health, analytics, rate limits, hooks, billing, SCIM provisioning, and license state via the FastAPI admin API.


[![Architecture](agentic-search-grpo-architecture.png)](https://htmlpreview.github.io/?https://github.com/lingduoduo/Agentic-Search-GRPO/blob/main/agentic-search-grpo-architecture.html)

*Click to open the interactive version.*


| Feature | Key modules |
|---------|-------------|
| 🔍 Agentic RAG | `src/agents/agentic_rag.py`, `src/context/query_enhancer.py`, `src/internal/servers/retrieval/hybrid_rerank.py` |
| 🤖 Custom Agents | `src/agents/search.py`, `src/agents/custom.py`, `src/agents/tool_calling.py`, `src/agents/base.py` |
| 🌍 Web Search | `src/internal/servers/web_search/google.py`, `src/internal/servers/web_search/serp.py`, `src/internal/servers/web_search/browser.py` |
| 📚 Document Indexing | `src/internal/document_index/`, `src/internal/servers/backgroundworker/` |
| 🔗 Connectors | `src/internal/connectors/`, `src/internal/servers/connectors/`, `src/internal/servers/oauth/` |
| 🛠️ Tool Use | `src/tools/base.py`, `src/tools/api.py`, `src/tools/search.py`, `src/agents/tool_calling.py` |
| 💬 Chat Orchestration | `src/internal/chat/process_message.py`, `src/internal/chat/llm_loop.py`, `src/internal/chat/citation_processor.py`, `src/internal/chat/compression.py` |
| 🧠 RL Training | `src/training/reward.py`, `src/training/grpo.py`, `src/training/ppo/search_agent_grpo_trainer.py` |
| 📐 Bamboogle Evaluation | `src/training/eval/bamboogle.py`, `examples/run_bamboogle_eval.py`, `bin/run_bamboogle_eval.sh` |
| 🔌 MCP Server | `src/internal/mcp_server/tools/`, `src/internal/mcp_server/resources/` |
| 📊 Admin & Observability | `src/internal/observability/`, `src/internal/servers/analytics/`, `src/internal/servers/reporting/`, `src/internal/servers/license/` |


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

**Retrieval service** — `http://localhost:8000`
```bash
python3 -m src.internal.servers.retrieval.demo --corpus_path data/corpus.jsonl
```

**Web API** — `http://localhost:7860`
```bash
PYTHONPATH=src uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

**Frontend** — `http://localhost:5173`
```bash
cd web && npm install && npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api/*` to the web API on port 7860.
For production, `npm run build` produces `web/dist`; the FastAPI app serves it automatically.


## Examples

**Agent CLI**

| Mode | Loop | Needs retrieval server | Use it for |
|------|------|------------------------|------------|
| `single` | `PlainGenerationLoop` | No | Local generation smoke tests |
| `search` | `SearchAgentLoop` | Yes | Multi-turn RAG, SFT, and RL traces |
| `tool` | `ToolAgentLoop` | Yes | Structured tool-calling experiments |

```bash
# single — no retrieval server needed (plain generation)
python3 -m examples.run_agentic_search \
  --mode single --question "What is FAISS?" \
  --model Qwen/Qwen2.5-1.5B-Instruct --local --device cpu --allow_remote_model_downloads

# search — local model, requires retrieval server on :8000
python3 -m examples.run_agentic_search \
  --mode search --question "What is FAISS?" \
  --model Qwen/Qwen2.5-1.5B-Instruct --local --device cpu \
  --search_url http://localhost:8000/retrieve --allow_remote_model_downloads

# tool — local model, requires retrieval server on :8000
python3 -m examples.run_agentic_search \
  --mode tool --question "What is FAISS?" \
  --model Qwen/Qwen2.5-1.5B-Instruct --local --device cpu \
  --search_url http://localhost:8000/retrieve

# search — server-backed, requires vLLM on :8080 and retrieval on :8000
python3 -m examples.run_agentic_search \
  --mode search --question "Compare dense and sparse retrieval" \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --vllm_url http://localhost:8080 --search_url http://localhost:8000/retrieve
```

**Bamboogle evaluation** (always requires retrieval server on :8000)

```bash
# Smoke test — local model, 1 example, full trace printed
python3 -m examples.run_bamboogle_eval \
  --model Qwen/Qwen2.5-1.5B-Instruct --local --device cpu \
  --search_url http://localhost:8000/retrieve --limit 1 --print_trace

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
- **Query enhancer** — `QueryEnhancer.decompose()` and `.hyde()` enrich any query; degrades gracefully without an LLM
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
- `ToolAgentLoop` — generic tool-calling loop usable from both search and chat flows
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

**Chat Processing**
- `build_chat_turn` — top-level orchestrator: resolves persona, tools, files, and LLM; dispatches to `run_llm_loop`; persists via `save_chat_turn` (`src/internal/chat/process_message.py`)
- `run_llm_loop` — multi-turn loop: message history, tool dispatch, context injection, token streaming
- `run_llm_step` — single LLM step: prompt → stream → extract tool calls → `LlmStepResult`
- `DynamicCitationProcessor` — streams tokens and extracts citation markers in REMOVE / KEEP / HYPERLINK modes
- `compress_chat_history` — summarises older turns when context exceeds the token budget; branch-aware
- `Emitter` — routes packets (tokens, tool calls, citations) from worker threads to the HTTP stream
- `build_system_prompt` — assembles system prompt from persona, tools, knowledge, and memory context

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
- Training data builders for search-QA and RAG parquet datasets (`src/training/data.py`)
- `bin/generate_training_data.sh` — one-command parquet generation for Bamboogle, NQ, TriviaQA, and HotpotQA; `--preview` mode prints sample records without writing

**Query Classification**
- **Search vs chat** (`classify_is_search_flow`) — LLM-backed binary router; defaults to chat on ambiguous input (`src/internal/servers/secondary_llm_flows/search_flow_classification.py`)
- **Intent classifier** (`IntentPipeline`) — trainable feedforward ML model classifying `purchase` / `navigate` / `qa` / `recommendation`; selects fast / balanced / reasoning model tier (`src/model/intent_classifier.py`)

**Observability & Feature Flags**
- `build_admin_surface_summary` — single-call health snapshot: connectors, indexing, users, auth, models, tools, analytics, enterprise controls with a composite health score
- `MonitoringWorker` — background poller for process memory (RSS), index queue depth, connector count; ships JSON snapshots to a cloud data-plane URL
- `event_telemetry` / `identify_user` — PostHog event capture helpers; no-ops when PostHog is not configured
- Feature flags — composable chain: `EnvFeatureFlagProvider` → `PostHogFeatureFlagProvider`; `StaticFeatureFlagProvider` for tests; single call-site via `is_feature_enabled`


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
curl -i -sS http://127.0.0.1:8000/health
curl -i -sS -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" -d '{"query":"What is FAISS?","topk":5}'
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
  --search_url http://localhost:8000/retrieve \
  --reward_preset second_pass --limit 125
```

Reward presets: `sparse_final_only` | `simple_sparse` | `second_pass` | `third_pass`

**Apple Silicon shell script** (auto-starts SerpAPI retrieval server, reads `SERP_API_KEY` from `.env`):

```bash
bin/run_bamboogle_eval.sh                        # 5 examples, mps device
bin/run_bamboogle_eval.sh --smoke                # 1 example, quick sanity check
bin/run_bamboogle_eval.sh --limit 125            # full benchmark
bin/run_bamboogle_eval.sh --device cpu --limit 10
```

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

Web backend: `http://localhost:7860` · Retrieval server: `http://localhost:8000`

**Generate a dev JWT** (required for admin endpoints):

```bash
export TOKEN=$(bin/gen_dev_token.sh)   # or: source bin/gen_dev_token.sh
```

**Core**

```bash
curl -s http://localhost:7860/health                  # web server
curl -s http://localhost:8000/health                  # retrieval server
curl -s http://localhost:7860/settings                # tier / license status (no auth)
```

**Search & chat**

```bash
curl -s -X POST http://localhost:7860/api/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "What is FAISS?", "mode": "search_tool"}'

curl -s http://localhost:7860/api/sessions/SESSION_ID -H "Authorization: Bearer $TOKEN"

curl -s -X POST http://localhost:8000/retrieve \
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
| `AGENTIC_SEARCH_RETRIEVAL_URL` | `http://localhost:8000/retrieve` | Retrieval server URL |
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
| `utils/test_license_utils.py` | RSA signature verification with real key pairs |
| `utils/test_license_expiry.py` | 18 parametrized `ExpiryWarningStage` boundary points |
| `utils/test_tier.py` | `get_tier` + `tier_at_least` matrix |


## Notes

- Dense retrieval defaults to CPU; set `--device cuda` on a dedicated retrieval node or `--device mps` on Apple Silicon.
- MPS acceleration is available for local inference (`--device mps`); add `--allow_unsafe_mps` to suppress PyTorch MPS safety warnings.
- BM25 serving requires Java because Pyserini uses Lucene.
- Empty or invalid queries return empty result lists.
- Some web pages block scraping or return little usable text.
- Google Custom Search and SerpAPI are subject to their own quota and billing rules.
- If `prepare_search_qa_dataset` fails with a `pyarrow` extension error, run `pip install -r requirements.txt`.
