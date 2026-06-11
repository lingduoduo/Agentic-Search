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

📐 **Benchmarking** — Evaluate agents on the Bamboogle two-hop QA benchmark with exact-match, contains-match, and shaped reward scoring.

🔌 **MCP Server** — Expose search, retrieval, and RAG as Model Context Protocol tools so any MCP-compatible LLM client (Claude Desktop, etc.) can query your knowledge base directly.

📊 **Admin & Observability** — Health, analytics, rate limits, hooks, billing, SCIM provisioning, and license state via the FastAPI admin API.

------

[Architecture Diagram (interactive)](https://htmlpreview.github.io/?https://github.com/lingduoduo/Agentic-Search-GRPO/blob/main/agentic-search-grpo-architecture.html)

------

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
| 📐 Benchmarking | `src/training/eval/bamboogle.py`, `examples/run_bamboogle_eval.py` |
| 🔌 MCP Server | `src/internal/mcp_server/tools/`, `src/internal/mcp_server/resources/` |
| 📊 Admin & Observability | `src/internal/observability/`, `src/internal/servers/analytics/`, `src/internal/servers/reporting/`, `src/internal/servers/license/` |


## Repository Structure

```
src/
├── agents/              # Agent loops: SearchAgentLoop, ToolAgentLoop, AgenticRAGLoop, CustomAgent
├── context/             # Retrieval-grounded context builders
│   ├── preprocessing/   # Permission-aware access filters applied before retrieval
│   └── retrieval/       # Retrieval client helpers
├── model/               # LLM generation, intent classifier, tensor utilities
├── tools/               # Tool schemas, search tools, OpenAPI registry
├── training/            # RL training utilities
│   ├── eval/            # Benchmark evaluation (Bamboogle two-hop QA)
│   └── ppo/             # PPO/GRPO trainers including SearchAgentGRPOTrainer
└── internal/            # Platform internals
    ├── access/          # ACL & permission helpers
    ├── auth/            # Authentication & authorization
    ├── cache/           # In-memory session state cache
    ├── chat/            # Chat pipeline: loop, steps, citations, compression
    ├── configs/         # Typed config dataclasses (AppSettings)
    ├── connectors/      # Data source connector implementations
    ├── db/              # SQLite store (AgenticSearchStore)
    ├── document_index/  # FAISS/BM25 index builders and retrievers
    ├── feature_flags/   # Feature flag providers (env, PostHog, composite)
    ├── file_store/      # In-memory file handling for chat turns
    ├── hooks/           # Outbound webhook execution
    ├── llm/             # LLM provider integrations (OpenAI, Anthropic, Ollama, vLLM…)
    ├── mcp_server/      # MCP server — tools and resources for LLM clients
    ├── observability/   # Admin surface health summary
    ├── prompts/         # Prompt templates
    ├── servers/         # FastAPI routers and server entry points
    │   ├── backgroundworker/  # Async workers (light, heavy, beat, monitoring)
    │   ├── retrieval/         # Dense/sparse/hybrid/rerank server entry points
    │   ├── web_search/        # Google, SerpAPI, playwright-cli proxies
    │   ├── web/               # FastAPI app assembly (create_web_app)
    │   ├── analytics/         # Usage analytics API
    │   ├── billing/           # Stripe billing proxy
    │   ├── connectors/        # Connector-credential management
    │   ├── middleware/        # License, tier gate, tenant tracking
    │   ├── oauth/             # OAuth 2.0 connector authorization
    │   ├── query_and_chat/    # Search and chat API endpoints
    │   ├── reporting/         # Usage report ZIP generation
    │   ├── scim/              # SCIM 2.0 user & group provisioning
    │   └── …                  # tenants, users, settings, limits, license, hooks…
    └── utils/           # License, encryption, telemetry utilities
tests/                   # Unit and integration test suites
examples/                # Runnable CLI scripts
```

The FastAPI app is assembled in `src/internal/servers/web/app.py` using a router-factory pattern (`create_*_router(db, settings)`). `AgenticSearchStore` (SQLite) is the persistence layer for the web backend. The background indexing workers use Redis for inter-process queuing, but Redis is not required for the basic 3-process demo stack.


## Install

Requires Python 3.10+.

```bash
conda activate agentic-search-local
pip install -e .          # one-time; makes src importable as a package
pip install -r requirements.txt
pip install -e ".[mcp]"   # optional: MCP server (fastmcp, httpx2)
```

For BM25, Java must be available. On Apple Silicon, install FAISS via conda:

```bash
conda install -c conda-forge faiss-cpu
```

Optional env vars:

```bash
GOOGLE_API_KEY=...   GOOGLE_CSE_ID=...   SERP_API_KEY=...   JAVA_HOME=/path/to/java
```


## Quick Start

| Mode | Retrieval server | Web backend | Frontend |
|------|:---:|:---:|:---:|
| Chat only (LLM, no retrieval) | — | ✓ | ✓ |
| Search / Agentic RAG | ✓ | ✓ | ✓ |
| Search + Chat (full stack) | ✓ | ✓ | ✓ |
| API only (no browser UI) | ✓ (optional) | ✓ | — |

**Chat only** — skip the retrieval server:

```bash
# Terminal 1 — web backend
uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860

# Terminal 2 — frontend
cd web && npm install && npm run dev
```

**Search + Chat (full stack):**

```bash
# Terminal 1 — retrieval server (TF-IDF demo, no Java required; binds to port 8001)
python3 -m src.internal.servers.retrieval.demo --corpus_path data/corpus.jsonl

# Terminal 2 — web backend
uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860

# Terminal 3 — frontend
cd web && npm install && npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api/*` to port 7860. For production, `npm run build` produces `web/dist`; the FastAPI app serves it automatically.

**Verify the stack is up:**

```bash
curl -s http://127.0.0.1:8001/health   # retrieval server → {"status":"ok"}
curl -s http://127.0.0.1:7860/health   # web backend     → {"status":"ok"}
```

**Optional — MCP server** (requires `pip install -e ".[mcp]"`):

```bash
# Terminal 4 — MCP server (port 8090)
MCP_SERVER_ENABLED=true uvicorn src.internal.mcp_server.api:app \
  --host 127.0.0.1 --port 8090
```

To swap in a dense (E5/BGE), sparse (BM25), or hybrid retrieval server instead of the TF-IDF demo, see [Retrieval Setup](#retrieval-setup).


## Examples

| Script | Needs model | Needs retrieval server | Description |
|--------|:-----------:|:---------------------:|-------------|
| `run_search_pipeline.py` | — | — | Filter + permission pipeline reference (no server) |
| `run_grpo_training_pipeline.py` | — | — | Reward + GRPO advantage helpers smoke test (no GPU) |
| `run_agentic_search.py` | ✓ | `search` mode only | Agent CLI — single / search / tool modes |
| `run_bamboogle_eval.py` | ✓ | ✓ | Evaluate `SearchAgentLoop` on Bamboogle two-hop QA |
| `evaluate_bamboogle.py` | — | — | Template — wire up your own agent (stub, `NotImplementedError`) |
| `prepare_search_qa_dataset.py` | — | — | Build search-QA parquet from FlashRAG datasets |
| `prepare_search_rag_dataset.py` | — | — | Build RAG parquet from cached retrieval results |

**Search pipeline reference** (no model, no server)

```bash
python3 -m examples.run_search_pipeline
```

Exercises the filter-building → retrieval → permission-filter → chunk-merge pipeline with in-memory fixtures.

**GRPO training smoke test** (no model, no GPU)

```bash
python3 -m examples.run_grpo_training_pipeline
```

Exercises `SearchRewardFunction`, `score_prompt_group`, and `compute_grpo_outcome_advantage` with fake rollouts. Runs PPO/GRPO policy-loss helpers too if PyTorch is installed.

**Agent CLI**

```bash
# mode=single — plain generation, no retrieval (local CPU)
python3 -m examples.run_agentic_search \
  --mode single --question "What is FAISS?" \
  --model Qwen/Qwen2.5-1.5B-Instruct --local --device cpu

# mode=search — multi-turn SearchAgentLoop (requires retrieval server)
python3 -m examples.run_agentic_search \
  --mode search --question "Compare dense and sparse retrieval" \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --vllm_url http://localhost:8080 --search_url http://localhost:8001/retrieve

# mode=tool — ToolAgentLoop with Hermes/Llama-3/JSON tool-call format
python3 -m examples.run_agentic_search \
  --mode tool --question "What's the weather in Paris?" \
  --model meta-llama/Llama-3.1-8B-Instruct --local \
  --tool_format llama3

# Intent-based model routing (routes to fast/balanced/reasoning model)
python3 -m examples.run_agentic_search \
  --mode search --question "Explain transformer attention" \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --vllm_url http://localhost:8080 --search_url http://localhost:8001/retrieve \
  --model_routing intent \
  --fast_model meta-llama/Llama-3.1-8B-Instruct \
  --reasoning_model meta-llama/Llama-3.3-70B-Instruct
```

| Mode | Loop | Use it for |
|------|------|------------|
| `single` | `PlainGenerationLoop` | Plain generation, no retrieval — smoke tests and SFT data collection |
| `search` | `SearchAgentLoop` | Multi-turn RAG, citation-grounded answers, RL trace collection |
| `tool` | `ToolAgentLoop` | Function-calling experiments with Hermes, Llama-3, or JSON format |

**Bamboogle benchmark evaluation** (requires model + retrieval server)

```bash
# Local CPU — slow but self-contained
python3 -m examples.run_bamboogle_eval \
  --model Qwen/Qwen2.5-1.5B-Instruct --local --limit 20

# Server-backed — full 125 examples with shaped reward scoring
python3 -m examples.run_bamboogle_eval \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --vllm_url http://localhost:8080 \
  --search_url http://localhost:8001/retrieve \
  --reward_preset second_pass \
  --limit 125 --output results/bamboogle.jsonl
```

Reward presets: `sparse_final_only` → `simple_sparse` → `second_pass` → `third_pass` (curriculum order).

**Dataset preparation**

Both scripts default to `RUC-NLPIR/FlashRAG_datasets` / `nq`; pass `--dataset_name` and `--dataset_config` to use a different HuggingFace dataset. Output parquets are consumed by the SFT and GRPO trainers.

```bash
# Search-agent prompt parquet (question + expected answer, no context)
python3 -m examples.prepare_search_qa_dataset --local_dir data/nq_search

# Preview before writing (inspect 5 converted rows from the test split)
python3 -m examples.prepare_search_qa_dataset \
  --splits test --max_examples 20 --preview --preview_rows 5

# RAG parquet (question + pre-retrieved context + expected answer)
python3 -m examples.prepare_search_rag_dataset \
  --corpus_path data/wiki-18.jsonl \
  --train_retrieval_cache data/nq_train_retrieval_cache.json \
  --test_retrieval_cache data/nq_test_retrieval_cache.json \
  --topk 3 --local_dir data/nq_rag
```

Optional metadata flags (both scripts): `--template_type` (default `base`), `--data_source` (default `nq`), `--ability` (default `fact-reasoning`).


## Features

**Agentic RAG**
- `AgenticRAGLoop` — multi-hop query decomposition, HyDE, iterative retrieval with evidence sufficiency gating, and grounded synthesis with citations
- **Hybrid + rerank** — dense (FAISS/E5) + sparse (BM25) RRF fusion with cross-encoder reranking in a single `/retrieve` endpoint
- **Query enhancer** — `QueryEnhancer.decompose()` and `.hyde()` enrich any query; degrades gracefully without an LLM
- Local dense retrieval with FAISS-compatible indexes (E5, BGE, custom embedders)
- Local sparse retrieval with BM25/Pyserini

**Custom Agents**
- Multi-turn `SearchAgentLoop` traces with `<think>`, `<search>`, `<information>`, `<fetch>`, and `<answer>` actions
- `ToolAgentLoop` — generic tool-calling loop usable from both search and chat flows
- `CustomAgent` — compose agents from instructions, knowledge sources, tools, and memory

**Web Search**
- Google Custom Search, SerpAPI, and playwright-cli browser automation — all behind the same `/retrieve` API (`src/internal/servers/web_search/`)
- `NUM_INTERNET_SEARCH_RESULTS` / `NUM_INTERNET_SEARCH_CHUNKS` control result volume per query

**Document Indexing**
- FAISS and BM25 index builders from a JSONL corpus (`src/internal/document_index/index_builder.py`)
- Background indexing pipeline — async workers fetch, parse, chunk, enrich, embed, and index; supports mini-chunks, vector-write retries, and document prefiltering
- `ChunkBatchStore` — temp disk buffer decoupling embedding from index insertion for large jobs

**Connectors**
- `LocalFileConnector` / `LocalFilePollConnector` — UTF-8 files from paths, directories, or globs
- `SearchConnector` — search results as documents via retrieval, Google, or SerpAPI
- `InMemoryConnector` — Python objects for testing and prototyping
- `OAuthConnector` — authorization-code flow for Google Drive, Slack, Confluence, GitHub, Jira, SharePoint, Salesforce, Zendesk, Notion
- `PollConnector` / `CheckpointedConnector` / `SlimConnector` — incremental sync with time-window, checkpoint, and permission-metadata variants

**Tool Use**
- Hermes, Llama-3, and JSON tool-call parsers
- `ApiToolRegistry` — load and execute tools from any OpenAPI 3.x schema at runtime
- `FunctionTool` — wrap any Python callable with auto-generated JSON schema
- `build_search_tool` — ready-made tool dispatching to retrieval, Google, or SerpAPI

**Chat Orchestration**
- `process_message` — top-level orchestrator: resolves persona, tools, files, and LLM; dispatches to `run_llm_loop`; persists via `save_chat_turn`
- `run_llm_loop` — multi-turn loop: message history, tool dispatch, context injection, token streaming
- `run_llm_step` — single LLM step: prompt → stream → extract tool calls → `LlmStepResult`
- `DynamicCitationProcessor` — streams tokens and extracts citation markers in REMOVE / KEEP / HYPERLINK modes
- `compress_chat_history` — summarises older turns when context exceeds the token budget; branch-aware
- `Emitter` — routes packets (tokens, tool calls, citations) from worker threads to the HTTP stream
- `build_system_prompt` — assembles system prompt from persona, tools, knowledge, and memory context
- `AgenticSearchStore` (SQLite) — connectors, documents, permissions, chat sessions, indexing, rate limits, SCIM tokens (`src/internal/db/store.py`)
- `InMemoryCache` — in-flight chat session state (processing flag, stop signal, cancel) during streaming

**RL Training**
- Composite reward shaping (`SearchRewardFunction`) — format, search-use, answer-length, exact-match, citation quality, unnecessary-search penalty, and search-efficiency components (`src/training/reward.py`)
- `PPORewardManager` — batched reward scoring adapter between the GRPO trainer and `SearchRewardFunction` (`src/training/ppo/reward_manager.py`)
- `SearchAgentGRPOTrainer` — GRPO trainer that replaces `model.generate()` rollouts with real `SearchAgentLoop` executions, enabling fully shaped rewards from live search trajectories (`src/training/ppo/search_agent_grpo_trainer.py`)
- Group-relative advantage helpers for PPO, GRPO, DAPO, and REINFORCE-style experiments (`src/training/grpo.py`, `src/training/ppo/core_algos.py`)
- PPO core: clipped policy loss, value loss, entropy, KL penalty, adaptive and fixed KL controllers (`src/training/ppo/core_algos.py`)
- SFT helpers — `build_search_sft_example` converts any `SearchAgentLoop` rollout into a supervised training example; supports full trajectory or completion-only mode (`src/training/sft.py`)
- Training data builders for search-QA and RAG parquet datasets (`src/training/data.py`)

**Benchmarking**
- **Bamboogle evaluation** (`src/training/eval/bamboogle.py`) — two-hop QA benchmark (125 examples) with EM, contains-match, and optional shaped reward scoring

**MCP Server**
- FastMCP server exposing search, retrieval, and RAG as Model Context Protocol tools (`src/internal/mcp_server/`)
- Bearer-token auth; optional install via `pip install -e ".[mcp]"`
- Compatible with Claude Desktop, MCP Inspector, and any OpenAI-tool-compatible client

**LLM Backends**
- `OpenAICompatibleLLM` — single HTTP client for OpenAI, Azure OpenAI, Anthropic, Ollama, LiteLLM, and vLLM via the OpenAI streaming chat-completions protocol (`src/internal/llm/providers.py`)
- `LiteLLM` singleton integration for provider-agnostic routing (`src/internal/llm/litellm_singleton/`)
- `VLLMServerManager` / `LocalServerManager` — example-layer helpers for launching vLLM server-backed and in-process HuggingFace inference (`examples/run_agentic_search.py`)
- Configured via `GEN_AI_MODEL_PROVIDER`, `GEN_AI_MODEL_VERSION`, `GEN_AI_API_KEY`, `GEN_AI_API_BASE`

**Query Classification**
- **Search vs chat** (`classify_is_search_flow`) — LLM-backed binary router; defaults to chat on ambiguous input (`src/internal/servers/secondary_llm_flows/search_flow_classification.py`)
- **Intent classifier** (`IntentPipeline`) — trainable feedforward ML model classifying `purchase` / `navigate` / `qa` / `recommendation`; selects fast / balanced / reasoning model tier (`src/model/intent_classifier.py`)
- `KEYWORD_EXPANSION_PROMPT` / `QUERY_TYPE_PROMPT` — broaden sparse queries and classify query intent for retrieval tuning (`src/internal/prompts/query_expansion.py`)

**Admin & Observability**
- `build_admin_surface_summary` — single-call health snapshot: connectors, indexing, users, auth, models, tools, analytics, enterprise controls with a composite health score
- `MonitoringWorker` — background poller for process memory (RSS), index queue depth, connector count; ships JSON snapshots to a cloud data-plane URL
- `event_telemetry` / `identify_user` — PostHog event capture helpers; no-ops when PostHog is not configured
- Feature flags — composable chain: `EnvFeatureFlagProvider` → `PostHogFeatureFlagProvider`; `StaticFeatureFlagProvider` for tests; single call-site via `is_feature_enabled`
- Search history per user (`GET /search/search-history`) and query history with CSV export (`GET /admin/query-history/export`)


## Agentic RAG

```python
from src.agents.agentic_rag import AgenticRAGConfig, AgenticRAGLoop

loop = AgenticRAGLoop(
    AgenticRAGConfig(max_rounds=3, topk=5, retrieval_url="http://localhost:8001/retrieve"),
    llm=my_llm_client,  # any LLMClient; pass None for extractive fallback
)
result = await loop.run("What is FAISS and how does it compare to ScaNN?")
print(result.answer)       # grounded answer with citations
print(result.rounds_used)  # retrieval rounds used
```

Via the web API:

```bash
curl -X POST http://localhost:7860/api/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "What is FAISS?", "mode": "agentic_rag", "top_k": 5}'
```

Loop flow:

1. **Query enhancement** — `QueryEnhancer` decomposes the question into sub-queries and generates a HyDE hypothetical answer; runs once before the loop
2. **Retrieval** — `retrieve_context()` fetches from the configured server (demo, dense, BM25, or hybrid+rerank) for each novel query; unique documents accumulate across rounds
3. **Sufficiency check** — LLM responds yes/no to `_SUFFICIENCY_PROMPT`; if yes (or on the last round), proceed to synthesis
4. **Gap analysis** — if insufficient, `_GAP_ANALYSIS_PROMPT` identifies missing information and emits targeted follow-up queries; loop repeats from step 2 up to `max_rounds`
5. **Grounded synthesis** — `generate_answer()` over all accumulated evidence with inline citations


## Retrieval Setup

**Index documents from Python:**

Use `src.internal.document_index` as the single indexing entry point. It handles
filtering, chunking, embedding, retry-isolated writes, and failure reporting:

```python
from src.internal.document_index import index_documents

result = index_documents(documents, sink=my_chunk_sink)
print(result.successful_chunk_counts)
print(result.failures)
```

Query-time indexing and local retrievers live in `src.internal.document_index`.
Search context contracts and the retrieval HTTP client live in `src.context`.
Reranker utilities live beside their server in `src.internal.servers.retrieval`.

**Retrieval servers** (`src/internal/servers/retrieval/`):

| Module | Description |
|--------|-------------|
| `demo.py` | TF-IDF over corpus.jsonl — no Java required |
| `retrieval_server.py` | BM25 or dense (E5/BGE via FAISS) |
| `retrieval_rerank.py` | Retrieval + cross-encoder reranker |
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

**Start a web search server:**

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


## Training

The training pipeline is modular: generate trajectories → score with rewards → compute advantages → optimize.

| Task | Entry point |
|------|-------------|
| QA parquet preparation | `python3 -m examples.prepare_search_qa_dataset` |
| RAG parquet preparation | `python3 -m examples.prepare_search_rag_dataset` |
| Reward/GRPO smoke test | `python3 -m examples.run_grpo_training_pipeline` |
| GRPO with real agent loops | `src/training/ppo/search_agent_grpo_trainer.py` |
| Bamboogle eval (local/vLLM) | `python3 -m examples.run_bamboogle_eval` |
| Bamboogle eval (judge\_fn CLI) | `python3 -m examples.evaluate_bamboogle` |
| Reward function | `src/training/reward.py` |
| GRPO helpers | `src/training/grpo.py` |
| PPO / policy loss helpers | `src/training/ppo/` |
| Benchmark eval helpers | `src/training/eval/bamboogle.py` |
| Rollout orchestration | `src/model/generation.py` |

**Reward components** (`SearchRewardFunction.reward_components()`):

| Component | What it measures |
|-----------|-----------------|
| `correctness` | `judge_fn(answer, ground_truth)` × `correctness_weight`; judge can be exact-match, token-F1, or an LLM |
| `citation_support` | Fraction of retrieved documents cited in the final answer |
| `subquestion_coverage` | Fraction of decomposed sub-queries with sufficient evidence |
| `search_quality` | Sufficiency verdict + average query quality across search rounds |
| `format_reward` | Inline `[D1]` citation markers present in the answer |
| `per_search_penalty` | Flat penalty × number of search rounds used |
| `unnecessary_search_penalty` | Additional penalty × rounds beyond the first |
| `duplicate_query_penalty` | Penalty × repeated queries issued across turns |
| `unsupported_claim_penalty` | Fires when agent searched, got results, but cited none |

Preset configs (zero-param shortcuts via `SearchRewardConfig`):

```python
from src.training.reward import SearchRewardFunction, SearchRewardConfig

# Minimal: correctness only
reward_fn = SearchRewardFunction(SearchRewardConfig.sparse_final_only())

# Curriculum stages
reward_fn = SearchRewardFunction(SearchRewardConfig.simple_sparse_with_search_penalty())
reward_fn = SearchRewardFunction(SearchRewardConfig.second_pass())
reward_fn = SearchRewardFunction(SearchRewardConfig.third_pass_with_format())
```

**GRPO** — group-relative advantages from G rollouts per prompt:

```python
from src.training.grpo import score_prompt_group, compute_grpo_outcome_advantage

scored = score_prompt_group(
    rollouts,                         # list[GRPORolloutSample]
    ground_truth=reference_answer,
    judge_fn=exact_match_fn,
    reward_fn=reward_fn,
)
rewards = [s.reward for s in scored]
advantages = compute_grpo_outcome_advantage(rewards)  # list[float], group-relative
```

**SearchAgentGRPOTrainer** — GRPO with real agent-loop rollouts (all shaped rewards fire):

```python
from src.training.ppo import SearchAgentGRPOTrainer
from src.agents.search import SearchAgentLoop, SearchAgentLoopConfig

trainer = SearchAgentGRPOTrainer(
    policy=policy,
    reference_policy=ref_policy,
    tokenizer=tokenizer,
    optimizer=optimizer,
    judge_fn=judge_fn,                              # (prediction, ground_truth) -> float
    loop_factory=lambda: SearchAgentLoop(           # one new loop per rollout
        tokenizer=tokenizer,
        server_manager=server_manager,
        search_config=SearchAgentLoopConfig(search_url="http://localhost:8001/retrieve"),
    ),
    reward_fn=SearchRewardFunction(SearchRewardConfig.second_pass()),
    max_concurrent=4,                               # parallel agent loops per step
)
trainer.step(prompts, ground_truths)
```

**Bamboogle benchmark evaluation** — two-hop QA (125 examples from FlashRAG):

```python
from src.training.eval.bamboogle import evaluate_bamboogle, load_bamboogle

summary, rows = evaluate_bamboogle(
    agent,                        # any object with .invoke({"messages": [...]})
    reward_fn=reward_fn,          # optional: score shaped reward alongside EM
    limit=125,
    output_path="results/bamboogle.jsonl",
)
print(f"EM: {summary.exact_match:.3f}  Contains: {summary.contains_match:.3f}")
if summary.avg_reward is not None:
    print(f"Avg reward: {summary.avg_reward:.3f}")
```

**PPO** — clipped policy + value loss with KL penalty:

```python
from src.training.ppo import compute_ppo_policy_loss_core, compute_value_loss, AdaptiveKLController

# Returns (pg_loss, clipfrac, approx_kl, surrogate)
pg_loss, clipfrac, kl, _ = compute_ppo_policy_loss_core(
    old_log_prob=old_logprobs,   # log probs from the frozen reference snapshot
    log_prob=logprobs,           # log probs from the current policy
    advantages=advantages,
    eos_mask=eos_mask,
    cliprange=0.2,
)
# Returns (vf_loss, vf_clipfrac)
vf_loss, _ = compute_value_loss(
    vpreds=values,               # current value predictions
    returns=returns,
    values=old_values,           # value baseline from the reference snapshot
    eos_mask=eos_mask,
    cliprange_value=0.2,
)
kl_ctrl = AdaptiveKLController(init_kl_coef=0.1, target_kl=6.0, horizon=10000)
```

**XML search protocol** — the ReAct-style trace format used by `SearchAgentLoop`:

```xml
<think>decide whether to answer or search</think>
<search>precise query</search>
<information>retrieval results injected by the environment</information>
<answer>final grounded answer</answer>
```

`<information>` is environment output — mask it from policy/SFT action loss.


## MCP Server

The MCP server exposes Agentic Search capabilities as [Model Context Protocol](https://modelcontextprotocol.io/) tools, letting any MCP-compatible client (Claude Desktop, Cursor, etc.) query your knowledge base directly.

**Start the server** (requires the `mcp` extra):

```bash
pip install -e ".[mcp]"   # fastmcp + httpx2
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
| `indexed_sources` | All connector types currently indexed (e.g. `"github"`, `"confluence"`) |

**Debug with MCP Inspector:**

```bash
npx @modelcontextprotocol/inspector http://localhost:8090/
```

**Environment variables:**

| Var | Default | Description |
|-----|---------|-------------|
| `MCP_SERVER_ENABLED` | `false` | Set to `true` to enable |
| `MCP_SERVER_PORT` | `8090` | Bind port |
| `MCP_SERVER_CORS_ORIGINS` | — | Comma-separated allowed origins |
| `API_SERVER_HOST` | `127.0.0.1` | Host of the main API server |


## API Reference

Base URLs: web backend `http://localhost:7860` · retrieval server `http://localhost:8001`

**Generate a dev JWT** (required for all authenticated endpoints):

```bash
export TOKEN=$(python3 -c "
from src.internal.auth import generate_user_jwt_token
print(generate_user_jwt_token(user_id='dev', email='dev@local'))
")
```

**Health**

```bash
curl -s http://localhost:7860/health       # web backend
curl -s http://localhost:8001/health       # retrieval server
curl -s http://localhost:7860/settings     # tier / license state (no auth)
```

**Auth & users**

```bash
curl -s -X POST http://localhost:7860/auth/register \
  -H "Content-Type: application/json" -d '{"email":"dev@local","password":"..."}'
curl -s -X POST http://localhost:7860/auth/login \
  -H "Content-Type: application/json" -d '{"email":"dev@local","password":"..."}'
curl -s http://localhost:7860/me                      -H "Authorization: Bearer $TOKEN"
curl -s http://localhost:7860/me/permissions          -H "Authorization: Bearer $TOKEN"
```

**Search & chat**

```bash
# Agentic search (streaming)
curl -s -X POST http://localhost:7860/api/agent \
  -H "Content-Type: application/json" -d '{"query":"What is FAISS?","mode":"search"}'

# Search-flow classification (search vs chat router)
curl -s -X POST http://localhost:7860/search/search-flow-classification \
  -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" \
  -d '{"query":"What is FAISS?"}'

# Send a search message
curl -s -X POST http://localhost:7860/search/send-search-message \
  -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" \
  -d '{"query":"dense retrieval","search_doc_ids":[]}'

# Chat sessions
curl -s -X POST http://localhost:7860/api/sessions    -H "Authorization: Bearer $TOKEN"
curl -s http://localhost:7860/api/sessions            -H "Authorization: Bearer $TOKEN"
curl -s http://localhost:7860/chat/get-user-chat-sessions -H "Authorization: Bearer $TOKEN"

# Retrieval server
curl -s -X POST http://localhost:8001/retrieve \
  -H "Content-Type: application/json" -d '{"query":"dense retrieval","top_k":3}'
```

**Connectors (admin)**

```bash
curl -s http://localhost:7860/admin/connectors                     -H "Authorization: Bearer $TOKEN"
curl -s -X POST http://localhost:7860/admin/connectors \
  -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" \
  -d '{"name":"my-files","source":"local_file","input_type":"load_state","connector_specific_config":{}}'
curl -s -X POST http://localhost:7860/admin/connector/1/sync       -H "Authorization: Bearer $TOKEN"
```

**Tools (admin)**

```bash
curl -s http://localhost:7860/admin/tools                          -H "Authorization: Bearer $TOKEN"
curl -s -X POST http://localhost:7860/admin/tools/my_tool/invoke \
  -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" \
  -d '{"input":{"query":"test"}}'
```

**Analytics (admin)**

```bash
curl -s "http://localhost:7860/analytics/query?start=2024-01-01&end=2025-12-31" \
  -H "Authorization: Bearer $TOKEN"
curl -s "http://localhost:7860/analytics/by-llm?start=2024-01-01&end=2025-12-31" \
  -H "Authorization: Bearer $TOKEN"
curl -s "http://localhost:7860/analytics/by-flow?start=2024-01-01&end=2025-12-31" \
  -H "Authorization: Bearer $TOKEN"
```

**Query history (admin)**

```bash
curl -s http://localhost:7860/admin/chat-sessions                  -H "Authorization: Bearer $TOKEN"
curl -s http://localhost:7860/admin/query-history/audit            -H "Authorization: Bearer $TOKEN"
curl -s http://localhost:7860/admin/query-history/export           -H "Authorization: Bearer $TOKEN"
```

**Hooks (admin)**

```bash
curl -s http://localhost:7860/admin/hooks/specs                    -H "Authorization: Bearer $TOKEN"
curl -s http://localhost:7860/admin/hooks                          -H "Authorization: Bearer $TOKEN"
curl -s -X POST http://localhost:7860/admin/hooks/1/activate       -H "Authorization: Bearer $TOKEN"
```

**Token rate limits (admin)**

```bash
curl -s http://localhost:7860/admin/token-rate-limits/users        -H "Authorization: Bearer $TOKEN"
curl -s http://localhost:7860/admin/token-rate-limits/user-groups  -H "Authorization: Bearer $TOKEN"
```

**User groups (admin)**

```bash
curl -s http://localhost:7860/manage/admin/user-group              -H "Authorization: Bearer $TOKEN"
```

**Evals**

```bash
curl -s -X POST http://localhost:7860/evals/eval_run \
  -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" \
  -d '{"questions":["What is FAISS?"]}'
```

**Observability (admin)**

```bash
curl -s http://localhost:7860/admin/observability/summary          -H "Authorization: Bearer $TOKEN"
```

**Reporting (admin)**

```bash
curl -s http://localhost:7860/admin/usage-report                   -H "Authorization: Bearer $TOKEN"
curl -s -X POST http://localhost:7860/admin/usage-report           -H "Authorization: Bearer $TOKEN"
```

**Billing (admin)**

```bash
curl -s http://localhost:7860/admin/billing/billing-information    -H "Authorization: Bearer $TOKEN"
```

**License**

```bash
curl -s http://localhost:7860/license                              -H "Authorization: Bearer $TOKEN"
curl -s http://localhost:7860/license/seats                        -H "Authorization: Bearer $TOKEN"
```

**OAuth**

```bash
curl -s -X POST http://localhost:7860/oauth/prepare-authorization-request \
  -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" \
  -d '{"connector":"google_drive","redirect_on_success":"/connectors"}'
```

**SCIM** (SCIM bearer token, not a JWT)

```bash
curl -s http://localhost:7860/scim/v2/ServiceProviderConfig        # no auth
curl -s http://localhost:7860/scim/v2/Users   -H "Authorization: Bearer $SCIM_TOKEN"
curl -s http://localhost:7860/scim/v2/Groups  -H "Authorization: Bearer $SCIM_TOKEN"
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

- Dense retrieval defaults to CPU; set `--device cuda` only on a dedicated retrieval node.
- BM25 serving requires Java because Pyserini uses Lucene.
- Empty or invalid queries return empty result lists.
- Some web pages block scraping or return little usable text.
- Google Custom Search and SerpAPI are subject to their own quota and billing rules.
- If `prepare_search_qa_dataset` fails with a `pyarrow` extension error, run `pip install -r requirements.txt`.
