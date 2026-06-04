# Agentic Search

A retrieval-backed agent and search-policy platform combining a FastAPI server layer with local dense/sparse retrieval, multi-turn agent traces, and RL training helpers.

🔍 **Agentic RAG** — Best-in-class search and answer quality via hybrid index + AI Agents for information retrieval. Benchmark to release soon!

🔬 **Deep Research** — In-depth reports with a multi-step research flow. Top of [leaderboard](https://github.com/onyx-dot-app/onyx_deep_research_bench) as of Feb 2026.

🤖 **Custom Agents** — Build AI Agents with unique instructions, knowledge, and actions.

🌍 **Web Search** — Browse the web for up-to-date information. Supports Serper, Google PSE, Brave, SearXNG, and others. Comes with an in-house web crawler and support for Firecrawl/Exa.

🧠 **PPO/GRPO Reward** — Train search agents with composite reward shaping, group-relative advantages, and PPO/GRPO/REINFORCE helpers. Plug in any LLM and retrieval backend.

| Feature | Key modules |
|---------|-------------|
| 🔍 Agentic RAG | `src/agents/agentic_rag.py`, `src/context/query_enhancer.py`, `src/backend/servers/retrieval/hybrid_rerank.py` |
| 🔬 Deep Research | `src/agents/deep_research/`, `src/context/`, `src/retrieval/` |
| 🤖 Custom Agents | `src/agents/custom.py`, `src/tools/`, `src/backend/servers/query_and_chat/` |
| 🌍 Web Search | `src/backend/servers/retrieval/google.py`, `serp.py`, `browser.py` |
| 🧠 PPO/GRPO Reward | `src/training/reward.py`, `src/training/grpo.py`, `src/training/ppo/` |
| 🔗 RAG over connectors | `src/backend/connectors/`, `src/backend/servers/documents/` |
| 📦 Document indexing | `src/backend/servers/backgroundworker/`, `src/retrieval/index_builder.py` |
| 🔒 Permission-aware retrieval | `src/backend/access/`, `src/context/preprocessing/` |
| 📊 Admin & observability | `src/backend/servers/analytics/`, `settings/`, `reporting/`, `license/` |

## Repository Structure

```
src/
├── agents/                      # Agent loops (SearchAgentLoop, ToolAgentLoop, …)
├── backend/                     # Backend services
│   ├── access/                  # Access control & ACL helpers
│   ├── auth/                    # Authentication & authorization
│   ├── chat/                    # Chat functionality & LLM interactions
│   ├── configs/                 # Environment-based configuration (AppSettings)
│   ├── connectors/              # Data source connectors
│   ├── db/                      # Database models & operations (AgenticSearchStore)
│   ├── document_index/          # Vespa integration
│   ├── federated_connectors/    # External search connectors
│   ├── feature_flags/           # Feature-flag providers
│   ├── hooks/                   # Outbound webhook execution
│   ├── llm/                     # LLM provider integrations
│   ├── prompts/                 # Prompt templates
│   ├── search/                  # Search query processing
│   ├── secondary_llm_flows/     # Query expansion, flow classification
│   ├── servers/                 # API endpoints & routers
│   │   ├── backgroundworker/    # Background workers (beat, docfetching, light, heavy, …)
│   │   ├── analytics/           # Usage analytics API
│   │   ├── billing/             # Stripe billing proxy
│   │   ├── documents/           # Connector-credential pair management
│   │   ├── middleware/          # License enforcement, tier gate, tenant tracking
│   │   ├── oauth/               # OAuth 2.0 connector authorization
│   │   ├── query_and_chat/      # Search and chat endpoints
│   │   ├── reporting/           # Usage report ZIP generation
│   │   ├── retrieval/           # Dense/sparse/rerank server entry points
│   │   ├── scim/                # SCIM 2.0 user & group provisioning
│   │   ├── tenants/             # Multi-tenant provisioning & management
│   │   └── web/                 # FastAPI app assembly
│   └── utils/                   # License, encryption, telemetry utilities
├── context/                     # Retrieval-grounded context & prompt builders
├── model/                       # LLM generation, intent classifier, tensor helpers
├── retrieval/                   # Dense/sparse retrievers, indexing pipeline, embedders
├── tools/                       # Tool schemas, search tools, OpenAPI tool registry
└── training/                    # SFT, rewards, PPO, GRPO helpers
tests/                           # Unit and integration test suites
examples/                        # Runnable CLI examples
```

The FastAPI application is assembled in `src/backend/servers/web/app.py`. Every feature area is a self-contained router factory registered via `_register_routers()`. The SQLite-backed `AgenticSearchStore` (`src/backend/db/`) is the single persistence layer — no Postgres, Redis, or Celery required for local deployments.


## Install

Requires Python 3.10+.

```bash
pip install -e .          # one-time; makes src importable as a package
pip install -r requirements.txt
```

For BM25, Java must be available. On Apple Silicon, FAISS is usually more stable installed through conda:

```bash
conda install -c conda-forge faiss-cpu
```

Optional environment variables:

```bash
GOOGLE_API_KEY=...
GOOGLE_CSE_ID=...
SERP_API_KEY=...
JAVA_HOME=/path/to/java
```


## Quick Start

Run the three-process local stack:

```bash
# Terminal 1 — retrieval server (TF-IDF demo, no Java required)
python3 -m src.backend.servers.retrieval.demo --corpus_path data/corpus.jsonl

# Terminal 2 — web backend
uvicorn src.backend.servers.web.app:app --host 127.0.0.1 --port 7860

# Terminal 3 — frontend dev server
cd web && npm install && npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api/*` to port 7860 during development. For production, `npm run build` produces `web/dist`; the FastAPI app serves it automatically.


## Examples

All examples run without a live model or retrieval server unless noted.

**Agent loops**

```bash
# Deterministic SearchAgentLoop trace with fake model + search (no GPU required)
python3 -m examples.run_search_trace_workflow

# Build a matching SFT training record from the trace
python3 -m examples.run_search_trace_workflow --sft

# Minimal SearchAgentLoop usage — shows the public API directly
python3 -m examples.run_search_agent_loop

# Search pipeline with access filters and permission filtering
python3 -m examples.run_search_pipeline
```

**Agent CLI** (requires a running retrieval server and optionally a vLLM endpoint)

```bash
# One-shot local generation (CPU, no retrieval server)
python3 -m examples.run_agentic_search \
  --mode single --question "What is FAISS?" \
  --model Qwen/Qwen2.5-1.5B-Instruct --local --device cpu

# Multi-turn search agent against a live retrieval server
python3 -m examples.run_agentic_search \
  --mode search --question "Compare dense and sparse retrieval" \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --vllm_url http://localhost:8080 --search_url http://localhost:8000/retrieve
```

**PPO/GRPO reward training**

```bash
# End-to-end reward + GRPO advantage smoke test (no GPU required)
python3 -m examples.run_grpo_training_pipeline
```

**Intent classifier**

```bash
# Generate training examples from a local corpus, then train the classifier
python3 -m examples.run_intent_training generate \
  --corpus data/corpus.jsonl \
  --vocabulary data/vocabulary_corpus.json \
  --output data/intent_examples.json

python3 -m examples.run_intent_training train \
  --examples data/intent_examples.json \
  --output models/intent_classifier.pt
```

**Dataset preparation**

```bash
# Prepare NQ search-QA parquet (downloads from HuggingFace)
python3 -m examples.prepare_search_qa_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets \
  --dataset_config nq --local_dir data/nq_search

# Preview 20 examples before writing
python3 -m examples.prepare_search_qa_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets \
  --dataset_config nq --splits test --max_examples 20 --preview --preview_rows 5

# Prepare RAG parquet from cached retrieval results
python3 -m examples.prepare_search_rag_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets \
  --dataset_config nq \
  --corpus_path data/wiki-18.jsonl \
  --train_retrieval_cache data/nq_train_retrieval_cache.json \
  --test_retrieval_cache data/nq_test_retrieval_cache.json \
  --topk 3 --local_dir data/nq_rag
```


## Features

**Retrieval, Indexing & Search**
- **Hybrid + rerank** — dense (FAISS/E5) + sparse (BM25) RRF fusion with cross-encoder reranking in a single `/retrieve` endpoint
- **Query enhancer** — `QueryEnhancer.decompose()` and `.hyde()` enrich any query; degrades gracefully without an LLM
- Local dense retrieval with FAISS-compatible indexes (E5, BGE, custom embedders)
- Local sparse retrieval with BM25/Pyserini
- Web search via Google Custom Search, SerpAPI, and playwright-cli (no API key required)
- FAISS index builder — chunk, embed, and write dense indexes from a JSONL corpus (`src/retrieval/index_builder.py`)
- BM25 index builder — build Pyserini/Lucene sparse indexes from the same corpus
- Background indexing pipeline — async workers fetch, parse, chunk, enrich, embed, and index documents; supports mini-chunks, vector-write retries, and document prefiltering (`src/backend/servers/backgroundworker/`)
- **Connectors** — collect documents from multiple data sources into the indexing pipeline (`src/backend/connectors/`):
  - `LocalFileConnector` / `LocalFilePollConnector` — ingest UTF-8 files from paths, directories, or globs; poll for changes by time window
  - `SearchConnector` — pull search results as documents via retrieval, Google, or SerpAPI
  - `InMemoryConnector` — load documents from Python objects for testing and prototyping
  - `OAuthConnector` interface — authorization-code flow for external sources (Google Drive, Slack, Confluence, GitHub, Jira, SharePoint, Salesforce, Zendesk, Notion)
  - `PollConnector` / `CheckpointedConnector` / `SlimConnector` — incremental sync patterns with time-window, checkpoint, and permission-metadata variants

**Agent Loops**
- **Agentic RAG** (`AgenticRAGLoop`) — multi-hop query decomposition, HyDE, iterative retrieval with evidence sufficiency gating, and grounded synthesis with citations
- Multi-turn `SearchAgentLoop` traces with `<think>`, `<search>`, `<information>`, `<fetch>`, and `<answer>` actions
- `ToolAgentLoop` — generic tool-calling loop usable from both search and chat flows

**LLM Backends**
- `OpenAICompatibleLLM` — single client for any OpenAI-compatible API: OpenAI, Azure OpenAI, Anthropic (via compatibility layer), Ollama, LiteLLM proxy, and vLLM (`src/backend/llm/providers.py`)
- `VLLMServerManager` — call any vLLM / Ollama / LiteLLM completions endpoint for server-backed inference
- `LocalServerManager` — load a HuggingFace model in-process (Qwen, Llama, Mistral, etc.) for CPU or GPU inference without a separate server
- Configured via four env vars: `GEN_AI_MODEL_PROVIDER`, `GEN_AI_MODEL_VERSION`, `GEN_AI_API_KEY`, `GEN_AI_API_BASE`

**Tool Use**
- Hermes, Llama-3, and JSON tool-call parsers — parse structured tool calls from any LLM output format
- OpenAPI-based `ApiToolRegistry` — load and execute tools from any OpenAPI 3.x schema at runtime
- `FunctionTool` — wrap any Python callable as a tool with auto-generated JSON schema
- `build_search_tool` — ready-made tool that dispatches to retrieval, Google, or SerpAPI

**RL Training**
- Composite reward shaping (`SearchRewardFunction`) with format, search-use, answer-length, and exact-match components
- Group-relative advantage helpers for PPO, GRPO, and REINFORCE-style experiments
- PPO core: clipped policy loss, value loss, entropy, KL penalty, adaptive and fixed KL controllers
- Training data builders for search-QA and RAG parquet datasets

**Chat Processing**
- `run_llm_loop` — multi-turn LLM execution loop: builds message history, dispatches tool calls, injects context files, streams tokens, and handles forced tool invocations (`src/backend/chat/llm_loop.py`)
- `run_llm_step` — single LLM step: sends the prompt, streams the response, extracts tool calls, and returns a structured `LlmStepResult` (`src/backend/chat/llm_step.py`)
- `process_message` — top-level chat turn orchestrator: resolves persona, tools, files, search config, and LLM; dispatches to `run_llm_loop`; persists the turn via `save_chat_turn` (`src/backend/chat/process_message.py`)
- `save_chat_turn` — persists messages, tool calls, and search doc references to the SQLite store after each turn (`src/backend/chat/save_chat.py`)
- **Dynamic citation processor** (`DynamicCitationProcessor`) — streams LLM token output and extracts citation markers in REMOVE / KEEP / HYPERLINK modes; emits `CitationInfo` objects in real time (`src/backend/chat/citation_processor.py`)
- **Chat history compression** (`compress_chat_history`) — summarises older turns with an LLM when context exceeds the token budget; summaries are branch-aware and keyed to the last message (`src/backend/chat/compression.py`)
- **Emitter** — routes packets (tokens, tool calls, citations) from worker threads onto a shared queue for ordered streaming to the HTTP response (`src/backend/chat/emitter.py`)
- **Stop signal checker** — lets a client cancel an in-progress stream; `set_fence` / `is_connected` / `reset_cancel_status` operate on the `CacheBackend` (`src/backend/chat/stop_signal_checker.py`)
- **System prompt builder** (`build_system_prompt`) — assembles the system prompt from persona instructions, knowledge base snippets, tool descriptions, and memory context (`src/backend/chat/prompt_utils.py`)

**Cache & Persistence**
- **SQLite store** (`AgenticSearchStore`) — single repository for connector configs, documents, document permissions, chat sessions & messages, indexing attempts, usage reports, rate-limit rules, SCIM tokens, and standard answers; no external database required (`src/backend/db/store.py`)
- **Search history** — past search queries persisted per user and retrievable via `GET /search/search-history` (`src/backend/servers/query_and_chat/search_backend.py`)
- **Query history** — full chat session history with time/feedback filters, exportable as CSV via `GET /admin/query-history/export` (`src/backend/servers/query_history/`)
- **In-memory cache** (`CacheBackend` / `InMemoryCache`) — tracks in-flight chat session state (processing flag, stop signal, cancel status) during streaming responses (`src/backend/cache/`)
- **Chunk batch store** (`ChunkBatchStore`) — temp disk buffer that decouples embedding from index insertion during large indexing jobs; auto-cleaned on exit (`src/retrieval/chunk_batch_store.py`)
- **File store** (`InMemoryChatFile`) — holds uploaded files (images, PDFs, plain text) in memory for the duration of a chat turn (`src/backend/file_store/`)

**Query Classification**
- **Search vs chat classifier** (`classify_is_search_flow`) — LLM-backed binary classifier that routes each query to document search or direct chat; defaults to chat on ambiguous input (`src/backend/secondary_llm_flows/search_flow_classification.py`)
- **Intent classifier** (`IntentPipeline`) — trainable feedforward ML model that classifies queries into `purchase`, `navigate`, `qa`, `recommendation` and selects the appropriate model tier (fast / balanced / reasoning) (`src/model/intent_classifier.py`)


## Run An Agent

`examples/run_agentic_search.py` is the main CLI.

```bash
# Local inference (CPU)
python3 -m examples.run_agentic_search \
  --mode single --question "What is FAISS?" \
  --model Qwen/Qwen2.5-1.5B-Instruct --local --device cpu

# Server-backed search mode
python3 -m examples.run_agentic_search \
  --mode search --question "Compare dense and sparse retrieval" \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --vllm_url http://localhost:8080 --search_url http://localhost:8000/retrieve
```

| Mode | Loop | Use it for |
|------|------|------------|
| `single` | `PlainGenerationLoop` | Simple local generation smoke tests |
| `search` | `SearchAgentLoop` | Multi-turn search traces for RAG, SFT, and RL |
| `tool` | `ToolAgentLoop` | Generic function/tool-calling experiments |


## Agentic RAG

`AgenticRAGLoop` delivers best-in-class answer quality by combining query enhancement, iterative hybrid retrieval, and evidence-gated synthesis.

```python
from src.agents.agentic_rag import AgenticRAGConfig, AgenticRAGLoop

loop = AgenticRAGLoop(
    AgenticRAGConfig(max_rounds=3, topk=5, retrieval_url="http://localhost:8000/retrieve"),
    llm=my_llm_client,  # any LLMClient; pass None for extractive fallback
)
result = await loop.run("What is FAISS and how does it compare to ScaNN?")
print(result.answer)       # grounded answer with citations
print(result.rounds_used)  # how many retrieval rounds were needed
```

Via the web API — pass `"mode": "agentic_rag"` to `POST /api/agent`:

```bash
curl -X POST http://localhost:7860/api/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "What is FAISS?", "mode": "agentic_rag", "top_k": 5}'
```

Loop flow per query:

1. **Query enhancement** — decompose into sub-queries; generate HyDE hypothetical answer
2. **Hybrid+rerank retrieval** — retrieve for each enhanced query; accumulate unique documents
3. **Sufficiency check** — LLM judges whether context is enough; break or continue
4. **Follow-up generation** — if insufficient, LLM proposes targeted follow-up queries
5. **Grounded synthesis** — answer from all accumulated evidence with inline citations


## Local Retrieval

```bash
# Dense retrieval server
python3 -m src.backend.servers.retrieval.retrieval \
  --model_path intfloat/e5-base-v2 \
  --index_path indexes/e5_Flat.index \
  --corpus_path data/corpus.jsonl \
  --retrieval_method e5 --device cpu --topk 5

# Sparse BM25 server
python3 -m src.backend.servers.retrieval.retrieval \
  --index_path indexes/bm25 --corpus_path data/corpus.jsonl \
  --retrieval_method bm25

# Health/test check
curl -i -sS http://127.0.0.1:8000/health
curl -i -sS -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"What is FAISS?","top_k":5}'
```

Retrieval servers available under `src/backend/servers/retrieval/`:

| Module | Description |
|--------|-------------|
| `demo.py` | TF-IDF over corpus.jsonl — no Java required |
| `retrieval.py` | BM25 or dense (E5/BGE via FAISS) |
| `retrieval_rerank.py` | Retrieval + cross-encoder reranker |
| `hybrid_rerank.py` | Dense + BM25 RRF fusion + rerank (recommended for `AgenticRAGLoop`) |
| `google.py` | Google Custom Search proxy |
| `serp.py` | SerpAPI proxy |
| `browser.py` | playwright-cli browser automation; no API key, ~5–10s/query |


## Build Indexes

```bash
# Dense FAISS index
python3 -m src.retrieval.index_builder \
  --retrieval_method e5 --model_path intfloat/e5-base-v2 \
  --corpus_path data/corpus.jsonl --faiss_type Flat --save_dir indexes/

# BM25 index
python3 -m src.retrieval.index_builder \
  --retrieval_method bm25 --corpus_path data/corpus.jsonl --save_dir indexes/
```


## Hybrid Retrieval + Rerank

```bash
# Pure dense (no BM25 index required)
python3 -m src.backend.servers.retrieval.hybrid_rerank \
  --dense_model intfloat/e5-base-v2 \
  --index_path indexes/e5_Flat.index \
  --corpus_path data/corpus.jsonl \
  --retrieval_topk 10 --rerank_topk 5

# Hybrid dense + BM25
python3 -m src.backend.servers.retrieval.hybrid_rerank \
  --dense_model intfloat/e5-base-v2 \
  --index_path indexes/e5_Flat.index \
  --corpus_path data/corpus.jsonl \
  --sparse_index_path indexes/bm25 \
  --hybrid_alpha 0.5 --retrieval_topk 10 --rerank_topk 5
```


## Web Search

`src.tools.search` routes calls to `retrieval`, `google`, or `serpapi`. Missing API keys return structured tool errors.

```bash
python3 -m src.backend.servers.retrieval.serp \
  --search_url "https://serpapi.com/search" --topk 3 --serp_api_key "$SERP_API_KEY"

python3 -m src.backend.servers.retrieval.google \
  --api_key "$GOOGLE_API_KEY" --topk 5 --cse_id "$GOOGLE_CSE_ID" --snippet_only
```


## Training Flow

The training pipeline is intentionally modular:

1. Generate trajectories with `SearchAgentLoop` or `LLMGenerationManager`
2. Score trajectories with `SearchRewardFunction`
3. Compute group-relative advantages with `src.training.grpo`
4. Save JSONL batches or compute token log probabilities
5. Optimize with PPO/GRPO/REINFORCE helpers in `src.model.generation` and `src.training.ppo`

| Task | Command or module |
|------|-------------------|
| Deterministic trace | `python3 -m examples.run_search_trace_workflow` |
| SFT record from trace | `python3 -m examples.run_search_trace_workflow --sft` |
| QA parquet preparation | `python3 -m examples.prepare_search_qa_dataset` |
| Reward/GRPO smoke test | `python3 -m examples.run_grpo_training_pipeline` |
| Reward function | `src/training/reward.py` |
| GRPO helpers | `src/training/grpo.py` |
| PPO helpers | `src/training/ppo/` |
| Generation and policy loss | `src/model/generation.py` |

Prepare NQ/FlashRAG-style QA pairs for training:

```bash
python3 -m examples.prepare_search_qa_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets \
  --dataset_config nq --local_dir data/nq_search

# Preview before writing parquet
python3 -m examples.prepare_search_qa_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets \
  --dataset_config nq --splits test --max_examples 20 --preview --preview_rows 5
```


## PPO/GRPO Reward

`SearchRewardFunction` scores trajectories across four composable components:

| Component | What it measures |
|-----------|-----------------|
| `format` | Well-formed XML trace with required action tags |
| `search_use` | Whether the agent issued at least one search action |
| `answer_length` | Answer is within acceptable token bounds |
| `exact_match` | Token-overlap correctness against reference answers |

```python
from src.training.reward import SearchRewardFunction, SearchRewardConfig

reward_fn = SearchRewardFunction(SearchRewardConfig(
    format_weight=0.2,
    search_use_weight=0.3,
    length_weight=0.1,
    exact_match_weight=0.4,
))
scores = reward_fn(trajectories, reference_answers)
```

**GRPO** — compute group-relative advantages from G rollouts per prompt:

```python
from src.training.grpo import score_prompt_group, compute_grpo_outcome_advantage

scored = score_prompt_group(rollouts, reward_fn, reference_answer)
advantages = compute_grpo_outcome_advantage(scored)
```

**PPO** — clipped policy + value loss with KL penalty:

```python
from src.training.ppo import (
    compute_ppo_policy_loss_core,
    compute_value_loss,
    AdaptiveKLController,
)

policy_loss = compute_ppo_policy_loss_core(logprobs, old_logprobs, advantages, clip_eps=0.2)
value_loss  = compute_value_loss(values, returns, old_values, clip_eps=0.2)
```

Run the end-to-end smoke test:

```bash
python3 -m examples.run_grpo_training_pipeline
```


## XML Search Protocol

The search-agent trace uses a compact ReAct-style protocol:

```xml
<think>decide whether to answer or search</think>
<search>precise query</search>
<information>retrieval results injected by the environment</information>
<answer>final grounded answer</answer>
```

`<information>` is environment output and should be masked out of policy/SFT action loss.


## Configuration

All settings are loaded from environment variables via `src/backend/configs/AppSettings`:

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
| `OAUTH_SLACK_CLIENT_ID` | — | Slack OAuth app client ID |
| `OAUTH_CONFLUENCE_CLOUD_CLIENT_ID` | — | Confluence OAuth app client ID |
| `OAUTH_GOOGLE_DRIVE_CLIENT_ID` | — | Google Drive OAuth app client ID |


## API Health Checks

All checks assume the web backend is running on `http://localhost:7860` and the retrieval server on `http://localhost:8000`.

**Generate a dev JWT** (required for admin endpoints):

```bash
export TOKEN=$(python3 -c "
from src.backend.auth import generate_user_jwt_token
print(generate_user_jwt_token(user_id='dev', email='dev@local'))
")
```

**Core health**

```bash
# Web server
curl -s http://localhost:7860/health

# Retrieval server
curl -s http://localhost:8000/health

# Application tier / license status (no auth required)
curl -s http://localhost:7860/settings
```

**Search and chat**

```bash
# One-shot agent query
curl -s -X POST http://localhost:7860/api/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "What is FAISS?", "mode": "search"}'

# List chat sessions
curl -s http://localhost:7860/api/sessions \
  -H "Authorization: Bearer $TOKEN"

# Retrieval endpoint
curl -s -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "dense retrieval", "top_k": 3}'
```

**Admin — analytics, billing, reporting** (requires admin JWT)

```bash
# Daily query analytics
curl -s "http://localhost:7860/analytics/query?start=2024-01-01&end=2025-12-31" \
  -H "Authorization: Bearer $TOKEN"

# Billing status
curl -s http://localhost:7860/admin/billing/billing-information \
  -H "Authorization: Bearer $TOKEN"

# List usage reports
curl -s http://localhost:7860/admin/usage-report \
  -H "Authorization: Bearer $TOKEN"
```

**Admin — connectors, hooks, rate limits**

```bash
# List webhook specs
curl -s http://localhost:7860/admin/hooks/specs \
  -H "Authorization: Bearer $TOKEN"

# List configured hooks
curl -s http://localhost:7860/admin/hooks \
  -H "Authorization: Bearer $TOKEN"

# Token rate limits (users)
curl -s http://localhost:7860/admin/token-rate-limits/users \
  -H "Authorization: Bearer $TOKEN"

# Web-search provider config
curl -s http://localhost:7860/admin/web-search/search-providers \
  -H "Authorization: Bearer $TOKEN"
```

**Admin — license**

```bash
curl -s http://localhost:7860/license \
  -H "Authorization: Bearer $TOKEN"

curl -s http://localhost:7860/license/seats \
  -H "Authorization: Bearer $TOKEN"
```

**SCIM** (uses a SCIM bearer token, not a JWT)

```bash
# Capability advertisement — no auth
curl -s http://localhost:7860/scim/v2/ServiceProviderConfig
curl -s http://localhost:7860/scim/v2/ResourceTypes
curl -s http://localhost:7860/scim/v2/Schemas

# User and group list — requires SCIM token
export SCIM_TOKEN=<token-from-POST-/scim/v2/tokens>
curl -s http://localhost:7860/scim/v2/Users \
  -H "Authorization: Bearer $SCIM_TOKEN"
curl -s http://localhost:7860/scim/v2/Groups \
  -H "Authorization: Bearer $SCIM_TOKEN"
```


## Tests

```bash
# Unit tests
pytest                                      # full suite
pytest tests/unit/ -v                       # unit only
pytest tests/unit/servers/ -v               # server-focused
pytest tests/unit/test_reward.py tests/unit/test_grpo.py tests/unit/test_llm_agent_generation.py -v

# Integration tests (requires live server at API_SERVER_URL, default http://localhost:8080)
pytest tests/integration/ -v
API_SERVER_HOST=localhost API_SERVER_PORT=8080 pytest tests/integration/
```

Unit test coverage:

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

- Dense retrieval defaults to CPU to avoid competing with trainer GPU memory; set `--device cuda` only on a dedicated retrieval node.
- Empty or invalid queries return empty result lists.
- Some web pages block scraping or return little usable text.
- Google Custom Search and SerpAPI are subject to their own quota and billing rules.
- BM25 serving requires Java because Pyserini uses Lucene.
- If `prepare_search_qa_dataset` fails with a `pyarrow` extension error, refresh with `pip install -r requirements.txt`.
