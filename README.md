# Agentic Search

A retrieval-backed agent and search-policy platform — FastAPI server, local dense/sparse retrieval, multi-turn agent traces, and RL training helpers.

🔍 **Agentic RAG** — Best-in-class search and answer quality via hybrid index + AI Agents. Benchmark to release soon!

🔬 **Deep Research** — In-depth reports with a multi-step research flow. Top of [leaderboard](https://github.com/onyx-dot-app/onyx_deep_research_bench) as of Feb 2026.

🤖 **Custom Agents** — Build AI Agents with unique instructions, knowledge, and actions.

🌍 **Web Search** — Browse the web for up-to-date information. Supports Google PSE, SerpAPI, Brave, SearXNG, Firecrawl/Exa, and playwright-cli.

🧠 **PPO/GRPO Reward** — Train search agents with composite reward shaping, group-relative advantages, and PPO/GRPO/REINFORCE helpers.

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
├── backend/
│   ├── access/                  # Access control & ACL helpers
│   ├── auth/                    # Authentication & authorization
│   ├── cache/                   # In-memory cache backend (chat session state)
│   ├── chat/                    # Chat pipeline (loop, steps, citations, compression)
│   ├── configs/                 # Environment-based configuration (AppSettings)
│   ├── connectors/              # Data source connectors
│   ├── db/                      # SQLite store (AgenticSearchStore)
│   ├── document_index/          # Vespa integration
│   ├── feature_flags/           # Feature-flag providers (env, PostHog, composite)
│   ├── file_store/              # In-memory chat file handling
│   ├── hooks/                   # Outbound webhook execution
│   ├── llm/                     # LLM provider integrations
│   ├── observability/           # Admin surface summary & health score
│   ├── prompts/                 # Prompt templates
│   ├── secondary_llm_flows/     # Search-vs-chat flow classification
│   ├── servers/
│   │   ├── backgroundworker/    # Async workers (beat, docfetching, light, heavy, monitoring)
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

The FastAPI app is assembled in `src/backend/servers/web/app.py`. Every feature area is a self-contained router factory. `AgenticSearchStore` (SQLite) is the single persistence layer — no Postgres, Redis, or Celery required locally.


## Install

Requires Python 3.10+.

```bash
pip install -e .          # one-time; makes src importable as a package
pip install -r requirements.txt
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

```bash
# Terminal 1 — retrieval server (TF-IDF demo, no Java required)
python3 -m src.backend.servers.retrieval.demo --corpus_path data/corpus.jsonl

# Terminal 2 — web backend
uvicorn src.backend.servers.web.app:app --host 127.0.0.1 --port 7860

# Terminal 3 — frontend
cd web && npm install && npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api/*` to port 7860. For production, `npm run build` produces `web/dist`; the FastAPI app serves it automatically.


## Examples

All examples run without a live model or retrieval server unless noted.

**Agent loops**

```bash
python3 -m examples.run_search_trace_workflow          # deterministic trace (no GPU)
python3 -m examples.run_search_trace_workflow --sft    # build SFT record from trace
python3 -m examples.run_search_agent_loop              # minimal SearchAgentLoop usage
python3 -m examples.run_search_pipeline                # pipeline with access filters
```

**Agent CLI** (requires retrieval server; `--vllm_url` optional)

```bash
# Local CPU inference
python3 -m examples.run_agentic_search \
  --mode single --question "What is FAISS?" \
  --model Qwen/Qwen2.5-1.5B-Instruct --local --device cpu

# Server-backed multi-turn search
python3 -m examples.run_agentic_search \
  --mode search --question "Compare dense and sparse retrieval" \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --vllm_url http://localhost:8080 --search_url http://localhost:8000/retrieve
```

| Mode | Loop | Use it for |
|------|------|------------|
| `single` | `PlainGenerationLoop` | Local generation smoke tests |
| `search` | `SearchAgentLoop` | Multi-turn RAG, SFT, and RL traces |
| `tool` | `ToolAgentLoop` | Generic tool-calling experiments |

**PPO/GRPO reward**

```bash
python3 -m examples.run_grpo_training_pipeline         # end-to-end reward + GRPO (no GPU)
```

**Intent classifier**

```bash
python3 -m examples.run_intent_training generate \
  --corpus data/corpus.jsonl --vocabulary data/vocabulary_corpus.json \
  --output data/intent_examples.json

python3 -m examples.run_intent_training train \
  --examples data/intent_examples.json --output models/intent_classifier.pt
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


## Features

**Retrieval, Indexing & Search**
- **Hybrid + rerank** — dense (FAISS/E5) + sparse (BM25) RRF fusion with cross-encoder reranking in a single `/retrieve` endpoint
- **Query enhancer** — `QueryEnhancer.decompose()` and `.hyde()` enrich any query; degrades gracefully without an LLM
- Local dense retrieval with FAISS-compatible indexes (E5, BGE, custom embedders)
- Local sparse retrieval with BM25/Pyserini
- Web search via Google Custom Search, SerpAPI, and playwright-cli
- FAISS and BM25 index builders from a JSONL corpus (`src/retrieval/index_builder.py`)
- Background indexing pipeline — async workers fetch, parse, chunk, enrich, embed, and index; supports mini-chunks, vector-write retries, and document prefiltering
- **Connectors** (`src/backend/connectors/`) — collect documents from multiple sources:
  - `LocalFileConnector` / `LocalFilePollConnector` — UTF-8 files from paths, directories, or globs
  - `SearchConnector` — search results as documents via retrieval, Google, or SerpAPI
  - `InMemoryConnector` — Python objects for testing and prototyping
  - `OAuthConnector` — authorization-code flow for Google Drive, Slack, Confluence, GitHub, Jira, SharePoint, Salesforce, Zendesk, Notion
  - `PollConnector` / `CheckpointedConnector` / `SlimConnector` — incremental sync with time-window, checkpoint, and permission-metadata variants

**Agent Loops**
- **Agentic RAG** (`AgenticRAGLoop`) — multi-hop query decomposition, HyDE, iterative retrieval with evidence sufficiency gating, and grounded synthesis with citations
- Multi-turn `SearchAgentLoop` traces with `<think>`, `<search>`, `<information>`, `<fetch>`, and `<answer>` actions
- `ToolAgentLoop` — generic tool-calling loop usable from both search and chat flows

**LLM Backends**
- `OpenAICompatibleLLM` — single client for OpenAI, Azure OpenAI, Anthropic, Ollama, LiteLLM, and vLLM (`src/backend/llm/providers.py`)
- `VLLMServerManager` — server-backed inference via any OpenAI-compatible endpoint
- `LocalServerManager` — in-process HuggingFace models (Qwen, Llama, Mistral, etc.) on CPU or GPU
- Configured via `GEN_AI_MODEL_PROVIDER`, `GEN_AI_MODEL_VERSION`, `GEN_AI_API_KEY`, `GEN_AI_API_BASE`

**Tool Use**
- Hermes, Llama-3, and JSON tool-call parsers
- `ApiToolRegistry` — load and execute tools from any OpenAPI 3.x schema at runtime
- `FunctionTool` — wrap any Python callable with auto-generated JSON schema
- `build_search_tool` — ready-made tool dispatching to retrieval, Google, or SerpAPI

**Chat Processing**
- `process_message` — top-level orchestrator: resolves persona, tools, files, and LLM; dispatches to `run_llm_loop`; persists via `save_chat_turn`
- `run_llm_loop` — multi-turn loop: message history, tool dispatch, context injection, token streaming
- `run_llm_step` — single LLM step: prompt → stream → extract tool calls → `LlmStepResult`
- `DynamicCitationProcessor` — streams tokens and extracts citation markers in REMOVE / KEEP / HYPERLINK modes
- `compress_chat_history` — summarises older turns when context exceeds the token budget; branch-aware
- `Emitter` — routes packets (tokens, tool calls, citations) from worker threads to the HTTP stream
- `build_system_prompt` — assembles system prompt from persona, tools, knowledge, and memory context

**Cache & Persistence**
- `AgenticSearchStore` (SQLite) — connectors, documents, permissions, chat sessions, indexing attempts, usage reports, rate limits, SCIM tokens, standard answers (`src/backend/db/store.py`)
- Search history per user (`GET /search/search-history`) and query history with CSV export (`GET /admin/query-history/export`)
- `InMemoryCache` — in-flight chat session state (processing flag, stop signal, cancel) during streaming
- `ChunkBatchStore` — temp disk buffer decoupling embedding from index insertion for large jobs
- `InMemoryChatFile` — uploaded files (images, PDFs, text) held in memory for one chat turn

**Prompts**
- Chat prompt constants — citation reminders, system prompt defaults, file/image/tool templates (`src/backend/prompts/chat_prompts.py`)
- `KEYWORD_EXPANSION_PROMPT` / `QUERY_TYPE_PROMPT` — broaden sparse queries and classify intent for retrieval tuning
- Binary search/chat classification prompt with labelled examples and strict single-word output
- Agentic RAG prompts — decompose (2–4 sub-questions) and HyDE (hypothetical ideal answer) for `QueryEnhancer`
- `build_search_agent_instruction` — assembles the ReAct-style system prompt for `SearchAgentLoop`

**RL Training**
- Composite reward shaping (`SearchRewardFunction`) — format, search-use, answer-length, and exact-match components
- Group-relative advantage helpers for PPO, GRPO, and REINFORCE-style experiments
- PPO core: clipped policy loss, value loss, entropy, KL penalty, adaptive and fixed KL controllers
- Training data builders for search-QA and RAG parquet datasets (`src/training/data.py`)

**Query Classification**
- **Search vs chat** (`classify_is_search_flow`) — LLM-backed binary router; defaults to chat on ambiguous input (`src/backend/secondary_llm_flows/`)
- **Intent classifier** (`IntentPipeline`) — trainable feedforward ML model classifying `purchase` / `navigate` / `qa` / `recommendation`; selects fast / balanced / reasoning model tier (`src/model/intent_classifier.py`)

**Observability & Feature Flags**
- `build_admin_surface_summary` — single-call health snapshot: connectors, indexing, users, auth, models, tools, analytics, enterprise controls with a composite health score
- `MonitoringWorker` — background poller for process memory (RSS), index queue depth, connector count; ships JSON snapshots to a cloud data-plane URL
- `event_telemetry` / `identify_user` — PostHog event capture helpers; no-ops when PostHog is not configured
- Feature flags — composable chain: `EnvFeatureFlagProvider` → `PostHogFeatureFlagProvider`; `StaticFeatureFlagProvider` for tests; single call-site via `is_feature_enabled`


## Agentic RAG

```python
from src.agents.agentic_rag import AgenticRAGConfig, AgenticRAGLoop

loop = AgenticRAGLoop(
    AgenticRAGConfig(max_rounds=3, topk=5, retrieval_url="http://localhost:8000/retrieve"),
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

1. **Query enhancement** — decompose into sub-queries; generate HyDE hypothetical answer
2. **Hybrid+rerank retrieval** — retrieve per enhanced query; accumulate unique documents
3. **Sufficiency check** — LLM judges if context is enough; break or continue
4. **Follow-up generation** — LLM proposes targeted follow-up queries if insufficient
5. **Grounded synthesis** — answer from all accumulated evidence with inline citations


## Retrieval Setup

**Retrieval servers** (`src/backend/servers/retrieval/`):

| Module | Description |
|--------|-------------|
| `demo.py` | TF-IDF over corpus.jsonl — no Java required |
| `retrieval.py` | BM25 or dense (E5/BGE via FAISS) |
| `retrieval_rerank.py` | Retrieval + cross-encoder reranker |
| `hybrid_rerank.py` | Dense + BM25 RRF fusion + rerank (recommended for `AgenticRAGLoop`) |
| `google.py` | Google Custom Search proxy |
| `serp.py` | SerpAPI proxy |
| `browser.py` | playwright-cli browser automation; no API key, ~5–10s/query |

**Start a retrieval server:**

```bash
# Dense (E5)
python3 -m src.backend.servers.retrieval.retrieval \
  --model_path intfloat/e5-base-v2 --index_path indexes/e5_Flat.index \
  --corpus_path data/corpus.jsonl --retrieval_method e5 --device cpu --topk 5

# Sparse BM25
python3 -m src.backend.servers.retrieval.retrieval \
  --index_path indexes/bm25 --corpus_path data/corpus.jsonl --retrieval_method bm25
```

**Build indexes:**

```bash
python3 -m src.retrieval.index_builder \
  --retrieval_method e5 --model_path intfloat/e5-base-v2 \
  --corpus_path data/corpus.jsonl --faiss_type Flat --save_dir indexes/

python3 -m src.retrieval.index_builder \
  --retrieval_method bm25 --corpus_path data/corpus.jsonl --save_dir indexes/
```

**Hybrid + rerank:**

```bash
python3 -m src.backend.servers.retrieval.hybrid_rerank \
  --dense_model intfloat/e5-base-v2 --index_path indexes/e5_Flat.index \
  --corpus_path data/corpus.jsonl \
  --sparse_index_path indexes/bm25 --hybrid_alpha 0.5 \
  --retrieval_topk 10 --rerank_topk 5
```

**Web search servers:**

```bash
python3 -m src.backend.servers.retrieval.serp \
  --search_url "https://serpapi.com/search" --topk 3 --serp_api_key "$SERP_API_KEY"

python3 -m src.backend.servers.retrieval.google \
  --api_key "$GOOGLE_API_KEY" --topk 5 --cse_id "$GOOGLE_CSE_ID" --snippet_only
```

**Health check:**

```bash
curl -i -sS http://127.0.0.1:8000/health
curl -i -sS -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" -d '{"query":"What is FAISS?","top_k":5}'
```


## Training

The training pipeline is modular: generate trajectories → score with rewards → compute advantages → optimize.

| Task | Entry point |
|------|-------------|
| Deterministic trace | `python3 -m examples.run_search_trace_workflow` |
| SFT record from trace | `python3 -m examples.run_search_trace_workflow --sft` |
| QA parquet preparation | `python3 -m examples.prepare_search_qa_dataset` |
| Reward/GRPO smoke test | `python3 -m examples.run_grpo_training_pipeline` |
| Reward function | `src/training/reward.py` |
| GRPO helpers | `src/training/grpo.py` |
| PPO helpers | `src/training/ppo/` |
| Generation and policy loss | `src/model/generation.py` |

**Reward components** (`SearchRewardFunction`):

| Component | What it measures |
|-----------|-----------------|
| `format` | Well-formed XML trace with required action tags |
| `search_use` | Agent issued at least one search action |
| `answer_length` | Answer within acceptable token bounds |
| `exact_match` | Token-overlap correctness against reference answers |

```python
from src.training.reward import SearchRewardFunction, SearchRewardConfig

reward_fn = SearchRewardFunction(SearchRewardConfig(
    format_weight=0.2, search_use_weight=0.3,
    length_weight=0.1, exact_match_weight=0.4,
))
```

**GRPO** — group-relative advantages from G rollouts per prompt:

```python
from src.training.grpo import score_prompt_group, compute_grpo_outcome_advantage

scored = score_prompt_group(rollouts, reward_fn, reference_answer)
advantages = compute_grpo_outcome_advantage(scored)
```

**PPO** — clipped policy + value loss with KL penalty:

```python
from src.training.ppo import compute_ppo_policy_loss_core, compute_value_loss, AdaptiveKLController

policy_loss = compute_ppo_policy_loss_core(logprobs, old_logprobs, advantages, clip_eps=0.2)
value_loss  = compute_value_loss(values, returns, old_values, clip_eps=0.2)
```

**XML search protocol** — the ReAct-style trace format used by `SearchAgentLoop`:

```xml
<think>decide whether to answer or search</think>
<search>precise query</search>
<information>retrieval results injected by the environment</information>
<answer>final grounded answer</answer>
```

`<information>` is environment output — mask it from policy/SFT action loss.


## API Health Checks

Web backend: `http://localhost:7860` · Retrieval server: `http://localhost:8000`

**Generate a dev JWT** (required for admin endpoints):

```bash
export TOKEN=$(python3 -c "
from src.backend.auth import generate_user_jwt_token
print(generate_user_jwt_token(user_id='dev', email='dev@local'))
")
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
  -d '{"query": "What is FAISS?", "mode": "search"}'

curl -s http://localhost:7860/api/sessions -H "Authorization: Bearer $TOKEN"

curl -s -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" -d '{"query": "dense retrieval", "top_k": 3}'
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

- Dense retrieval defaults to CPU; set `--device cuda` only on a dedicated retrieval node.
- BM25 serving requires Java because Pyserini uses Lucene.
- Empty or invalid queries return empty result lists.
- Some web pages block scraping or return little usable text.
- Google Custom Search and SerpAPI are subject to their own quota and billing rules.
- If `prepare_search_qa_dataset` fails with a `pyarrow` extension error, run `pip install -r requirements.txt`.
