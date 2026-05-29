# Agentic Search

Agentic Search is a retrieval-backed agent and search-policy platform. It
combines a full-featured FastAPI server layer (admin APIs, SCIM provisioning,
billing proxy, OAuth connectors, query history, usage reporting) with local
dense/sparse retrieval, multi-turn agent traces, SFT data builders, and
PPO/GRPO reward helpers.

## What Is Here

| Area | Main modules |
|------|--------------|
| **FastAPI server layer** | `src/servers/` |
| **Admin APIs** | `analytics`, `billing`, `evals`, `license`, `manage`, `query_history`, `reporting`, `settings`, `token_rate_limits` |
| **Auth & access** | `src/servers/_auth.py`, `src/servers/middleware/` |
| **Connectors API** | `src/servers/documents/`, `src/servers/oauth/` |
| **Identity provisioning** | `src/servers/scim/` |
| **User & group management** | `src/servers/user_group/`, `src/servers/tenants/` |
| **Search & chat** | `src/servers/query_and_chat/`, `src/servers/web/` |
| **Retrieval servers** | `src/servers/retrieval/`, `src/servers/web_search/` |
| **Indexing pipeline** | `src/servers/indexing/` |
| Agent loops | `src/agents/` |
| Retrieval and search engines | `src/retrieval/` |
| Tool schemas and search tools | `src/tools/` |
| Model generation and intent routing | `src/model/` |
| SFT, rewards, PPO, and GRPO helpers | `src/training/` |
| Runnable examples | `examples/` |

The FastAPI application is assembled in `src/servers/web/app.py`. Every feature
area is a self-contained router factory registered via `_register_routers()`.
The SQLite-backed `AgenticSearchStore` (`src/db/`) is the single persistence
layer — no Postgres, Redis, or Celery required for local deployments.

## Features

- Local dense retrieval with FAISS-compatible indexes (E5, BGE, and custom embedders).
- Local sparse retrieval with BM25/Pyserini.
- Optional web search through Google Custom Search and SerpAPI.
- Retrieval + cross-encoder reranking pipeline.
- Multi-turn `SearchAgentLoop` traces with `<think>`, `<search>`,
  `<information>`, `<fetch>`, and `<answer>` actions.
- `SingleTurnAgentLoop` for parse-and-dispatch single-action flows.
- One-shot generation, one-shot RAG, full search-agent, and generic tool-agent
  loops.
- Hermes, Llama-3, and JSON tool-call parsers.
- OpenAPI-based `ApiToolRegistry` for dynamic tool loading.
- Composite reward shaping (`SearchRewardFunction`) with format, search-use,
  answer-length, and exact-match components.
- Group-relative advantage helpers for PPO, GRPO, and REINFORCE-style
  experiments.
- PPO core algorithms: clipped policy loss, value loss, entropy, KL penalty,
  adaptive and fixed KL controllers.
- Intent classifier (`IntentPipeline`) for query routing and model routing.
- Intent-driven model routing: pick fast/balanced/reasoning model by intent.
- Training data builders for search-QA and RAG parquet datasets.
- SFT example builder from search traces.

## Install

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

For BM25, Java must be available. On Apple Silicon, FAISS is usually more stable
when installed through conda:

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

## Server Architecture

The full web application is a single FastAPI instance created by
`src.servers.web.app.create_web_app()`. Router factories are grouped by feature
area and registered in `_register_routers()`.

```python
from src.servers.web.app import create_web_app
from src.db import AgenticSearchStore

store = AgenticSearchStore("agentic-search.sqlite3")
app   = create_web_app(store=store)
```

All state lives in `AgenticSearchStore` — a SQLite-backed repository for
connectors, documents, users, groups, chat sessions, SCIM tokens, usage reports,
rate-limit rules, and more. No external database is required.

### Middleware Stack

| Middleware | Module | Purpose |
|-----------|--------|---------|
| Tenant tracking | `src/servers/middleware/tenant_tracking.py` | Sets tenant context from request headers |
| License enforcement | `src/servers/middleware/license_enforcement.py` | Returns 402 for gated paths when license is invalid |
| Tier gate | `src/servers/middleware/tier_gate.py` | Returns 402 for paths that require a higher plan tier |

### Admin Auth

Every router factory that needs admin-only endpoints calls
`make_require_admin(app_settings)` from `src/servers/_auth.py`. It returns a
FastAPI dependency that checks `Authorization: Bearer <jwt>` against
`AppSettings.auth.super_users`.

```python
from src.servers._auth import make_require_admin
_require_admin = make_require_admin(app_settings)

@router.get("/admin/my-endpoint")
def my_endpoint(_: AuthenticatedUser = Depends(_require_admin)):
    ...
```

### API Routers

#### Analytics — `src/servers/analytics/`

`create_analytics_router(store, app_settings)` — admin-only daily aggregates.

| Endpoint | Description |
|----------|-------------|
| `GET /analytics/query` | Daily session + message counts for a date window |
| `GET /analytics/user` | Daily distinct active user counts |

#### Billing — `src/servers/billing/`

`create_billing_router(app_settings)` — Stripe proxy with in-memory circuit breaker.

| Endpoint | Description |
|----------|-------------|
| `POST /admin/billing/create-checkout-session` | Start a Stripe checkout session |
| `POST /admin/billing/create-customer-portal-session` | Open the Stripe billing portal |
| `GET  /admin/billing/billing-information` | Current subscription status |
| `POST /admin/billing/seats/update` | Change seat count |
| `POST /admin/billing/end-trial` | End trial early (cloud only) |
| `GET  /admin/billing/stripe-publishable-key` | Cached Stripe publishable key |
| `POST /admin/billing/reset-connection` | Clear circuit breaker |

Configure with `AGENTIC_SEARCH_CLOUD_DATA_PLANE_URL`, `STRIPE_PUBLISHABLE_KEY_OVERRIDE`, and `WEB_DOMAIN`.

#### Documents — `src/servers/documents/`

`create_documents_router(store, app_settings)` — connector-credential pair management.

#### Enterprise Settings — `src/servers/enterprise_settings/`

`create_enterprise_settings_routers(app_settings)` — branding, logo, analytics script.

| Endpoints | Description |
|-----------|-------------|
| `GET/PUT /enterprise-settings` | Branding / UI configuration |
| `PUT /admin/enterprise-settings/logo` | Upload custom logo |
| `GET/PUT /admin/enterprise-settings/custom-analytics-script` | Custom analytics JS |
| `GET/POST /admin/enterprise-settings/scim/token` | SCIM bearer-token management |

#### Evals — `src/servers/evals/`

`create_evals_router(app_settings, search_url)` — synchronous search quality evaluation.

| Endpoint | Description |
|----------|-------------|
| `POST /evals/eval_run` | Run a search eval synchronously, return structured results |
| `POST /evals/eval_run_ack` | Fire-and-forget background eval |

#### Features / Hooks — `src/servers/features/hooks/`

`create_hooks_router(store, app_settings)` — admin CRUD for outbound webhooks with SSRF protection and live reachability checks.

| Endpoint | Description |
|----------|-------------|
| `GET  /admin/hooks/specs` | List all available hook-point specs |
| `GET  /admin/hooks` | List configured hooks |
| `POST /admin/hooks` | Create hook (validates endpoint reachability) |
| `GET  /admin/hooks/{id}` | Get hook |
| `PATCH /admin/hooks/{id}` | Update hook |
| `DELETE /admin/hooks/{id}` | Delete hook |
| `POST /admin/hooks/{id}/activate` | Re-activate and re-validate |
| `POST /admin/hooks/{id}/deactivate` | Deactivate |
| `POST /admin/hooks/{id}/validate` | Test reachability |
| `GET  /admin/hooks/{id}/logs` | Execution logs |

Hooks fire at `HookPoint.QUERY_PROCESSING` and `HookPoint.ANSWER_GENERATED`.

#### License — `src/servers/license/`

`create_license_router(app_settings)` — file-backed RSA-signed license management.

| Endpoint | Description |
|----------|-------------|
| `GET    /license` | Current license status and expiry stage |
| `GET    /license/seats` | Seat usage from stored license |
| `POST   /license/upload` | Upload a `.lic` file (air-gapped) |
| `POST   /license/claim` | Claim license from cloud data plane |
| `POST   /license/refresh` | Re-verify stored license |
| `DELETE /license` | Delete stored license |

License files are stored at `$AGENTIC_SEARCH_DATA_DIR/license.dat`.

#### Manage (Standard Answers) — `src/servers/manage/`

`create_manage_router(store, app_settings)` — keyword → answer mappings with categories.

| Endpoint | Description |
|----------|-------------|
| `GET/POST /manage/admin/standard-answer` | List / create standard answers |
| `PATCH    /manage/admin/standard-answer/{id}` | Update a standard answer |
| `DELETE   /manage/admin/standard-answer/{id}` | Delete a standard answer |
| `GET/POST /manage/admin/standard-answer/category` | List / create categories |
| `PATCH    /manage/admin/standard-answer/category/{id}` | Update a category |

#### OAuth — `src/servers/oauth/`

`create_oauth_router(app_settings)` — OAuth 2.0 URL generation for connector authorization.

| Endpoint | Description |
|----------|-------------|
| `POST /oauth/prepare-authorization-request` | Generate OAuth URL for Slack, Confluence, or Google Drive |
| `POST /oauth/connector/{connector}/callback` | OAuth callback stub (501 — credential DB not yet implemented) |

Session state is stored in-memory (10-minute TTL). Configure connector credentials via `OAUTH_SLACK_CLIENT_ID`, `OAUTH_CONFLUENCE_CLOUD_CLIENT_ID`, `OAUTH_GOOGLE_DRIVE_CLIENT_ID`.

#### Query and Chat — `src/servers/query_and_chat/`

`create_search_router(store, search_url)` and `basic_router` — search and chat APIs.

#### Query History — `src/servers/query_history/`

`create_query_history_router(store, app_settings)` — admin access to chat session history.

| Endpoint | Description |
|----------|-------------|
| `GET /admin/chat-sessions` | Sessions for a specific user |
| `GET /admin/chat-session-history` | Paginated history with time / feedback filters |
| `GET /admin/chat-session-history/{id}` | Full message list for one session |
| `GET /admin/query-history/export` | Stream CSV of all Q&A pairs (synchronous) |

#### Reporting (Usage Export) — `src/servers/reporting/`

`create_reporting_router(store, app_settings)` — ZIP reports with chat messages and users CSVs.

| Endpoint | Description |
|----------|-------------|
| `POST /admin/usage-report` | Generate and store a ZIP usage report |
| `GET  /admin/usage-report` | List all stored reports |
| `GET  /admin/usage-report/{name}` | Stream the ZIP for a report |

Reports are stored as BLOBs in the SQLite store.

#### SCIM 2.0 — `src/servers/scim/`

`create_scim_router(store)` — RFC 7644 user and group provisioning for Okta, Entra ID, and other IdPs.

| Endpoint | Description |
|----------|-------------|
| `GET  /scim/v2/ServiceProviderConfig` | SCIM capability advertisement (no auth) |
| `GET  /scim/v2/ResourceTypes` | Resource type list (no auth) |
| `GET  /scim/v2/Schemas` | Schema definitions (no auth) |
| `GET/POST/PUT/PATCH/DELETE /scim/v2/Users` | User CRUD with SCIM filtering |
| `GET/POST/PUT/PATCH/DELETE /scim/v2/Groups` | Group CRUD |
| `POST /scim/v2/tokens` | Create a SCIM bearer token |
| `GET  /scim/v2/tokens` | List tokens |
| `DELETE /scim/v2/tokens/{id}` | Revoke a token |

Tokens are SHA-256 hashed and stored in SQLite. The `ScimDAL` provides all SCIM
queries through `AgenticSearchStore`. Providers `OktaProvider` and `EntraProvider`
handle IdP-specific PATCH quirks.

#### Settings — `src/servers/settings/`

`create_settings_router(app_settings)` — license-aware application status for the UI.

| Endpoint | Description |
|----------|-------------|
| `GET /settings` | Returns `{ee_features_enabled, tier, application_status, license_enforcement_enabled}` |

#### Token Rate Limits — `src/servers/token_rate_limits/`

`create_token_rate_limits_router(store, app_settings)` — configurable token budget rules per user or group.

| Endpoint | Description |
|----------|-------------|
| `GET  /admin/token-rate-limits/users` | Global user-scoped limits |
| `POST /admin/token-rate-limits/users` | Create a global limit |
| `GET  /admin/token-rate-limits/user-groups` | All group-scoped limits keyed by group name |
| `GET  /admin/token-rate-limits/user-group/{id}` | Limits for one group |
| `POST /admin/token-rate-limits/user-group/{id}` | Create a group limit |

#### Tenants — `src/servers/tenants/`

Multi-tenant scaffolding: provisioning, billing, schema management, team membership, and user invitation APIs.

#### User Groups — `src/servers/user_group/`

`create_user_group_router(store, app_settings)` — group CRUD with document-permission integration.

### Configuration

All settings are loaded from environment variables via `src/configs/AppSettings`:

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
| `STRIPE_PUBLISHABLE_KEY_OVERRIDE` | — | Override for local Stripe testing |
| `DEV_MODE` | `false` | Use `redirectmeto.com` for OAuth callbacks |
| `OAUTH_SLACK_CLIENT_ID` | — | Slack OAuth app client ID |
| `OAUTH_CONFLUENCE_CLOUD_CLIENT_ID` | — | Confluence OAuth app client ID |
| `OAUTH_GOOGLE_DRIVE_CLIENT_ID` | — | Google Drive OAuth app client ID |

### Database Store

`AgenticSearchStore` (`src/db/`) is a single SQLite repository. All tables are
created at init time — no migrations needed:

| Table | Purpose |
|-------|---------|
| `users` | User identity records |
| `groups` / `group_members` | Groups and membership |
| `connector_configs` | Connector configurations |
| `documents` / `document_permissions` | Document content and ACLs |
| `chat_sessions` / `chat_messages` | Conversation state |
| `hooks` | Webhook configurations |
| `index_attempts` | Indexing job records |
| `usage_reports` | ZIP usage report BLOBs |
| `token_rate_limits` | Rate-limit rules |
| `standard_answers` / `standard_answer_categories` | Keyword → answer mappings |
| `scim_tokens` / `scim_user_mappings` / `scim_group_mappings` | SCIM provisioning state |
| `user_is_active` | SCIM-managed user active flags |

## Web Search Provider Admin API

The repo also includes a lightweight FastAPI admin surface for configuring web
search providers without adding a database dependency:

```python
from src.servers.web_search.api import create_web_search_router

router = create_web_search_router()
```

It exposes `/admin/web-search/search-providers` and
`/admin/web-search/content-providers` routes for listing, upserting, activating,
deactivating, deleting, and validation-testing provider settings. Validation
tests are local by default; pass `"live": true` to make a real provider request.

## Quick Start

Run a deterministic search-agent trace with fake model/search backends:

```bash
python3 -m examples.run_search_trace_workflow
```

Build the matching full-trace SFT example:

```bash
python3 -m examples.run_search_trace_workflow --sft
```

Run the reward and GRPO helper smoke test:

```bash
python3 -m examples.run_grpo_training_pipeline
```

Prepare NQ/FlashRAG-style question-answer pairs for search-agent training:

```bash
python3 -m examples.prepare_search_qa_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets \
  --dataset_config nq \
  --local_dir data/nq_search
```

Inspect converted question/answer pairs before writing parquet:

```bash
python3 -m examples.prepare_search_qa_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets \
  --dataset_config nq \
  --splits test \
  --max_examples 20 \
  --preview \
  --preview_rows 5
```

If this command fails with a `pyarrow` extension error, refresh dependencies
with `python -m pip install -r requirements.txt`; this repo pins a
`datasets`-compatible `pyarrow` range.

Create a tiny local parquet slice for a dry run:

```bash
python3 -m examples.prepare_search_qa_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets \
  --dataset_config nq \
  --splits test \
  --max_examples 100 \
  --local_dir data/nq_search_debug
```

Prepare RAG-style NQ records from cached retrieval results:

```bash
python3 -m examples.prepare_search_rag_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets \
  --dataset_config nq \
  --corpus_path data/wiki-18.jsonl \
  --train_retrieval_cache data/nq_train_retrieval_cache.json \
  --test_retrieval_cache data/nq_test_retrieval_cache.json \
  --topk 3 \
  --local_dir data/nq_rag
```

Preview the RAG prompt/context shape before writing parquet:

```bash
python3 -m examples.prepare_search_rag_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets \
  --dataset_config nq \
  --corpus_path data/wiki-18.jsonl \
  --train_retrieval_cache data/nq_train_retrieval_cache.json \
  --test_retrieval_cache data/nq_test_retrieval_cache.json \
  --splits test \
  --topk 3 \
  --max_examples 20 \
  --preview \
  --preview_rows 5
```

## Run An Agent

`examples/run_agentic_search.py` is the main CLI.

```bash
python3 -m examples.run_agentic_search \
  --mode single \
  --question "What is FAISS?" \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --local --device cpu \
  --max_tokens 64 --temperature 0
```

Modes:

| Mode | Loop | Use it for |
|------|------|------------|
| `single` | `PlainGenerationLoop` | Simple local generation smoke tests |
| `search` | `SearchAgentLoop` | Multi-turn search traces for RAG, SFT, and RL |
| `tool` | `ToolAgentLoop` | Generic function/tool-calling experiments |

For server-backed inference:

```bash
python3 -m examples.run_agentic_search \
  --mode search \
  --question "Compare dense and sparse retrieval" \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --vllm_url http://localhost:8080 \
  --search_url http://localhost:8000/retrieve
```

## Indexing Helpers

The repo-native indexing pipeline lives in `src/retrieval/index_builder.py`.
For a server-style facade, use `src.servers.indexing`:

```python
from src.connectors import Document
from src.servers.indexing import index_document_batch

result = index_document_batch(
    [Document(id="doc-1", title="Example", contents="hello world")],
    save_dir="indexes/example",
)
```

The facade includes `Chunker`, `DefaultIndexingEmbedder`, `ChunkBatchStore`,
`embed_and_stream`, document prefiltering, mini-chunk support, and vector-write
retry helpers.

## Metadata Store

`src.db` provides a lightweight SQLite store for local connector and retrieval
metadata:

```python
from src import AgenticSearchStore, ConnectorConfig, StoredDocument

with AgenticSearchStore("agentic-search.sqlite3") as store:
    connector = store.upsert_connector(
        ConnectorConfig(id="local", name="Local files", source="local_file")
    )
    store.upsert_document(
        StoredDocument(
            id="doc-1",
            title="Example",
            contents="hello world",
            connector_id=connector.id,
        )
    )
```

The store tracks connector configs, documents, users, groups, document
permissions, chat/session state, indexing attempts, and related JSON metadata.

## Search Context

`src.context` contains small, repo-native helpers for retrieval-grounded chat:

```python
from src import SearchResult, build_context_bundle, build_answer_prompt

context = build_context_bundle(
    "What is FAISS?",
    [SearchResult(title="FAISS", contents='"FAISS"\nA vector search library.')],
)
prompt = build_answer_prompt("What is FAISS?", context)
```

It includes normalized context documents, citation extraction, retrieval prompt
builders, answer prompt builders, LLM protocol types, and an
`answer_with_retrieval` pipeline for connecting a `/retrieve` service to answer
generation.

## Web Search Experience

Serve the lightweight browser UI for search and agent answers.
Three processes must be running at the same time:

**Terminal 1 — retrieval server:**

For a quick local demo (no index or Java required):

```bash
python3 -m src.servers.retrieval.demo --corpus_path data/corpus.jsonl
```

For production BM25 (requires `pyserini` and Java), see [Build Indexes](#build-indexes) then:

```bash
python3 -m src.servers.retrieval.retrieval \
  --retrieval_method bm25 \
  --index_path indexes/bm25 \
  --corpus_path data/corpus.jsonl
```

**Terminal 2 — web backend** (run from the repo root):

```bash
pip install -e .   # one-time, installs src as a package
uvicorn src.servers.web.app:app --host 127.0.0.1 --port 7860
```

**Terminal 3 — frontend**:

```bash
cd web
npm install
npm run dev
```

Open `http://127.0.0.1:5173` and ask a question. The app shows the generated answer,
citations, source cards, and session history. The JSON API is available at
`POST /api/agent` and persists chat state with `AgenticSearchStore`.

During development Vite proxies `/api/*` to the FastAPI server on port `7860`.
For production, run `npm run build`; `src.servers.web.app` serves `web/dist`
when that bundle exists and falls back to its built-in HTML otherwise.

## Local Retrieval

Start a dense retrieval server:

```bash
python3 -m src.servers.retrieval.retrieval \
  --model_path intfloat/e5-base-v2 \
  --index_path indexes/e5_Flat.index \
  --corpus_path data/corpus.jsonl \
  --retrieval_method e5 \
  --device cpu \
  --workers 1 \
  --topk 5
```

Start a sparse BM25 server:

```bash
python3 -m src.servers.retrieval.retrieval \
  --index_path indexes/bm25 \
  --corpus_path data/corpus.jsonl \
  --retrieval_method bm25 \
  --workers 1
```

Check the server:

```bash
curl -i -sS http://127.0.0.1:8000/health

curl -i -sS -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"What is FAISS?","top_k":5}'
```

The retrieval endpoint accepts both single-query and batch-query request shapes.

## Build Indexes

Dense FAISS index:

```bash
python3 -m src.retrieval.index_builder \
  --retrieval_method e5 \
  --model_path intfloat/e5-base-v2 \
  --corpus_path data/corpus.jsonl \
  --faiss_type Flat \
  --save_dir indexes/
```

BM25 index:

```bash
python3 -m src.retrieval.index_builder \
  --retrieval_method bm25 \
  --corpus_path data/corpus.jsonl \
  --save_dir indexes/
```

## Web Search

`src.tools.search` routes calls to `retrieval`, `google`, or `serpapi`. Missing API keys return structured tool errors.

Standalone web-search servers are available under `src.servers.retrieval`:

```bash
python3 -m src.servers.retrieval.serp \
  --search_url "https://serpapi.com/search" \
  --topk 3 \
  --serp_api_key "$SERP_API_KEY"

python3 -m src.servers.retrieval.google \
  --api_key "$GOOGLE_API_KEY" \
  --topk 5 \
  --cse_id "$GOOGLE_CSE_ID" \
  --snippet_only
```

## Retrieval Plus Rerank

```bash
python3 -m src.servers.retrieval.retrieval_rerank \
  --retriever_model intfloat/e5-base-v2 \
  --index_path indexes/e5_Flat.index \
  --corpus_path data/corpus.jsonl \
  --retrieval_method e5 \
  --retrieval_topk 10 \
  --rerank_topk 3
```

For BM25 plus reranking, use `--retrieval_method bm25` and omit
`--retriever_model`.

## Training Flow

The training pipeline is intentionally modular:

1. Generate trajectories with `SearchAgentLoop` or `LLMGenerationManager`.
2. Score trajectories with `SearchRewardFunction`.
3. Compute group-relative advantages with `src.training.grpo`.
4. Save JSONL batches or compute token log probabilities.
5. Optimize with PPO/GRPO/REINFORCE helpers in `src.model.generation` and
   `src.training.ppo`.

Useful entry points:

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

## XML Search Protocol

The search-agent trace uses a compact ReAct-style protocol:

```xml
<think>decide whether to answer or search</think>
<search>precise query</search>
<information>retrieval results injected by the environment</information>
<answer>final grounded answer</answer>
```

`<information>` is environment output and should be masked out of policy/SFT
action loss.

## Model Routing

`--model_routing intent` uses the intent classifier before model loading:

```bash
python3 -m examples.run_agentic_search \
  --mode search \
  --question "Recommend a dense retrieval setup for a small budget" \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --model_routing intent \
  --intent_model models/intent_classifier.pt \
  --fast_model Qwen/Qwen2.5-0.5B-Instruct \
  --balanced_model Qwen/Qwen2.5-1.5B-Instruct \
  --reasoning_model Qwen/Qwen2.5-7B-Instruct \
  --local --device cpu
```

Default routes:

| Intent | Route |
|--------|-------|
| `qa`, `navigate` | `fast_model` |
| `recommendation` | `balanced_model` |
| `purchase` | `reasoning_model` |

If confidence is too low or a route model is missing, the CLI falls back to
`--model`.

## Module Reference

### Agent Loops (`src/agents/`)

| Class / function | Description |
|-----------------|-------------|
| `AgentLoopBase` | Abstract base all loops inherit from; exposes `run()` |
| `AgentLoopConfig` | Shared config dataclass (max tokens, temperature, etc.) |
| `AgentLoopOutput` | Return value of `run()`: steps, final answer, metrics |
| `RolloutStep` | Single turn record: role, text, token ids, action mask |
| `PlainGenerationLoop` | One-shot generation, no search |
| `SearchAgentLoop` | Multi-turn XML trace loop (`<think>/<search>/<answer>`) |
| `SearchAgentLoopConfig` | Adds topk, max search limit, search URL, fetch URL |
| `SingleTurnAgentLoop` | Parse first action from generation, dispatch tool call |
| `ToolAgentLoop` | Generic tool-calling loop with `ToolParser` |
| `AgentState` | Full state: `TaskNode` graph, `Plan`, `RouteDecision`, metrics |
| `register` / `get_registered_agent_loop` | Decorator-based loop registry |
| `build_search_agent_instruction` | Build the system-prompt instruction string |

### Retrieval (`src/retrieval/`)

| Class / function | Description |
|-----------------|-------------|
| `DenseRetriever` | FAISS-backed dense retrieval; supports E5, BGE, custom encoders |
| `DenseRetrieverConfig` | Model path, index path, device, topk, batch size |
| `SparseRetriever` | Pyserini BM25 retriever |
| `SparseRetrieverConfig` | Index path, topk, language |
| `SentenceTransformerReranker` | Cross-encoder reranker via `sentence-transformers` |
| `RerankerConfig` | Model name, device, batch size |
| `get_reranker` | Factory: returns a configured `SentenceTransformerReranker` |
| `SearchClient` | Async HTTP client for the retrieval server (`/retrieve`) |
| `SearchClientConfig` | Base URL, timeout, retry settings |
| `SearchResult` | Single retrieved document: id, title, contents, score |
| `SearchContext` | Ordered list of `SearchResult`s for one query |
| `AgentContext` | Per-turn context accumulator across search rounds |
| `Vocabulary` | Freq-filtered token vocabulary with `build` / `encode` |
| `TextProcessor` | Tokenization, stopword filtering, field extraction |
| `normalize_text` / `tokenize_text` | Fast text normalisation and word tokenisation |
| `normalize_document` / `tokenize_document` | Document-level normalisation |
| `extract_keywords` | TF-style keyword extraction from document fields |
| `build_vocabulary_from_sequences` | Build a `Vocabulary` from a token sequence list |

#### Retrieval Servers (`src/retrieval/servers/`)

| Module | Server |
|--------|--------|
| `retrieval` | Dense (E5/BGE) or sparse (BM25) retrieval; `/retrieve`, `/health` |
| `retrieval_rerank` | Retrieval + cross-encoder rerank in one server |
| `rerank` | Standalone rerank endpoint |
| `google` | Google Custom Search proxy server |
| `serp` | SerpAPI proxy server |

### Tools (`src/tools/`)

| Class / function | Description |
|-----------------|-------------|
| `Tool` / `FunctionTool` | Abstract tool base; `FunctionTool` wraps a plain callable |
| `ToolSchema` | JSON-schema definition attached to a tool |
| `SearchPage` | Search result page: url, title, snippet, contents |
| `build_search_tool` | Build a `FunctionTool` that calls retrieval, Google, or SerpAPI |
| `format_search_pages` | Render a list of `SearchPage`s to text for the model |
| `HermesToolParser` | Parse Hermes-format `<tool_call>` XML |
| `Llama3ToolParser` | Parse Llama-3 `<\|python_tag\|>` tool calls |
| `JSONToolParser` | Parse bare JSON tool call blobs |
| `FunctionCall` | Parsed tool call: name + arguments dict |
| `ApiToolRegistry` | Load tools from an OpenAPI 3.x schema string |
| `ApiRequestTool` | Auto-generated tool that executes one OpenAPI operation |
| `parse_openapi_schema` | Parse an OpenAPI 3.x YAML/JSON string into `OpenAPISchema` |

### Connectors (`src/connectors/`)

| Class / function | Description |
|-----------------|-------------|
| `Document` / `SlimDocument` | Native document containers emitted by connectors |
| `BaseConnector` / `LoadConnector` | Connector interfaces for shared behavior and full-state document loading |
| `PollConnector` | Incremental sync by time window |
| `CheckpointedConnector` | Incremental sync that returns a persisted checkpoint |
| `CheckpointedConnectorWithPermSync` | Checkpointed sync with document permission metadata |
| `SlimConnector` | Pull only document ids for pruning or expired document deletion |
| `SlimConnectorWithPermSync` | Pull document ids plus permission metadata |
| `OAuthConnector` | OAuth authorization-code connector contract |
| `InMemoryConnector` | Load documents from Python objects or dictionaries |
| `LocalFileConnector` | Load UTF-8 text files from paths, directories, or glob patterns |
| `SearchConnector` | Load search results as documents through retrieval, Google, or SerpAPI |
| `StaticCredentialsProvider` | Simple in-memory provider for connector credentials |

### Model (`src/model/`)

#### Generation (`src/model/generation.py`)

| Class / function | Description |
|-----------------|-------------|
| `LLMGenerationManager` | Orchestrates batched GRPO rollouts: sample → score → pack |
| `GenerationConfig` | vLLM sampling params + retrieval URLs + safety config |
| `EndpointRetriever` | HTTP retrieval via `/retrieve` endpoint |
| `SimulateRetriever` | Deterministic fake retriever for tests |
| `GoogleRetriever` | Google Custom Search retriever |
| `EndpointFetcher` | Fetch a URL via the retrieval server's fetch endpoint |
| `ask_llm` | Low-level single-prompt vLLM call |
| `search_simulate` | Run a full search-agent trace with any retriever |
| `score_group_rollout` | Score a group of rollouts with a reward function |
| `assign_group_relative_advantages` | Compute GRPO advantages for a scored group |
| `apply_rollout_safety_penalties` | Penalise length, repetition, and format violations |
| `apply_safety_penalties_to_scored_rollouts` | Batch version of the above |
| `trajectory_log_prob_pack` | Pack token logprobs into training tensors |
| `format_search_trajectory_log` | Render a `SearchTrajectoryLog` to human-readable text |
| `format_trajectory_batch` | Render a batch of trajectories |
| `save_training_batch_jsonl` | Write a scored rollout batch to JSONL |
| `RolloutTrajectory` | Full trajectory: prompt, steps, final answer, reward |
| `SearchTrajectoryLog` | Per-query search turn log with documents and scores |
| `ActorRolloutStep` | One generation step: tokens, logprobs, action mask |
| `ReActStep` | ReAct observation-action pair |
| `GroupedRolloutBatch` | G rollouts for one prompt (GRPO group) |
| `ScoredGroupedRollout` | `GroupedRolloutBatch` with rewards and advantages |
| `GRPORolloutSafetyConfig` | Thresholds for safety penalties |

#### Intent Classifier (`src/model/intent_classifier.py`)

| Class / function | Description |
|-----------------|-------------|
| `IntentPipeline` | Trainable feedforward classifier; `train`, `predict`, `save`, `load` |
| `IntentPrediction` | `(intent, confidence)` result dataclass |
| `INTENT_LABELS` | `["purchase", "navigate", "qa", "recommendation"]` |
| `resolve_search_settings` | Map a prediction to adjusted topk / evidence / internal-knowledge flags |
| `load_training_data` | Load a JSON examples file into `(token_list, label)` pairs |

#### Intent Training (`src/model/intent_training.py`)

| Function | Description |
|----------|-------------|
| `train_intent_classifier` | Train an `IntentPipeline` from an examples file and save it |
| `generate_intent_examples` | Generate intent-labelled examples from a JSONL corpus |
| `write_intent_examples` | Write examples list to a pretty JSON file |
| `load_corpus` | Load a JSONL corpus into a list of document dicts |
| `load_vocabulary_tokens` | Load top-N tokens from a vocabulary metadata file |

#### Tensor Helper (`src/model/tensor_helper.py`)

| Class | Description |
|-------|-------------|
| `TensorHelper` | Pad, pack, and mask token sequences for PPO/GRPO training |
| `TensorConfig` | Padding token id, max sequence length, device |

### Training (`src/training/`)

#### Reward (`src/training/reward.py`)

| Class / function | Description |
|-----------------|-------------|
| `SearchRewardFunction` | Composite reward: format + search-use + length + exact-match |
| `SearchRewardConfig` | Weights for each reward component |
| `simple_sparse_correctness_reward` | Fast token-overlap correctness reward |
| `normalize_answer_text` | Lowercase, strip articles and punctuation |

#### Evaluation (`src/training/evaluation.py`)

| Class | Description |
|-------|-------------|
| `SearchResultEvaluator` | Evaluate retrieval quality across a batch of queries |
| `SearchEvaluationConfig` | Relevance threshold, topk, exact-match mode |
| `QueryEvaluation` | Per-query evaluation: precision, recall, hit |
| `SearchRoundEvaluation` | Per-search-round aggregate metrics |

#### GRPO (`src/training/grpo.py`)

| Class / function | Description |
|-----------------|-------------|
| `score_prompt_group` | Score G rollouts for one prompt, return `ScoredGRPORollout` |
| `score_prompt_batch` | Score a full batch of prompt groups |
| `compute_grpo_outcome_advantage` | Normalise rewards to group-relative advantages |
| `build_grpo_sampling_params` | Build vLLM sampling params for G samples per prompt |
| `GRPORolloutSample` | One rollout sample: tokens, logprobs, reward |
| `ScoredGRPORollout` | Rollout with advantage assigned |
| `PromptGroupSamplingConfig` | Group size, temperature, top-p |

#### SFT (`src/training/sft.py`)

| Class / function | Description |
|-----------------|-------------|
| `SFTExample` | Prompt + completion pair with action mask |
| `build_search_sft_example` | Build an `SFTExample` from a search-agent trace |

#### Data (`src/training/data.py`)

| Class / function | Description |
|-----------------|-------------|
| `PromptOnlyDataset` | PyTorch `Dataset` over tokenised prompt records |
| `PromptBatch` | Collated batch of padded prompt tensors |
| `build_prompt_dataloader` | Build a `DataLoader` from a parquet dataset |
| `build_search_qa_record` | Build a search-QA prompt/answer training record |
| `build_search_rag_record` | Build a RAG prompt/answer training record |
| `build_search_qa_prompt` | Format a question into a search-agent prompt string |
| `build_search_qa_messages` | Format as an OpenAI-style messages list |
| `build_search_rag_prompt` | Inject retrieved context into a RAG prompt |
| `format_rag_reference` | Format a retrieval result list as a reference block |
| `make_search_qa_map_fn` | HuggingFace `datasets` map function for QA records |
| `make_search_rag_map_fn` | HuggingFace `datasets` map function for RAG records |
| `normalize_question_text` | Normalise raw question fields |
| `normalize_answer_aliases` | Normalise answer alias lists |
| `collate_prompt_batch` | DataLoader collate function for `PromptSample`s |
| `prompt_batch_to_search_batch` | Convert a `PromptBatch` to a `SearchBatch` |

#### PPO (`src/training/ppo/`)

| Class / function | Description |
|-----------------|-------------|
| `compute_ppo_policy_loss_core` | Clipped PPO surrogate loss |
| `compute_reinforce_policy_loss` | REINFORCE policy gradient loss |
| `compute_trajectory_policy_loss` | Full trajectory loss with KL and entropy terms |
| `compute_grpo_outcome_advantage` | GRPO outcome-level advantage normalisation |
| `compute_rewards` | Apply KL penalty to per-token reward signal |
| `compute_value_loss` | Clipped value function loss |
| `kl_penalty` | Per-token KL divergence penalty (k1–k3 estimators) |
| `entropy_from_logits` | Per-token entropy from raw logits |
| `masked_mean` / `masked_whiten` | Masked tensor statistics for variable-length sequences |
| `clip_by_value` | Symmetric value clipping |
| `AdaptiveKLController` | PID-style KL coefficient adapter |
| `FixedKLController` | Constant KL coefficient |
| `PPORewardManager` | Assign per-sample rewards during PPO rollout collection |
| `LocalGRPOController` | In-process GRPO rollout loop with reward assignment |
| `PPOPolicyLossConfig` | Clip epsilon, entropy coefficient, KL coefficient |

## Tests

### Unit Tests

```bash
# Full unit suite
python3 -m pytest tests/unit/ -v

# Server-focused tests
python3 -m pytest tests/unit/servers/ -v

# Specific server areas
python3 -m pytest tests/unit/servers/server/billing/ -v
python3 -m pytest tests/unit/servers/server/features/hooks/ -v
python3 -m pytest tests/unit/servers/server/middleware/ -v
python3 -m pytest tests/unit/servers/server/settings/ -v
python3 -m pytest tests/unit/servers/utils/ -v          # license, tier, expiry

# ML / training tests
python3 -m pytest tests/unit/test_reward.py tests/unit/test_grpo.py tests/unit/test_llm_agent_generation.py -v
```

The server unit tests cover:

| Test area | What is tested |
|-----------|----------------|
| `server/billing/` | Circuit breaker state, endpoint responses, service layer HTTP mocks |
| `server/features/hooks/` | SSRF safety, endpoint validation, `HookValidateStatus` |
| `server/license/` | PEM stripping, `_strip_pem` boundary cases |
| `server/middleware/` | Path allowlist, license enforcement, tier gating |
| `server/settings/` | `_load_license_status`, `/settings` endpoint |
| `utils/test_license_utils.py` | RSA signature verification with real key pairs |
| `utils/test_license_expiry.py` | 18 parametrized `ExpiryWarningStage` boundary points |
| `utils/test_tier.py` | `get_tier` + `tier_at_least` matrix |

### Integration Tests

Integration tests run against a live server at `API_SERVER_URL`
(default `http://localhost:8080`). Start the web backend first, then:

```bash
# Key server integration areas
python3 -m pytest tests/integration/tests/scim/ -v
python3 -m pytest tests/integration/tests/query_history/ -v
python3 -m pytest tests/integration/tests/reporting/ -v
python3 -m pytest tests/integration/tests/search/ -v

# Full suite
python3 -m pytest tests/integration/ -v
```

Configure the target server:

```bash
API_SERVER_HOST=localhost API_SERVER_PORT=8080 python3 -m pytest tests/integration/
```

## Notes

- Dense retrieval defaults to CPU to avoid competing with trainer GPU memory.
- Set dense retrieval to `--device cuda` only on a dedicated retrieval node.
- Empty or invalid queries return empty result lists.
- Some web pages block scraping or return little usable text.
- Google Custom Search and SerpAPI are subject to their own quota and billing rules.
- BM25 serving requires Java because Pyserini uses Lucene.
