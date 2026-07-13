# Architecture

[← Back to README](../README.md)

This guide explains the repository layout, agent families, request routing, and retrieval-grounded agent flow.

## Repository structure

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
│   ├── reward.py                # SearchRewardFunction + 4 reward dimensions
│   ├── judge.py                 # SimulatedPreferenceJudge (RLAIF stand-in)
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

## Agent framework and control flow

The agent layer (`src/agents/`) behind every loop the [Web backend API](api-reference.md#web-backend-api) and [runnable agent examples](training-and-evaluation.md#agent-cli) drive.

### Agent taxonomy — two families

The repo has **two parallel agent designs**; "the agent framework" is the first, and the registry covers only it.

| Family | Members | `run()` contract | LLM access | Registry? | GRPO-trainable? |
|---|---|---|---|---|---|
| **Framework loops** (`AgentLoopBase`) | `plain_generation`, `single_turn_agent`, `search_agent` (the search/tool agents), `tool_agent` | `run(messages, sampling_params, *, on_turn) → AgentLoopOutput` | injected `server_manager` (token-level) | ✅ | ✅ |
| **RAG pipeline** | `AgenticRAGLoop` (web `chat_loop`) | `run(question, *, chat_history) → AgenticRAGResult` | `LLMClient` (chat-level) | ❌ | ❌ |
| **Retrieval pipelines** | `search_tool`, `hybrid_search`, `chat_once` | retrieve → answer functions | — | ❌ | ❌ |

**Tool agents and search agents are members of the framework** — siblings under `AgentLoopBase`, sharing the registry, the `LoopController` + components, and the `server_manager` model boundary. **Agentic RAG sits *beside* the framework, not inside it:** its constructor, `run()` signature, and return type diverge from `AgentLoopBase`, so registering it would break the `dict[str, type[AgentLoopBase]]` contract — it stays a deliberate non-registry loop. A dispatch layer (registry + `resolve_agent_name` + the web intent router) picks one target per request, treating all three families as interchangeable.

**Why they're kept separate (by design).** The framework loops are *token-level* because they're built for GRPO RL training (policy gradients need `prompt_ids`/`response_ids`). `AgenticRAGLoop` is a lighter *chat-level* serving pipeline (`LLMClient`, no tokenizer/`server_manager`) doing query decomposition + HyDE + grounded synthesis. Two simple designs for two purposes beat one contract forced onto both; the boundary is enforced (the registry rejects non-conforming loops) and documented in the [agent invocation consolidation design](superpowers/specs/2026-06-25-agent-invocation-consolidation-design.md) so the families don't quietly drift together. Consolidation *is* feasible — `SearchAgentLoop` already does most of what `AgenticRAGLoop` does (sub-questions, iterative retrieval, evidence gating, citations); express agentic-RAG as a `SearchAgentLoop` config + a HyDE query-transform, bridged via the `ServerManager` protocol, and retire `AgenticRAGLoop`. It's deferred architectural-debt work, not a feature, so the families stay separate for now.

**Loop registry — one source of truth.** Agent loops register by name (`@register`) and are resolved through `get_registered_agent_loop(name)`; `resolve_agent_name` maps CLI/web aliases to the canonical loop. The registry covers the four `AgentLoopBase` loops below. `AgenticRAGLoop` (constructor + `run()` signature diverge from `AgentLoopBase`) and the retrieval pipelines (`search_tool` / `hybrid_search` / `chat_once`) are a distinct, non-registry category — see the [agent invocation consolidation design](superpowers/specs/2026-06-25-agent-invocation-consolidation-design.md).

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
See the [Web backend API](api-reference.md#web-backend-api) for the full request/response schema.

## Intent routing

The backend auto-classifies every query and dispatches to the right agent without any configuration:

| Intent | Agent loop | Trigger |
|--------|-----------|---------|
| `search` | `SearchAgentLoop` | Query needs external retrieval (web or indexed docs), or a bare entity lookup (e.g. `FAISS`) |
| `chat` | `AgenticRAGLoop` | Descriptive/conversational questions and generative asks — grounded synthesis |
| `tool` | `ToolAgentLoop` | Explicit tool use (`search_routing_tool`, custom tools) |

The router is `route_query` (`src/internal/servers/web/intent_routing.py`), dispatched by `_run_auto_routed` in `src/internal/servers/web/app.py`. It runs an LLM-backed 3-way classifier (`classify_route`) and falls back to a rule-based route (default `chat`) on ambiguous input.

**RAG-Fusion in tool mode** — `search_routing_tool` aggregates results from all configured retrieval sources (local index, Google, SerpAPI) in a single call, deduplicates by URL, and returns a ranked list with `[D1]`/`[D2]` citation labels.

**SSE streaming with progress events** — All three agent paths emit SSE events:

| Event type | When emitted | Payload |
|------------|-------------|---------|
| `progress` | Each agent turn | `{type, turn, text}` |
| `answer` | Answer token chunks | `{type, text}` |
| `done` | Stream complete | `{type, session_id, citations, documents, intent, tool_calls}` |
| `error` | Unhandled exception | `{type, detail}` |

The `on_turn` callback (`OnTurnCallback` in `src/agents/core/base.py`) is the hook that feeds per-turn events into the SSE queue from inside the agent loop.

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
