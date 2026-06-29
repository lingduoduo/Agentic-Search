# Routing axes

The system makes **three independent routing decisions**. They are often
conflated ("the loop bypasses the router"), so this note pins down what each one
decides, where it lives, and how they relate.

| Axis | Decides | Where | Values |
|---|---|---|---|
| **Strategy** | *how* to answer | `route_query` — `src/internal/servers/web/intent_routing.py` | `direct_llm` / `agentic_rag` / `search_agent` / `tool_agent` |
| **Source** (web vs internal) | *which corpora* a search reads | `source_provider` → `_run_hybrid_search` / `_run_direct_search` in `app.py` | `auto` (internal + serpapi, merged), `retrieval`, `serpapi`, `browser`, `all` |
| **Backend** | *which internal index* serves a query | M10 `Router` inside `RetrievalService` (`src/internal/routing/`) | `sparse` / `dense` / `hybrid` / `metadata` / `sql` / `graph` / `api` |

These are orthogonal: Strategy picks the loop; Source picks web-vs-internal for the
retrieval-bearing strategies; Backend picks the internal index server-side.

## Key facts (so the confusion doesn't recur)

- **A web-vs-internal decision already exists** — it is `source_provider`, not the
  M10 router. Default `auto` fans out to internal `retrieval` **+** web `serpapi`
  concurrently and merges via MMR. An explicit (non-`auto`) source forces
  `route_query → SEARCH_AGENT`.

- **The M10 `Router` has no concept of "web."** Its routes are all internal
  (`hybrid`/`sql`/`graph`/`api`); it cannot drive a web-vs-internal choice. It
  selects the internal backend and runs **server-side** inside
  `RetrievalService.search` — the `/search` API takes no backend hint, so callers
  (including the agent loops) neither can nor need to re-pick it. SQL/graph/api
  targets are construct-only (no execution backend) and degrade to empty.

- **`SearchAgentLoop`'s web retriever is not wired in the web backend.** The loop
  has its own `Retriever` enum (`WEB`/`VECTOR_DB`) and a `web_search_url` config,
  selectable by the policy via `<search retriever="web">`. But `_run_search_agent`
  in `app.py` never sets `web_search_url`, so `_web_search_client` is `None` and
  `retriever="web"` **silently degrades to the internal corpus**
  (`_client_for`, `src/agents/search.py`). `web_search_url` is set only in the
  CLI training example, not in production. This is separate from `source_provider`.

### Consequence

Auto-routed search behaves differently by capability:

- `SEARCH_AGENT` **with** a local model → multi-turn `SearchAgentLoop`, **internal
  corpus only** (web silently degrades; `source_provider` fanout not applied).
- `SEARCH_AGENT` **without** a local model → degrades to the pipeline
  (`_auto_search_pipeline` → `_run_hybrid_search`), which **does** honor
  `source_provider` and can reach the web.

Making the multi-turn loop reach the web (via the existing `source_provider`/web
infrastructure, not M10) would remove that inconsistency — a possible future
change, intentionally out of scope here.
