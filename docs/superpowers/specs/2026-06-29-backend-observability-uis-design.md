# Spec: Backend Observability UIs (Dev Console)

**Date:** 2026-06-29
**Status:** Draft — awaiting confirmation before implementation
**Scope:** Add dev/debug observability panels to the existing 7860 React app for every backend server.

---

## 1. Objective

Give developers a single in-app **Dev Console** (in the existing 7860 React frontend) to observe what each backend server is actually doing, so failures are diagnosable in the UI instead of via ad-hoc `curl`.

Concretely, make these recurring confusions self-evident:
- "Sources came back but the **Answer is empty**" → show whether the answer/generation leg ran.
- "**Hybrid looks identical to sparse**" → show per-mode results + whether the dense leg is available (503).
- "Endpoint returns nothing" → distinguish *server down* / *endpoint not mounted (404)* / *dense not configured (503)* from real results.
- "Is indexing actually running?" → show worker health, queue depth, last-run.

**Target users:** developers/operators running this stack locally (and, gated, on a deployed instance).

**Non-goals:** end-user product features; production operator console with full RBAC (a later spec can harden it); per-server standalone HTML pages (explicitly rejected — we extend the one React app).

**Decisions locked (from clarification):**
- Delivery: **new panels in the existing 7860 React app** (unified console).
- Scope: **all services** — retrieval, indexing/workers, chat loop, web-backend internals.
- Purpose: **dev/debug observability** — dev-only, no new auth, **off by default in production**.

---

## 2. Features & Acceptance Criteria

A top-level nav toggles between the existing **Search** view and a new **Console** view. Console hosts four panels.

### F1 — Retrieval Lab
Inspect the retrieval server's per-mode endpoints.
- Inputs: retrieval base URL (defaults to the backend's configured `search_url` host), query, `top_k`, hybrid knobs (`rrf_k`, `mmr_lambda`, `over_fetch`).
- Runs `sparse` / `dense` / `hybrid` / `graph` and renders, per mode: `retrieval_mode`, `latency_ms`, and a result table (`rank | doc_id | score | title`).
- A side-by-side diff highlights ranking differences across modes.
- **Acceptance:**
  - Against `server.py`: all four modes render with correct `retrieval_mode` strings.
  - Against `demo.py` (no `/internal/search/*`): shows a clear "endpoint not available (404)" message, not a cryptic parse error.
  - Dense unavailable: `/dense` shows **503 "dense not configured"**; hybrid panel notes it collapsed to sparse-only.

### F1b — Rerank A/B (Retrieval Lab enhancement)
Reranking is **not** a 5th mode peer to sparse/dense/hybrid/graph — it is a
post-retrieval reordering stage applied *on top of* a mode (the cross-encoder in
`RetrievalService`, tagged `{mode}+reranked`). The per-mode endpoints today apply
only `mmr_rerank` (diversity), never the cross-encoder. So the useful observability
is **before vs. after on identical candidates**.
- Internal endpoints gain an optional `rerank: bool` (default false). When true,
  the endpoint fetches the mode's candidates then applies the env-configured
  cross-encoder (`build_reranker_from_env()`) and returns `retrieval_mode = "{mode}+reranked"`.
- The Lab gains a **rerank toggle**; when on, each mode column shows the reranked
  ordering (and, ideally, a marker on rows whose rank changed).
- **Acceptance:**
  - With a reranker configured: `hybrid` with `rerank=true` returns
    `retrieval_mode = "hybrid+reranked"` and a (possibly) reordered list over the
    *same* candidate doc_ids.
  - With **no** reranker configured: the endpoint returns the un-reranked order
    unchanged (and the mode string makes "no reranker active" visible) rather than
    erroring — itself a useful signal.

#### F1b-2 — Reranker stack observability (which reranker, cache, timeout)
The reranker isn't one object — it's a **composable stack** (verified 2026-06-29,
`src/internal/retrieval/`), all sharing a `rerank(query, results, top_k)` interface:
- `Reranker` (`reranker.py`) — base cross-encoder; `from_env()` → `None` if
  `RERANKER_PROVIDER` unset (**fallback-safe, verified**).
- `ONNXReranker` (`onnx_reranker.py`) — drop-in via ONNX runtime (needs `optimum`);
  `from_env()` falls back to `Reranker`, `None` if no provider (**verified**).
- `CachedReranker` (`cached_reranker.py`) — Redis score cache (key = query + sorted
  doc_ids); `from_env(base)` returns **`base` unchanged** when `RERANKER_CACHE_REDIS_URL`
  is unset (transparent no-op, **verified `is base`**). Exposes `stats() →
  {hits, misses, hit_rate}`.
- `AsyncReranker` (`async_reranker.py`) — wraps any reranker, offloads to a thread pool
  with a timeout (`RerankerTimeoutError` on overrun).
- `TwoStageReranker` (`two_stage_reranker.py`) — fast pre-filter on all candidates →
  heavy scorer on top-N.
- Assembled by `reranker_factory.py`; `reranker_benchmark.py` exists for offline timing.

So beyond before/after ordering, the debug surface should answer **"what is actually
running and is it healthy?"** Additive fields on the rerank response / a small status block:
- **active stack** — base provider (`Reranker`/`ONNX`/none) + which wrappers are on
  (cached? async? two-stage?). Makes "no reranker active" and "ONNX fell back to
  cross-encoder" visible.
- **cache stats** — `CachedReranker.stats()` (`hits / misses / hit_rate`) — currently
  surfaced **nowhere**; high-value, cheap to expose.
- **timeout signal** — surface `RerankerTimeoutError` (async overrun) as a per-call
  warning rather than a silent drop to un-reranked order.
- **Acceptance:**
  - Status block reports the active base + wrappers; with nothing configured it reads
    "no reranker active" (not an error).
  - When `CachedReranker` is in the stack, `hit_rate` renders and increments on a repeat
    query; absent cache → stats omitted/"n/a", no error.
  - An async timeout shows as a labeled per-mode warning, and the column still renders the
    un-reranked fallback order.

> **Naming note (verified):** the cache wrap helper is `CachedReranker.from_env(base)`,
> **not** `CachedReranker.wrap(...)`. Build against `from_env`.

### F2 — Indexing / Workers Monitor
Show background-worker health (`light` / `heavy` / `beat` / `monitoring`).
- Per worker: status, last-run timestamp, queue depth, docs indexed, recent errors.
- Data source: `monitoring_worker` health snapshots persisted to `AgenticSearchStore` (workers expose no HTTP; the backend reads the store).
- **Acceptance:** panel lists each worker with status + last-seen; when no monitoring data exists, degrades to "no data yet" rather than erroring.

> **Connectors are out of scope as a new panel** — they already have a full UI
> (`ConnectorPanel.tsx`: CRUD + run + per-connector `last_attempt` status, last sync,
> doc count). That is the **source-side** view. F2 is the complementary **pipeline-side**
> view (the workers that process connector output: in-flight index attempts, queue
> depth, `ConnectorFailure` error detail that ConnectorPanel does not surface). F2 should
> deep-link a worker/attempt row to the matching ConnectorPanel row rather than duplicate
> connector management. No standalone connectors console panel.

### F3 — Chat Loop Trace
Visualize `AgenticRAGLoop` (`chat_loop`) stages for a query.
- Renders the full per-stage trace: sub-query decomposition → HyDE → per-round retrieval → sufficiency check → follow-up queries → grounded synthesis.
- Reuses the existing `/api/agent/stream` `progress` events; adds an expanded (non-collapsed) debug rendering.
- **Acceptance:** each loop stage appears as a row showing its inputs/outputs; works for a `chat_loop` query end-to-end.

### F4 — Server Health Grid + Grounding Debug
- Health grid: reachability/up-down per configured server (retrieval, web, indexing/monitoring).
- Grounding debug: for the last agent run, show whether retrieval grounded (citations present) **and** whether the answer leg produced text — directly explaining the "sources but empty answer" case.
- **Acceptance:** grid shows up/down per server; grounding view distinguishes "grounded, no answer" from "answer, ungrounded".

### F5 — Query Transform Inspector
Query transform is a **pre-retrieval** stage (the mirror of reranking's post-retrieval
A/B): `QueryTransformPipeline` turns one input query into N variants + merged filters +
a route decision (legs gated by `QT_DECOMPOSE` / `QT_HYDE` / `QT_STEP_BACK` /
`QT_KEYWORDS` / `QT_MULTI_QUERY` / `QT_CONSTRUCT_FILTERS` / `QT_ROUTER`). The per-mode
`/internal/search/*` endpoints bypass the pipeline, so this is its own panel, not a Lab
toggle.
- `POST /api/debug/query-transform` runs **only** `pipeline.transform(query, filters)`
  and returns `{ variants, merged_filters, route }` — no retrieval.
- Panel: enter a raw query, see `raw → [variants]` + merged filters + route target;
  surface which `QT_*` legs are active.
- Decompose/HyDE are LLM-backed: with no LLM configured the legs no-op — the panel
  shows "legs inactive (no LLM)" rather than erroring.
- **Acceptance:**
  - With LLM + a multi-leg config: a decomposable query returns >1 variant; the active
    legs are listed.
  - With no LLM / pipeline disabled: returns `variants == [query]` and an explicit
    "no transform active" state, no 500.

#### F5a — Primitive layer: `QueryEnhancer` (the raw building block)
`QueryTransformPipeline` is built on a lower-level primitive,
`QueryEnhancer` (`src/context/query_enhancer.py`) — the raw decompose/HyDE/step-back
methods the pipeline composes. Used directly by `AgenticRAGLoop` (`chat_loop`) and by
`query_transform.py` / `cached_query_transform.py`. **Verified behavior (exercised
2026-06-29, no LLM + LLM-raises + real-output paths):**
- `decompose(query) -> list[str]` — prompt asks for **2–4** focused, keyword-rich
  sub-questions; strips list markers and dedups the original out. **The 2–4 count is
  prompt-advisory, not code-enforced** — it can return 1 (atomic) or >4 depending on the LLM.
- `hyde(query) -> str | None` — one hypothetical-answer paragraph.
- `step_back(query) -> str | None` — one broader background reformulation.
- `rewrite(query) -> str | None` — canonical cleanup. **Exists but is NOT part of
  `enhance()`** — `enhance()` runs exactly decompose + hyde + step_back.
- `enhance(query) -> QueryBundle{ original, sub_queries, hyde_text, step_back_query }`;
  `QueryBundle.all_queries()` dedups sub_queries → hyde → step_back, falling back to
  `[original]`.
- **Fallback-safe on both no-LLM *and* LLM failure** (each method try/excepts and logs a
  warning): `decompose -> [query]`, `hyde / step_back / rewrite -> None`.

**Implementation option — minimal primitive inspector:** a thin
`POST /api/debug/query-enhance` running `QueryEnhancer(llm).enhance(query)` and returning
the `QueryBundle` gives the F5 essence at the raw layer without the full pipeline (no
filters/route). Cheaper than the pipeline endpoint and useful for debugging
`chat_loop`/AgenticRAGLoop specifically. Decision for F5 build time: ship the
`QueryEnhancer` inspector first (smaller, already verified callable) and layer the
`QueryTransformPipeline` view (`variants + filters + route`) on top, OR go straight to the
pipeline endpoint. Either way the panel renders `original → bundle/variants`.
- **Acceptance (primitive path):** no LLM → `QueryBundle` with `sub_queries == [query]`,
  `hyde_text == None`, `step_back_query == None`, and an explicit "no LLM" state; with an
  LLM, sub_queries/hyde/step_back populate and render.

#### F5b — Pipeline wrapper stack (which layers are active)
Like the reranker (§F1b-2), query transform is a **composable `*Pipeline` stack**
(verified 2026-06-29), assembled by `build_query_transform_pipeline_from_env`
(`src/internal/retrieval/query_transform_factory.py`), outermost → innermost:
`RoutedQueryTransformPipeline → CachedQueryTransformPipeline →
AsyncQueryTransformPipeline → QueryTransformPipeline`. Each layer optional.
- **Naming (verified):** the classes are all `*Pipeline`-suffixed —
  `QueryTransformPipeline` / `AsyncQueryTransformPipeline` / `CachedQueryTransformPipeline`
  / `RoutedQueryTransformPipeline`. There are **no** bare `QueryTransform` /
  `AsyncQueryTransform` / `CachedQueryTransform` classes. Build against the real names.
- **Fallback-safe (verified):** `build_query_transform_pipeline_from_env(None|llm)` → `None`
  when no `QT_*` legs are set (service degrades to single-query). `QueryTransformPipeline.from_env(None)` → `None`.
- **Gating is inconsistent across layers — the status surface must read the factory's
  flags, not infer from object types:**
  - `CachedQueryTransformPipeline.from_env(base)` **self-gates** on its cache URL —
    returns `base` unchanged when unset (transparent no-op, verified `is base`).
  - `AsyncQueryTransformPipeline.from_env(base)` **wraps unconditionally**; the factory
    gates it behind the `QT_ASYNC` flag, not the object.
  - `RoutedQueryTransformPipeline` is applied only under `QT_ROUTER` (and can run standalone
    over an all-off leaf).
- **Debug surface:** beyond `raw → variants`, report the **active stack** — base + which
  wrappers (routed? cached? async?) are on, derived from the `QT_*` flags / factory — so
  "transform inactive," "router-only," and "cached/async enabled" are all visible. If
  `CachedQueryTransformPipeline` exposes cache stats, surface them like the reranker's.
- **Acceptance:** status reports the active layer set from the factory flags; nothing
  configured → "transform inactive" (pipeline `None`), not an error; `QT_ASYNC`/`QT_ROUTER`
  on → those layers show active even though `Async` wouldn't self-report from its type.

> **Related finding (out of scope here):** on the main `/search`, `executed_queries`
> is hardcoded to `[request.query]` ([server.py:89]) — `service.search()` returns only
> `(results, mode)` and discards the variants, so the schema's `executed_queries` field
> never reflects the real transformed queries. F5 reads the pipeline directly and does
> not depend on fixing this, but the gap is worth a separate change.

### F6 — Request Trace (the console spine)
A single query's full journey across every stage as a timed waterfall — the
unifying view the per-stage inspectors drill into.

**Key fact: the data model already exists.** The agent loop emits a per-request
`control_flow_trace` of `ControlFlowEventView { sequence, timestamp, turn, component,
action, status, duration_ms, details }` from the *real* path (this is what
`ControlFlowTracePanel` renders live, streamed over SSE). So F6 needs no new span
collection, no hot-path instrumentation, no debug re-run — it observes the real
request for free. F6 is **render + enrich**, not greenfield.

- **D1.0 — Waterfall (no backend change):** a `RequestTracePanel` lays out
  `control_flow_trace` as a horizontal timeline (bar width ∝ `duration_ms`, color by
  `status`, grouped by `component` / `turn`). Click a span → show its `details`.
- **D1.1 — Enrich `details` at emit sites → absorbs R1 / F5 / P1 as drill-downs:**
  add additive payload keys where each component emits —
  route span → `{retriever, confidence, construction_target}` (**R1**),
  query-transform span → `variants[]` + filters (**F5**),
  search_tool span → top docs (Lab-style table),
  answer_generator span → assembled prompt + completion (**P1**).
  The per-stage inspectors become **drill-down renderers keyed by `component`** — R1,
  F5, and P1 ship as trace drill-downs rather than standalone panels.
- **D1.2 — Live waterfall:** the trace already streams; render it filling in during a run.

**Design decisions:**
1. New `RequestTracePanel` (console) rather than mutating `ControlFlowTracePanel` —
   surgical; the existing list panel is untouched.
2. **Byte-stability guard:** heavy payloads (full prompt, completion, all docs) are added
   to `details` **only when the debug/trace flag is set**. The default trace stays lean;
   production `/api/agent` responses and reward byte-stability are unchanged.

- **Acceptance:**
  - D1.0: given a `control_flow_trace`, the panel renders one bar per event with width
    proportional to `duration_ms`; clicking a bar reveals its `details`.
  - D1.1: a route event's drill-down shows retriever + confidence; an answer_generator
    event's drill-down shows prompt + completion **only** under the debug flag.
  - With the flag off, the agent response payload is byte-identical to today.

> **Supersession:** R1 (Router Inspector) and P1 (Prompt Inspector) are realized as F6
> drill-downs (D1.1), not standalone panels. F5 keeps its standalone `/api/debug/query-transform`
> endpoint (lets you inspect the transform *without* running a full request) **and** also
> appears as the query-transform drill-down in the trace.

**Global gating:** all Console panels are gated behind a `DEBUG_PANELS` flag (backend env + frontend build flag). **Default off in production.**

---

## 3. Commands

No new processes. Existing stack is unchanged (see CLAUDE.md):
```bash
# retrieval (use server.py to exercise /internal/search/* in the Retrieval Lab)
python3 -m uvicorn 'src.internal.servers.retrieval.server:create_app' --factory --port 8000
# web backend (hosts the Console)
PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
# frontend dev
cd web && npm run dev
```
Enable the Console:
```bash
export AGENTIC_SEARCH_DEBUG_PANELS=1   # backend gate
# web build flag: VITE_DEBUG_PANELS=1 (dev) / set at build for dist
```
Checks: `pytest`, `cd web && npm run typecheck && npm run test`, `ruff check . --fix && ruff format .`

---

## 4. Project Structure

**Frontend** (`web/src/`)
```
components/debug/
  ConsoleNav.tsx        # Search <-> Console toggle
  RetrievalLab.tsx      # F1
  WorkerMonitor.tsx     # F2
  ChatTracePanel.tsx    # F3
  ServerHealthGrid.tsx  # F4 (incl. grounding debug)
api.ts                  # + getDebugRetrieval(), getWorkers(), getServerHealth()
types.ts                # + Debug* view types
App.tsx                 # nav state; render Console when active + flag on
styles.css              # debug panel styles (no new CSS framework)
```

**Backend** (`src/internal/servers/web/`)
```
routers/debug.py        # create_debug_router(db, settings, search_url):
                        #   GET  /api/debug/health            -> per-server reachability
                        #   POST /api/debug/retrieval/{mode}  -> proxy to retrieval /internal/search/*
                        #   GET  /api/debug/workers           -> worker health from store
app.py                  # register router iff DEBUG_PANELS enabled
```
Worker health persistence: extend `monitoring_worker` to write snapshots to `AgenticSearchStore` if not already.

**Docs:** this spec + paired plan in `docs/superpowers/plans/2026-06-29-backend-observability-uis.md` (per project convention).

---

## 5. Code Style

- Frontend: React 19 function components, hooks, **no component library** (custom components only); follow existing `api.ts` `requestJson<T>` pattern; TypeScript strict; styles in `styles.css` (no new CSS deps).
- Backend: FastAPI `APIRouter` + Pydantic models mirroring existing routers; reuse the existing retrieval-proxy HTTP client/pattern (`search_url`) rather than introducing a new one; surface upstream status codes verbatim (no swallowing — the original bug).
- Lint/format: `ruff`. Match surrounding naming and module layout.

---

## 6. Testing Strategy

- **Backend (`pytest`):** unit tests for `debug.py` — proxy forwarding + status pass-through (200/404/503), health aggregation, graceful degradation when monitoring data absent. Mirror `tests/unit/servers/retrieval/test_eval_router.py` (MagicMock backend) and existing web-router tests. Run web tests via `examples/run_web_integration_tests.sh` to avoid the lifespan model-load hang (known gotcha).
- **Frontend (`vitest`):** component tests per panel — render, mode toggles, and **error states** (503 dense, 404 endpoint, server down). Match `web/src/__tests__` conventions.
- **Acceptance mapping:** every F1–F4 acceptance bullet has a corresponding test. Keep existing agent/reward outputs byte-stable.

---

## 7. Boundaries

**Always**
- Dev-only; **default-off in production** via `DEBUG_PANELS`.
- Read-only/inspection (the only "action" is running a query you typed).
- Reuse existing proxy/auth/HTTP patterns; surface upstream errors verbatim.

**Ask first**
- Exposing the Console in any non-local/deployed environment (auth implications).
- Adding heavy new deps (charting libraries) — prefer lightweight custom rendering first.
- Adding any new long-running process or port.

**Never**
- Expose secrets/env values in the UI.
- Allow destructive worker/index operations (reindex, purge, mutate data) from the debug UI.
- Ship per-server standalone HTML pages (rejected in favor of the unified React app).
- Break byte-stability of existing reward/agent outputs.

---

## 8. Open Questions (resolve during planning)

1. **Worker health source:** does `monitoring_worker` already persist snapshots to `AgenticSearchStore`, or do we add that write? (Affects F2 size.)
2. **Retrieval base URL:** lock to the backend-configured `search_url`, or let the Lab target an arbitrary host (handy for comparing demo vs server.py)? Arbitrary host = small SSRF surface, acceptable dev-only but worth a note.
3. **Nav placement:** simple top toggle vs. a left rail — pick the lower-footprint option that fits current `App.tsx`.
