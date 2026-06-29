# Plan: Backend Observability UIs (Dev Console)

**Spec:** [2026-06-29-backend-observability-uis-design.md](../specs/2026-06-29-backend-observability-uis-design.md)
**Date:** 2026-06-29
**Branch:** `feat/dev-console-observability`

Sequenced, independently verifiable tasks. Each task lists its verify step. Phases ship in order; within a phase, items are ordered by dependency. Backend-first so the frontend has real endpoints to render.

---

## Phase 0 — Gating + scaffolding

**T0.1 — Backend `DEBUG_PANELS` gate**
Add `debug_panels: bool` to web settings (`AGENTIC_SEARCH_DEBUG_PANELS`, default false). Register `create_debug_router(...)` in `create_web_app()` only when enabled.
- verify: unit test — router present when flag on, 404 on `/api/debug/health` when flag off.

**T0.2 — Frontend Console nav + flag**
`VITE_DEBUG_PANELS` flag; `ConsoleNav.tsx` toggle (Search ↔ Console); `App.tsx` view state. Console hidden when flag off.
- verify: vitest — Console nav renders only when flag on; toggling switches views; `npm run typecheck` clean.

---

## Phase 1 — Retrieval Lab (F1)  *(highest value; mirrors this session's debugging)*

**T1.1 — Debug retrieval proxy**
`POST /api/debug/retrieval/{mode}` (`mode ∈ sparse|dense|hybrid|graph`) forwards to retrieval `search_url` host's `/internal/search/{mode}`, passing `top_k` + hybrid knobs. **Pass upstream status through verbatim** (404/503/200) — no swallowing.
- verify: pytest mirroring `test_eval_router.py` — 200 forwards results; 404 (endpoint missing) and 503 (dense not configured) propagate; mode string preserved.

**T1.2 — `RetrievalLab.tsx`**
Inputs (base URL default = configured host, query, `top_k`, `rrf_k`, `mmr_lambda`, `over_fetch`); run all four modes; per-mode table (`rank | doc_id | score | title`, + `retrieval_mode`, `latency_ms`); side-by-side diff.
- verify: vitest — renders four mode tables; **error states**: 404 → "endpoint not available", 503 → "dense not configured", hybrid notes sparse-only collapse.

**T1.3 — `api.ts` + types**
`getDebugRetrieval(mode, params)` via `requestJson`; `DebugRetrievalView` types.
- verify: `npm run typecheck`; covered by T1.2 tests.

---

## Phase 1b — Rerank A/B (F1b)  *(PR-2; builds on the Retrieval Lab)*

Reranking is a post-retrieval stage, surfaced as a before/after toggle — not a new mode.

**T1b.1 — `rerank` param on internal endpoints**
Add `rerank: bool = False` to the internal search request(s). When true, fetch the
mode's candidates then apply `build_reranker_from_env()`; return
`retrieval_mode = "{mode}+reranked"`. No reranker configured → return the
un-reranked order unchanged (do not error).
- verify: pytest — `rerank=true` with a stub reranker returns `{mode}+reranked` and
  reordered ids over the same candidate set; with no reranker, order is unchanged
  and no 500.

**T1b.2 — Debug proxy passes `rerank` through**
`/api/debug/retrieval/{mode}` forwards the `rerank` flag.
- verify: pytest (MockTransport) — flag present in the forwarded body when set.

**T1b.3 — Lab rerank toggle**
Toggle in `RetrievalLab`; when on, each mode column requests `rerank=true` and shows
the reranked ordering (mark rows whose rank changed vs. the un-reranked run).
- verify: vitest — toggling re-requests with `rerank: true`; reranked `retrieval_mode`
  renders; "no reranker active" state renders when order is unchanged.

**T1b.4 — Reranker stack status (which reranker + cache stats + timeout)** *(spec §F1b-2)*
Additive status fields: active base (`Reranker`/`ONNX`/none) + wrappers (cached/async/
two-stage); `CachedReranker.stats()` (hits/misses/hit_rate) when present; surface
`RerankerTimeoutError` as a per-call warning. Build against `CachedReranker.from_env(base)`
(not `wrap`). Reranker stack is already verified fallback-safe (no provider → None; no
Redis → base unchanged).
- verify: pytest — status reports base+wrappers; with a cached stub, hit_rate increments on
  repeat query; no cache → stats omitted, no error; simulated timeout → warning + un-reranked
  fallback order.

---

## Phase 2 — Server Health Grid + Grounding Debug (F4)  *(small, unblocks "empty answer")*

**T2.1 — `GET /api/debug/health`**
Aggregate reachability of configured servers (retrieval `/health`, web self, indexing/monitoring presence). Never raises — returns up/down per server.
- verify: pytest — mixed up/down aggregates correctly; unreachable host → down, not 500.

**T2.2 — `ServerHealthGrid.tsx` + grounding view**
Health grid; grounding panel reads last agent run → shows grounded? (citations) and answered? (text) → labels "grounded, no answer" vs "answer, ungrounded".
- verify: vitest — grid renders up/down; grounding view distinguishes the two cases from fixture data.

---

## Phase 3 — Workers Monitor (F2)

**T3.1 — Persist monitoring snapshots**
If `monitoring_worker` doesn't already persist to `AgenticSearchStore`, add a snapshot write (status, last-run, queue depth, docs indexed, recent errors).
- verify: pytest — `run_once()` writes a readable snapshot row.

**T3.2 — `GET /api/debug/workers`**
Read latest per-worker snapshots from store; empty → `[]` (not error).
- verify: pytest — returns persisted snapshots; empty store → `[]`.

**T3.3 — `WorkerMonitor.tsx`**
Per-worker cards (status, last-seen, queue depth, errors); "no data yet" when empty.
Pipeline-side view that complements the existing `ConnectorPanel` (source-side) — surface
in-flight index attempts + `ConnectorFailure` detail, and deep-link a row to the matching
ConnectorPanel entry. **No standalone connectors panel** (already covered).
- verify: vitest — renders worker cards; empty state renders without error; a row with a
  connector id links to the connector view.

---

## Phase 4 — Chat Loop Trace (F3)

**T4.1 — `ChatTracePanel.tsx`**
Reuse `/api/agent/stream` `progress` events for a `chat_loop` query; render expanded per-stage trace (decompose → HyDE → retrieve → sufficiency → follow-up → synthesis) instead of the collapsed summary.
- verify: vitest — each stage from a fixture event stream renders as a row.
- note: no new backend if existing progress events carry stage detail; if not, **stop and confirm** before extending the stream schema (boundary: no agent-output changes without sign-off).
- agent-loop UI already exists (progress log via `OnTurnCallback` + `ControlFlowTracePanel`
  + `ToolCallTracePanel`) — F3 extends it. Parse `ToolAgentLoop.action_trace` as
  newline-JSON of `ToolExecutionResult.to_dict()`.
- ⚠️ honor finding: `ToolAgentLoop` passes `doc_count=0` to `on_turn` — the "· N docs"
  progress line is `0` in tool mode; label honestly, don't imply "no docs."
- `BaseAgent`/`graph_base.py` is a separate Pydantic agent track, **not** covered here.

---

## Phase 4b — Query Transform Inspector (F5)  *(PR-3)*

Pre-retrieval stage; its own panel + endpoint (per-mode endpoints bypass the pipeline).

**T4b.0 — (optional, smaller first slice) `/api/debug/query-enhance` (primitive layer)**
Thin endpoint running `QueryEnhancer(llm).enhance(query)` → `QueryBundle`
{ original, sub_queries, hyde_text, step_back_query }. No filters/route — the raw
decompose/HyDE/step-back layer that `AgenticRAGLoop` uses. `QueryEnhancer` is already
verified fallback-safe (no-LLM and LLM-raise) and trivially callable. See spec §F5a.
- verify: pytest — no LLM → `sub_queries == [query]`, `hyde_text/step_back_query == None`,
  no 500; stub LLM → populated bundle.

**T4b.1 — `/api/debug/query-transform` endpoint**
Build a `QueryTransformPipeline` from env (`build_query_transform_pipeline_from_env`)
and run **only** `pipeline.transform(query, filters)`; return
`{ variants, merged_filters, route, active_legs }`. No pipeline / no LLM →
`variants == [query]` + "no transform active", never 500. Builds on / supersedes T4b.0's
view by adding filters + route + per-leg state.
- verify: pytest — stub pipeline returns >1 variant + filters; disabled pipeline
  returns `[query]` and the inactive state; no exception when LLM absent.

**T4b.2 — Wrapper-stack status (which `*Pipeline` layers are active)** *(spec §F5b)*
Report the active layer set — base + routed/cached/async — derived from the **factory
`QT_*` flags**, not from object types (`Async` wraps unconditionally; `Cached` self-gates
on URL). Build against the verified `*Pipeline` names (no bare `QueryTransform` classes).
Stack already verified fallback-safe (no flags → factory returns `None`; no cache URL →
base unchanged).
- verify: pytest — factory with no flags → status "transform inactive" (`None`), no error;
  `QT_ASYNC`/`QT_ROUTER` set → those layers report active; no cache URL → cached layer
  reported off.

**T4b.2 — `QueryTransformInspector` panel**
Raw query input → render `raw → [variants]`, merged filters, route target, active legs.
- verify: vitest — variants render; "no transform active" state renders; api called with the query.

---

## Phase 6 — Request Trace spine (F6)  *(spans PR-2/PR-3; absorbs R1/P1, links F5)*

The control-flow event model already exists (`control_flow_trace` with `duration_ms` +
`details`), emitted from the real path. F6 is render + enrich, not greenfield.

**T6.1 — `RequestTracePanel` waterfall (D1.0; frontend-only, PR-2)**
Lay out `control_flow_trace` as a timeline (bar width ∝ `duration_ms`, color by `status`,
grouped by `component`/`turn`); click a span → render `details`.
- verify: vitest — from a fixture trace, bars render with width ∝ duration_ms; clicking a
  bar reveals its details; empty trace → "no trace yet".

**T6.2 — Drill-down renderer registry (D1.1; PR-3)**
Map `component → renderer`: route → R1 view (retriever/confidence/construction_target),
query_transform → F5 view (variants/filters), search_tool → docs table, answer_generator →
prompt+completion, **`SearchAgentLoop` turn → raw `<think>/<search>/<information>/<answer>`
view** (the only four tags emitted; the model's reasoning the action trace elides). Fall
back to raw JSON for unknown components.
- verify: vitest — each known component renders its typed drill-down; a turn span with raw
  tags renders the think/search/answer view; unknown → JSON.

**T6.3 — Enrich `details` at emit sites (D1.1; PR-3)**
Add additive payload keys at the route / query-transform / search_tool / answer_generator
emit sites. Heavy payloads (prompt, completion, all docs) **only when the debug/trace flag
is set**.
- verify: pytest — with flag on, the emitted events carry the new keys; **with flag off,
  the agent response payload is byte-identical to a golden snapshot** (byte-stability guard).

**T6.4 — Live waterfall (D1.2; PR-3, optional)**
Render the streamed trace events filling the timeline during a run.
- verify: vitest — events appended to the panel as a fixture stream yields them.

---

## Phase 5 — Wire-up, docs, ship

**T5.1 — Console assembly**
Mount F1–F4 under Console view; consistent layout in `styles.css` (no new CSS deps).
- verify: vitest smoke — Console renders all four panels with flag on.

**T5.2 — Docs**
README "Dev Console" subsection: how to enable (`AGENTIC_SEARCH_DEBUG_PANELS` + `VITE_DEBUG_PANELS`), what each panel shows.
- verify: manual read; links resolve.

**T5.3 — Full gates**
`pytest`, `cd web && npm run typecheck && npm run test`, `ruff check . && ruff format .`. Web tests via `examples/run_web_integration_tests.sh` (avoid lifespan model-load hang).
- verify: all green.

**T5.4 — PR**
Branch `feat/dev-console-observability`; PR links spec + this plan. Specific title.

---

## Resolved open questions (from spec §8)

Carried as decisions to confirm at the top of Phase 1/3:
1. **Worker health source** → T3.1 adds persistence if absent (assumed absent until checked).
2. **Retrieval base URL** → default to configured `search_url`; allow arbitrary host as an explicit dev-only input, with a code comment noting the (accepted, dev-only) SSRF surface.
3. **Nav placement** → simple top toggle (`ConsoleNav`), lowest footprint in current `App.tsx`.

If any of these three should flip, say so before Phase 1.

## Suggested PR slicing
- PR-1: Phase 0 + Phase 1 (gate + Retrieval Lab) — self-contained, highest value. **(open: #353)**
- PR-2: Phase 1b + Phase 2 + Phase 3 + T6.1 (rerank A/B + health/grounding + workers + Request Trace waterfall).
- PR-3: Phase 4 + Phase 4b + Phase 6 (T6.2–T6.4) + Phase 5 (chat trace + QT inspector + trace drill-downs/enrichment + docs/ship).
Each PR carries its own spec/plan reference per convention.

**F6 is the spine:** R1 (Router) and P1 (Prompt) ship as F6 drill-downs (T6.2/T6.3), not
standalone panels. F5 keeps its standalone endpoint *and* appears as a trace drill-down.
D2 (Compare two traces) and D3 (Diagnostics overlay) are natural follow-ups once F6 lands.
