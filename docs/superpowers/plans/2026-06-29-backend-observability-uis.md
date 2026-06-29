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
- verify: vitest — renders worker cards; empty state renders without error.

---

## Phase 4 — Chat Loop Trace (F3)

**T4.1 — `ChatTracePanel.tsx`**
Reuse `/api/agent/stream` `progress` events for a `chat_loop` query; render expanded per-stage trace (decompose → HyDE → retrieve → sufficiency → follow-up → synthesis) instead of the collapsed summary.
- verify: vitest — each stage from a fixture event stream renders as a row.
- note: no new backend if existing progress events carry stage detail; if not, **stop and confirm** before extending the stream schema (boundary: no agent-output changes without sign-off).

---

## Phase 4b — Query Transform Inspector (F5)  *(PR-3)*

Pre-retrieval stage; its own panel + endpoint (per-mode endpoints bypass the pipeline).

**T4b.1 — `/api/debug/query-transform` endpoint**
Build a `QueryTransformPipeline` from env (`build_query_transform_pipeline_from_env`)
and run **only** `pipeline.transform(query, filters)`; return
`{ variants, merged_filters, route, active_legs }`. No pipeline / no LLM →
`variants == [query]` + "no transform active", never 500.
- verify: pytest — stub pipeline returns >1 variant + filters; disabled pipeline
  returns `[query]` and the inactive state; no exception when LLM absent.

**T4b.2 — `QueryTransformInspector` panel**
Raw query input → render `raw → [variants]`, merged filters, route target, active legs.
- verify: vitest — variants render; "no transform active" state renders; api called with the query.

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
- PR-2: Phase 1b + Phase 2 + Phase 3 (rerank A/B + health/grounding + workers).
- PR-3: Phase 4 + Phase 4b + Phase 5 (chat trace + query-transform inspector + docs/ship).
Each PR carries its own spec/plan reference per convention.
