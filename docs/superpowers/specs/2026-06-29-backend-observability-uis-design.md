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

### F2 — Indexing / Workers Monitor
Show background-worker health (`light` / `heavy` / `beat` / `monitoring`).
- Per worker: status, last-run timestamp, queue depth, docs indexed, recent errors.
- Data source: `monitoring_worker` health snapshots persisted to `AgenticSearchStore` (workers expose no HTTP; the backend reads the store).
- **Acceptance:** panel lists each worker with status + last-seen; when no monitoring data exists, degrades to "no data yet" rather than erroring.

### F3 — Chat Loop Trace
Visualize `AgenticRAGLoop` (`chat_loop`) stages for a query.
- Renders the full per-stage trace: sub-query decomposition → HyDE → per-round retrieval → sufficiency check → follow-up queries → grounded synthesis.
- Reuses the existing `/api/agent/stream` `progress` events; adds an expanded (non-collapsed) debug rendering.
- **Acceptance:** each loop stage appears as a row showing its inputs/outputs; works for a `chat_loop` query end-to-end.

### F4 — Server Health Grid + Grounding Debug
- Health grid: reachability/up-down per configured server (retrieval, web, indexing/monitoring).
- Grounding debug: for the last agent run, show whether retrieval grounded (citations present) **and** whether the answer leg produced text — directly explaining the "sources but empty answer" case.
- **Acceptance:** grid shows up/down per server; grounding view distinguishes "grounded, no answer" from "answer, ungrounded".

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
