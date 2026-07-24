# Remove the dead ingestion scaffolding (PR4 — Scope C)

**Date:** 2026-07-24
**Branch:** `chore/prune-ingestion-scaffolding` (off `main`)
**Status:** design pending review

## Context

The final follow-up to the indexing/document-store simplification campaign (PR1
#461, PR2 #462 merged; PR3 #463 open). PRs 2–3 deleted the async worker fleet,
connectors, and Weaviate. That left **write-only ingestion scaffolding**: the
`connector_configs` / `documents` / `index_attempts` / `document_permissions` DB
tables, the admin routers that CRUD them, a startup seeding path, and a frontend
Connectors panel — all still "working" but feeding a pipeline that no longer
exists. `POST /admin/connectors/{id}/run` records an `IndexAttemptRecord` nothing
processes.

Unlike PR1–3 this touches **live surface** (mounted routers, a startup path, and
the React frontend), so it changes the admin API and UI — intentionally. A
blast-radius audit confirmed the safety boundaries below.

**Chosen scope: C (full clean cut)** — remove all four ingestion tables as a unit
(they form a self-contained FK cluster), both routers, the frontend panel, and the
unused ACL permission-readers.

## Safety boundaries (audit-verified)

- **The four ingestion tables are a self-contained FK cluster** (`connector_configs
  ← documents ← document_permissions`; `connector_configs ← index_attempts`). No
  live table (chat/memory/users/hooks/etc.) has an FK into them or joins them.
- **ACL is independent.** The live access path `context/preprocessing/access_filters.py`
  imports only the pure string helpers (`PUBLIC_ACL`, `prefix_user/email/group`)
  from `src.internal.access`. It never reads `documents`/`document_permissions`.
  The permission-table readers (`get_access_for_document`,
  `document_access_from_permissions`, `acl_for_document_permissions`,
  `documents_visible_to_user`) have **no live consumer** — safe to remove. The
  string helpers stay.
- **App startup stays green:** tables use `CREATE TABLE IF NOT EXISTS`; removal is
  clean once `seed_db`'s connector seeding is dropped in the same PR (seeding runs
  at startup via `web/app.py`).

## What stays (must not break)

- All non-ingestion tables/methods: `users`, `groups`, `group_members`,
  `chat_sessions`, `chat_messages`, `hooks`, `usage_reports`, `token_rate_limits`,
  `standard_answer*`, `scim*`, `user_is_active`, `user_memories`, `user_profiles`,
  `memory_trajectories`, `retrieval_feedback`.
- The ACL string-builder path: `access_filters.py` + the prefix helpers/`PUBLIC_ACL`
  in `src.internal.access`.
- User seeding in `seeding.py`.
- The `/admin/observability/summary` endpoint + `AdminOverview.tsx` — kept, but the
  summary drops its connectors/documents/indexing cards.

## Scope

### Backend — DELETE
- `src/internal/servers/connectors/api.py` (`create_connectors_router`) — whole router.
- `src/internal/servers/documents/cc_pair.py` (`create_documents_router`) — zero callers, 2 always-501 endpoints. (Check `documents/document_utils.py`, `models.py`, `private_key_types.py` for other importers before deleting; delete only what's orphaned.)
- Store methods in `db/store.py` for the four tables (all ingestion CRUD: `upsert_connector`, `get/list/delete_connector`, `upsert_document(_bulk)`, `get/list/delete_document(_bulk)`, `create/update/get/list/delete_*_index_attempt(s)`, `grant_document_access(_bulk)`, `get_document_permissions`, `documents_visible_to_user`), plus the four `CREATE TABLE` statements.
- Now-dead dataclasses in `db/models.py`: `ConnectorConfig`, `StoredDocument`, `DocumentPermission`, `IndexAttemptRecord` (and `IndexAttemptStatus` enum if orphaned) — verify no other importer.
- Unused ACL permission-readers in `src/internal/access/`: `get_access_for_document`, `document_access_from_permissions`, `acl_for_document_permissions`, `documents_visible_to_user` + their `access/__init__.py` re-exports. KEEP `PUBLIC_ACL` + prefix helpers.
- Test files exercising the removed routers/methods.

### Backend — EDIT
- `src/internal/servers/web/app.py` — remove the two imports (`create_connectors_router`, `create_documents_router`) and their two `include_router(...)` mount lines.
- `src/internal/servers/web/seeding.py` — drop `ConnectorSeed` model, the `connectors` field on `SeedConfiguration`, `_seed_connectors`, its call in `seed_db`, and the `ConnectorConfig` import. Keep user seeding.
- `src/internal/observability/admin_surface.py` — drop the connectors/documents/indexing cards + the `list_connectors`/`list_documents`/`list_index_attempts` reads from `build_admin_surface_summary`. Keep the endpoint + remaining cards.
- `src/__init__.py` — remove the now-dead lazy exports (`ConnectorConfig`, `DocumentPermission`, `IndexAttemptRecord`, `StoredDocument`).

### Frontend — DELETE / EDIT (`web/`)
- Delete `web/src/components/ConnectorPanel.tsx` + its CSS block in `styles.css` (~775-879).
- Delete `web/src/components/debug/WorkerMonitor.tsx` + `WorkerMonitor.test.tsx` (already fed by the null `/workers` endpoint).
- `web/src/App.tsx` — remove the ConnectorPanel import, the "connectors" toolbar toggle state + button, the panel render, and any WorkerMonitor usage.
- `web/src/api.ts` — remove `listConnectors`, `createConnector`, `updateConnector`, `deleteConnector`, `runConnector` (and `getWorkerMetrics` if now unused).
- `web/src/types.ts` — remove `IndexAttemptStatus`, `IndexAttemptView`, `ConnectorView`, `ConnectorCreateRequest`, the `"connectors"` view-union member, and worker-metric types.
- Reconcile `web/src/components/AdminOverview.tsx` (+ the `AdminSurfaceSummary` type) to the trimmed backend summary shape so `tsc` passes.

### Out of scope
- `VectorDbSettings`/`DISABLE_VECTOR_DB` (PR3 note — separate).
- Any live chat/memory/user/SCIM surface.

## Order (callers before callees, tables last)

1. Frontend removal + AdminOverview reconcile (removes UI callers; the summary shape it depends on is trimmed in step 4 — coordinate the type change here).
2. Backend: unmount + delete the two routers (`app.py` edits + router files) + their tests.
3. Backend: drop connector seeding (`seeding.py`) + trim `admin_surface.py` summary.
4. Backend: remove the store methods + four `CREATE TABLE`s + `db/models.py` dataclasses + `src/__init__.py` lazy exports + the tests that call them.
5. Backend: remove the unused ACL permission-readers from `access/` (keep string helpers).

(Steps 2–3 remove every caller of the store methods so step 4 is safe. Step 1's AdminOverview type must match step 3's trimmed summary — the plan pins the exact shape.)

## Verification / success criteria

1. `python -c "import src"` succeeds throughout.
2. `ruff check .` + `ruff format --check .` pass; `pytest` green (only removed-surface tests deleted).
3. **Frontend:** `cd web && npm run typecheck` passes and `npm run build` succeeds; `npm run test` (vitest) green after the WorkerMonitor/ConnectorPanel test removals.
4. App startup path: `seed_db` runs without `ConnectorConfig`; the app constructs (a TestClient/lifespan smoke check).
5. The `/admin/observability/summary` endpoint still returns (with the trimmed card set); `AdminOverview` renders it.
6. ACL/access unaffected: `access_filters.py` + prefix helpers untouched; no live path referenced the removed permission-readers.
7. `grep` confirms no surviving reference to the removed routers, store methods, dataclasses, or frontend symbols.

## Risks

Medium-high — the campaign's most cross-cutting PR (backend + DB schema + startup + React/TS). Main hazards, all with audit-backed mitigations: (a) the admin-summary contract couples backend `admin_surface.py` and frontend `AdminOverview.tsx` — the plan pins the trimmed shape and changes both together; (b) the ACL permission-readers *look* load-bearing but are verified unused — reviewer scrutiny expected, criterion 6 guards it; (c) `document_permissions` FKs `documents`, so all four tables must drop together (Scope C) to avoid a dangling FK; (d) frontend typecheck breaks unless component/api/type deletions land together — criterion 3 gates it.
