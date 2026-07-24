# Remove Dead Ingestion Scaffolding Implementation Plan (PR4, Scope C)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the dead ingestion scaffolding (4 DB tables, connectors + documents admin routers, connector seeding, the frontend Connectors panel + WorkerMonitor, unused ACL permission-readers) with zero live-behavior change to the surviving system.

**Architecture:** Callers-before-callees. Remove UI callers (frontend) and API callers (routers, seeding, admin_surface) first, THEN the store methods + tables, THEN the now-dead models + ACL readers. The four ingestion tables are a self-contained FK cluster; the live ACL path is independent. Spec: `docs/superpowers/specs/2026-07-24-prune-ingestion-scaffolding-design.md`.

**Tech Stack:** Python, pytest, ruff; React 19 + TypeScript + Vite + vitest (`web/`).

## Global Constraints

- Branch: `chore/prune-ingestion-scaffolding`, off `main`. Never commit to `main`.
- `python -c "import src"` + `ruff check .` + `pytest` green at the end of every backend task.
- Frontend tasks: `cd web && npm run typecheck && npm run test:unit` green; `npm run build` succeeds.
- **KEEP:** all chat/memory/user/SCIM/hooks tables + methods; the ACL string-builder path (`context/preprocessing/access_filters.py` + `PUBLIC_ACL`/`prefix_user`/`prefix_email`/`prefix_group` in `src.internal.access`); user seeding; the `/admin/observability/summary` endpoint (trimmed, not removed).
- The admin summary is consumed generically by `AdminOverview.tsx` (`summary.metrics.map`, `summary.sections.map`) — trimming cards needs NO frontend type change, only removal of now-unused icon imports.

> **Execution reorder (discovered mid-run, 2026-07-24):** Task 4 (store methods)
> was BLOCKED — two callers survived: `access.py get_access_for_document` (calls
> `store.get_document_permissions`) and `examples/seed_monitoring_demo.py` (calls
> `upsert_connector`/`upsert_document`/index-attempt methods). Both must be removed
> BEFORE the store methods. New execution order: **T1 frontend → T2 routers → T3
> seeding+admin_surface → T6 ACL readers → T-new seed_monitoring_demo → T4 store
> methods+tables → T5 dataclasses → T7 final**. The task bodies below are unchanged
> except: the store-methods Step-1 grep now also expects `access.py` and the demo
> script to be clean (they're removed first). The seed-demo removal is a small added
> task; ACL removal (T6 body) simply runs earlier.

---

### Task 1: Remove the frontend Connectors panel + WorkerMonitor

**Files:**
- Delete: `web/src/components/ConnectorPanel.tsx`
- Delete: `web/src/components/debug/WorkerMonitor.tsx` + `web/src/components/debug/WorkerMonitor.test.tsx` (grep first for the exact test path)
- Modify: `web/src/App.tsx`, `web/src/api.ts`, `web/src/types.ts`, `web/src/styles.css`, `web/src/components/AdminOverview.tsx`, and any Dev Console component that renders `WorkerMonitor`

**Interfaces:** none (removal).

- [ ] **Step 1: Map the wiring**

Run (from `web/`):
```
grep -rnE "ConnectorPanel|WorkerMonitor|listConnectors|createConnector|updateConnector|deleteConnector|runConnector|getWorkerMetrics|ConnectorView|ConnectorCreateRequest|IndexAttemptView|IndexAttemptStatus" src
```
This lists every reference to remove. Note where `WorkerMonitor` is rendered (likely a Dev Console panel, not `App.tsx`).

- [ ] **Step 2: Delete the components + tests**

```bash
git rm web/src/components/ConnectorPanel.tsx web/src/components/debug/WorkerMonitor.tsx web/src/components/debug/WorkerMonitor.test.tsx
```

- [ ] **Step 3: Remove App.tsx wiring**

In `web/src/App.tsx` remove: the `import { ConnectorPanel } from "./components/ConnectorPanel";` line; the `const [showConnectors, setShowConnectors] = useState(false);` state; the "Connectors" toolbar button block (the `icon-button` with `title="Manage connectors"`, ~lines 332-338); and the `{showConnectors && <ConnectorPanel />}` render (~line 410). Remove any WorkerMonitor import/render found in Step 1.

- [ ] **Step 4: Remove the API functions**

In `web/src/api.ts` remove `listConnectors`, `createConnector`, `updateConnector`, `deleteConnector`, `runConnector`, and `getWorkerMetrics` (only if Step 1 shows no other caller of getWorkerMetrics). Do NOT remove `getAdminSurfaceSummary`.

- [ ] **Step 5: Remove the now-dead types**

In `web/src/types.ts` remove `IndexAttemptStatus`, `IndexAttemptView`, `ConnectorView`, `ConnectorCreateRequest`, the `"connectors"` member of the view/panel union, and the worker-metric types used only by WorkerMonitor. Fix the union's remaining members.

- [ ] **Step 6: Remove the CSS + unused AdminOverview icons**

Remove the ConnectorPanel CSS block in `web/src/styles.css` (~lines 775-879; scope by its selectors). In `web/src/components/AdminOverview.tsx` remove the now-unused `connectors`/`indexing` entries from `iconByKey` and their icon imports (`Boxes`, `DatabaseZap`) so `noUnusedLocals` stays satisfied. Do NOT change the generic `.map` rendering.

- [ ] **Step 7: Verify frontend green**

Run: `cd web && npm run typecheck && npm run test:unit && npm run build`
Expected: typecheck passes (no dangling refs), vitest green, build succeeds.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore(ingestion): remove frontend Connectors panel + WorkerMonitor

The connectors admin API and worker metrics back nothing after the ingestion
pipeline removal. Drops ConnectorPanel, WorkerMonitor, their api.ts/types.ts, App
wiring, CSS, and the now-unused AdminOverview icons.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Unmount + delete the connectors and documents routers

**Files:**
- Modify: `src/internal/servers/web/app.py`
- Delete: `src/internal/servers/connectors/api.py` (+ `connectors/__init__.py` if it only re-exports the router)
- Delete: `src/internal/servers/documents/cc_pair.py` (+ trim `documents/__init__.py` re-export)
- Delete: the router tests

- [ ] **Step 1: Check the documents/ siblings before deleting**

Run: `grep -rnE "servers\.documents\.(document_utils|models|private_key_types)|from \.document_utils|from \.models|from \.private_key_types" src/ tests/ --include="*.py" | grep -v "servers/documents/"`
If `document_utils.py`/`models.py`/`private_key_types.py` have importers OUTSIDE `servers/documents/`, KEEP them (delete only `cc_pair.py` + its `__init__` export). Record findings in the report.

- [ ] **Step 2: Remove the mounts + imports in app.py**

In `src/internal/servers/web/app.py` remove the two imports (`from src.internal.servers.connectors.api import create_connectors_router`, `from src.internal.servers.documents.cc_pair import create_documents_router`) and the two `app.include_router(...)` lines that mount `create_connectors_router(db, settings)` and `create_documents_router(db, settings)`.

- [ ] **Step 3: Delete the router modules + their tests**

```bash
git rm src/internal/servers/connectors/api.py src/internal/servers/documents/cc_pair.py
```
Also delete/trim `connectors/__init__.py` and `documents/__init__.py` re-exports of the deleted routers, and `git rm` the backend tests that exercise these routers (grep `tests/` for `create_connectors_router`/`create_documents_router`/`/admin/connectors`/`/manage/admin/connector`).

- [ ] **Step 4: Verify green**

Run: `python -c "import src" && ruff check . && pytest -q`
Expected: all pass (app constructs without the two routers).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore(ingestion): unmount + delete connectors and documents admin routers

Both CRUD dead ingestion tables; the /run and /sync endpoints enqueued index
attempts for the deleted worker. No frontend calls the documents router.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Drop connector seeding + trim the admin observability summary

**Files:**
- Modify: `src/internal/servers/web/seeding.py`
- Modify: `src/internal/observability/admin_surface.py`
- Modify: admin_surface tests (`tests/` — grep `build_admin_surface_summary`)

- [ ] **Step 1: Drop connector seeding**

In `src/internal/servers/web/seeding.py` remove: the `ConnectorSeed` model, the `connectors` field on `SeedConfiguration`, the `_seed_connectors` function, its call inside `seed_db`, and the `ConnectorConfig` import. Keep `_seed_users` and everything user-related.

- [ ] **Step 2: Trim `build_admin_surface_summary`**

In `src/internal/observability/admin_surface.py`, edit `build_admin_surface_summary` to stop calling `store.list_connectors()`, `store.list_documents()`, `store.list_index_attempts()`. Remove the "Connectors" and "Indexed docs" metrics and the "connectors" and "indexing" section cards. Keep the "Users/groups" and "Tools/actions" metrics and the remaining cards (access/users, tools/hooks, etc.). Recompute `health_score` from the surviving live signals only (e.g. base it on hooks/users/SCIM state, or a constant `100` if no watch signal remains) — it must NOT reference attempts/connectors/documents. Keep the returned type `AdminSurfaceSummary` valid. Remove any now-unused helpers (`_index_attempt_detail`, `_latest_attempt_item`) if they become dead.

- [ ] **Step 3: Update admin_surface tests**

Update the tests asserting the summary shape to the trimmed card/metric set (grep `tests/` for `build_admin_surface_summary` / `"connectors"` / `"Indexed docs"`). Assertions on removed cards/metrics go; keep assertions on surviving ones.

- [ ] **Step 4: Verify green + summary still builds**

Run: `python -c "from src.internal.observability.admin_surface import build_admin_surface_summary" && ruff check . && pytest -q`
Expected: all pass; no reference to removed store methods remains in admin_surface.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore(ingestion): drop connector seeding + trim admin summary cards

seed_db no longer seeds connectors; build_admin_surface_summary drops the
connectors/indexed-docs metrics and connectors/indexing cards (which read the
dead ingestion tables) and rebases health_score on surviving signals.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Remove the ingestion store methods + the four tables

**Files:**
- Modify: `src/internal/db/store.py`
- Modify: the store tests (`tests/` — grep the removed method names)

- [ ] **Step 1: Confirm no surviving live caller of the ingestion store methods**

Run: `grep -rnE "upsert_connector|get_connector|list_connectors|delete_connector|upsert_document|upsert_documents_bulk|get_document\b|list_documents|delete_document|delete_documents_bulk|grant_document_access|get_document_permissions|documents_visible_to_user|create_index_attempt|update_index_attempt|get_index_attempt|list_index_attempts|delete_old_index_attempts|delete_stale_index_attempts" src/ examples/ --include="*.py" | grep -v "db/store.py"`
Expected: no output (routers gone in Task 2, seeding + admin_surface in Task 3). If any live `src/` caller remains, STOP and report BLOCKED.

- [ ] **Step 2: Remove the methods + the four CREATE TABLE statements**

In `src/internal/db/store.py` delete all the ingestion store methods listed in Step 1, and remove the `CREATE TABLE IF NOT EXISTS` blocks (and any index-creation) for `connector_configs`, `documents`, `document_permissions`, `index_attempts` from `_init_schema`. Leave every other table + method intact.

- [ ] **Step 3: Remove/trim the store tests for these methods**

In `tests/unit/test_db_store.py` (and any sibling) remove the test functions that exercise the deleted methods/tables (grep the method names). Keep tests for surviving tables.

- [ ] **Step 4: Verify green**

Run: `python -c "import src; from src.internal.db import AgenticSearchStore; AgenticSearchStore(':memory:')" && ruff check . && pytest -q`
Expected: store constructs (schema builds without the 4 tables); ruff clean; pytest green.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore(ingestion): remove ingestion store methods + the 4 tables

Drops connector_configs/documents/document_permissions/index_attempts and their
CRUD from AgenticSearchStore. Self-contained FK cluster; no live table references
them. Chat/memory/user/SCIM tables untouched.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Remove the now-dead DB dataclasses + lazy exports

**Files:**
- Modify: `src/internal/db/models.py`
- Modify: `src/internal/db/__init__.py` (if it re-exports them)
- Modify: `src/__init__.py` (lazy `_LAZY_EXPORTS`)

- [ ] **Step 1: Confirm the dataclasses have no surviving importer**

Run: `grep -rnE "ConnectorConfig|StoredDocument|DocumentPermission|IndexAttemptRecord|IndexAttemptStatus" src/ examples/ tests/ --include="*.py" | grep -vE "db/models.py|db/store.py"`
Expected: any remaining hits are in tests being removed or none. For each surviving live hit, STOP and report BLOCKED (something still uses the type).

- [ ] **Step 2: Remove the dataclasses**

In `src/internal/db/models.py` remove `ConnectorConfig`, `StoredDocument`, `DocumentPermission`, `IndexAttemptRecord`, and `IndexAttemptStatus` (only if the enum is now unused). Remove their re-exports from `src/internal/db/__init__.py`.

- [ ] **Step 3: Remove the lazy exports in `src/__init__.py`**

In `src/__init__.py` `_LAZY_EXPORTS`, remove the `"ConnectorConfig"`, `"DocumentPermission"`, `"IndexAttemptRecord"`, `"StoredDocument"` entries (they point at `.internal.db`).

- [ ] **Step 4: Verify green**

Run: `python -c "import src" && ruff check . && pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore(ingestion): drop dead DB dataclasses + lazy exports

ConnectorConfig/StoredDocument/DocumentPermission/IndexAttemptRecord no longer
have a table or caller.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Remove the unused ACL permission-readers

**Files:**
- Modify: `src/internal/access/access.py` (and `access/__init__.py`)
- Modify: ACL tests (`tests/` — grep the removed function names)

- [ ] **Step 1: Confirm the permission-readers have no live consumer**

Run: `grep -rnE "get_access_for_document|document_access_from_permissions|acl_for_document_permissions|documents_visible_to_user" src/ examples/ --include="*.py" | grep -v "internal/access/"`
Expected: no output. (The live path `context/preprocessing/access_filters.py` imports only `PUBLIC_ACL`/`prefix_user`/`prefix_email`/`prefix_group`.) If any live consumer exists, STOP and report BLOCKED.

- [ ] **Step 2: Remove the readers, keep the string helpers**

In `src/internal/access/access.py` remove `get_access_for_document`, `document_access_from_permissions`, `acl_for_document_permissions`, `documents_visible_to_user` (the last is on the store — remove there if not already gone in Task 4). KEEP `PUBLIC_ACL`, `prefix_user`, `prefix_email`, `prefix_group`, and anything `access_filters.py` imports. Remove their `access/__init__.py` re-exports.

- [ ] **Step 3: Remove/trim the ACL tests for the removed readers**

Grep `tests/` for the removed names and delete those test functions; keep the string-helper tests.

- [ ] **Step 4: Verify green + ACL live path intact**

Run: `python -c "from src.context.preprocessing.access_filters import *" && python -c "import src" && ruff check . && pytest -q`
Expected: the live access-filters module imports fine; ruff clean; pytest green.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore(ingestion): remove unused ACL permission-table readers

get_access_for_document / document_access_from_permissions /
acl_for_document_permissions / documents_visible_to_user had no live consumer;
the live access-filter path uses only the prefix string helpers, which stay.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Final full-stack gate + push + open PR

**Files:** none (verification + integration).

- [ ] **Step 1: Backend gate**

Run: `python -c "import src" && ruff check . && ruff format --check . && pytest -q`
Expected: all green.

- [ ] **Step 2: Frontend gate**

Run: `cd web && npm run typecheck && npm run build && npm run test:unit`
Expected: all green.

- [ ] **Step 3: App-startup smoke (seed path)**

Run: `python -c "import os; os.environ['AGENTIC_SEARCH_WEB_DB_PATH']=':memory:'; from src.internal.servers.web.app import create_web_app; create_web_app(); print('app OK')"`
Expected: `app OK` (constructs + seeds without ConnectorConfig / the removed tables). (If `create_web_app` needs args, adapt to the actual factory signature.)

- [ ] **Step 4: Confirm no surviving references + diff shape**

Run: `grep -rnE "ConnectorPanel|create_connectors_router|create_documents_router|connector_configs|index_attempts|IndexAttemptRecord|documents_visible_to_user" src/ web/src tests/ --include="*.py" --include="*.ts" --include="*.tsx"`
Expected: no output (archived docs excluded).
Run: `git diff --stat main...HEAD`
Expected: connectors/api.py, documents/cc_pair.py, ConnectorPanel.tsx, WorkerMonitor.tsx deleted; store.py, models.py, seeding.py, admin_surface.py, access.py, app.py, App.tsx, api.ts, types.ts, styles.css, AdminOverview.tsx edited; spec + plan added. Chat/memory/user tables + `access_filters.py` untouched.

- [ ] **Step 5: Push and open PR**

```bash
git push -u origin chore/prune-ingestion-scaffolding
gh pr create --base main --title "chore: remove dead ingestion scaffolding (PR4, Scope C)" --body "$(cat <<'EOF'
Follow-up to the indexing/document-store simplification campaign (PR1 #461, PR2 #462, PR3 #463).

Removes the write-only ingestion scaffolding left after the worker fleet / connectors
/ Weaviate were deleted — the pipeline it fed no longer exists.

**Backend removed**
- `connectors/api.py` + `documents/cc_pair.py` admin routers (unmounted from app.py)
- The 4 ingestion DB tables (`connector_configs`, `documents`, `document_permissions`, `index_attempts`) — a self-contained FK cluster — and all their store methods
- Dead DB dataclasses (`ConnectorConfig`/`StoredDocument`/`DocumentPermission`/`IndexAttemptRecord`) + lazy exports
- Connector seeding in `seeding.py`
- Unused ACL permission-readers (`get_access_for_document` et al.) — the live access-filter path uses only the prefix string helpers, which stay

**Backend edited**
- `admin_surface.py` — the observability summary drops its connectors/indexed-docs metrics + connectors/indexing cards; `health_score` rebased on surviving signals (endpoint kept)

**Frontend removed**
- `ConnectorPanel.tsx` (the admin Connectors UI) + `WorkerMonitor.tsx` (already fed by the null `/workers` endpoint), their api.ts/types.ts, App wiring, CSS, and now-unused AdminOverview icons

**Verified safe (audit):** the 4 tables are a self-contained FK cluster with no live table referencing them; the live ACL path (`access_filters.py`) never reads `documents`/`document_permissions`; chat/memory/user/SCIM surface untouched. Backend suite green; `npm run typecheck`/`build`/`test:unit` green; app-startup + seed smoke passes.

This is a user-facing change: the admin **Connectors panel is removed** from the dashboard and the connectors/documents admin API endpoints are gone.

Spec: `docs/superpowers/specs/2026-07-24-prune-ingestion-scaffolding-design.md`
Plan: `docs/superpowers/plans/2026-07-24-prune-ingestion-scaffolding.md`

Closes out the campaign's dead-Onyx-heritage removal.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

**Spec coverage:** every spec removal maps to a task — frontend panel/WorkerMonitor (T1), routers (T2), seeding + admin_surface (T3), store methods + tables (T4), dataclasses + lazy exports (T5), ACL readers (T6). Order (callers→callees→tables→models→ACL) matches the spec. Frontend/backend summary coupling resolved: AdminOverview consumes the summary generically, so T1 only drops unused icons and T3 trims the data — no shared-type change. Final full-stack gate + PR is T7.

**Placeholder scan:** no vague steps. Mechanical edits (app.py mounts, App.tsx wiring, lazy exports) have exact targets; the two genuine refactors (`admin_surface` health_score, `store.py` method/table removal) have explicit intent + constraints + a grep/test gate rather than guessed code, which is correct for read-then-edit work verified by the suite.

**Type consistency:** the kept ACL helpers (`PUBLIC_ACL`, `prefix_user`, `prefix_email`, `prefix_group`) are named identically in the spec, T6 removal-exclusion list, and the T6 verification import. The four table names and four dataclass names are consistent across T4/T5 and the T7 grep. `AdminSurfaceSummary`/`AdminSurfaceMetric`/`AdminSurfaceCard` shapes are unchanged (only entries removed).
