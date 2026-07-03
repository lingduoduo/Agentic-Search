# Monitoring Dashboard Demo Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate the configured file-backed SQLite database, launch the existing dev-console stack, and show verified monitoring results in the browser.

**Architecture:** A one-shot runtime script uses the repository's `AgenticSearchStore` APIs to add idempotent demo connectors, documents, and index attempts to `data/agentic_search.sqlite3`. The existing retrieval, FastAPI, and Vite processes run with matching debug/database configuration; API probes and Playwright verify the Console without modifying product code.

**Tech Stack:** Python 3.12, SQLite, FastAPI/Uvicorn, React/Vite, `AgenticSearchStore`, Playwright CLI.

## Global Constraints

- Do not modify product behavior or UI code.
- Do not create or modify a repository `.env` file.
- Do not delete or replace existing SQLite rows.
- Use stable `monitoring-demo-*` IDs so setup is repeatable.
- Keep the runtime seed helper, logs, and screenshot outside tracked source files.
- Do not expose environment secrets in output or screenshots.

---

### Task 1: Preflight the Runtime

**Files:**
- Read: `data/corpus.jsonl`
- Read: `src/internal/servers/web/app.py`
- Read: `src/internal/servers/retrieval/demo.py`
- Read: `web/package.json`

**Interfaces:**
- Consumes: documented ports `8001`, `7860`, and `5173`.
- Produces: a clean port/dependency baseline for the three-process stack.

- [ ] **Step 1: Confirm the demo corpus and frontend dependencies exist**

Run: `test -f data/corpus.jsonl && test -d web/node_modules`

Expected: exit status `0`.

- [ ] **Step 2: Check whether required ports are already occupied**

Run: `lsof -nP -iTCP:8001 -sTCP:LISTEN; lsof -nP -iTCP:7860 -sTCP:LISTEN; lsof -nP -iTCP:5173 -sTCP:LISTEN`

Expected: no listeners, or listeners positively identified as this worktree's intended processes. Do not kill unrelated processes.

- [ ] **Step 3: Confirm the worktree is clean before runtime data is created**

Run: `git status --short`

Expected: no output.

### Task 2: Populate the Configured SQLite Database

**Files:**
- Create runtime-only: `/tmp/seed_monitoring_demo.py`
- Create runtime data: `data/agentic_search.sqlite3`
- Read: `src/internal/db/store.py`
- Read: `src/internal/db/models.py`

**Interfaces:**
- Consumes: `AgenticSearchStore(path)`, `ConnectorConfig`, `StoredDocument`, and `create_index_attempt(...)`.
- Produces: two enabled connectors, one disabled connector, six documents, one pending attempt, one in-progress attempt, one successful attempt, and one failed attempt with stable IDs.

- [ ] **Step 1: Create the runtime-only seed helper**

Create `/tmp/seed_monitoring_demo.py` with:

```python
from pathlib import Path

from src.internal.db import AgenticSearchStore
from src.internal.db.models import ConnectorConfig, StoredDocument

DB_PATH = Path("data/agentic_search.sqlite3")
store = AgenticSearchStore(DB_PATH)

connectors = [
    ConnectorConfig(id="monitoring-demo-docs", name="Product Docs", source="filesystem", enabled=True, metadata={"demo": True}),
    ConnectorConfig(id="monitoring-demo-support", name="Support Knowledge", source="web", enabled=True, metadata={"demo": True}),
    ConnectorConfig(id="monitoring-demo-archive", name="Archive", source="filesystem", enabled=False, metadata={"demo": True}),
]
for connector in connectors:
    store.upsert_connector(connector)

documents = [
    ("architecture", "Agentic Search Architecture", "Product Docs", "The system separates retrieval, web orchestration, and the React dashboard."),
    ("monitoring", "Monitoring Guide", "Product Docs", "The Console shows server health, worker metrics, query transforms, and retrieval comparisons."),
    ("retrieval", "Retrieval Operations", "Product Docs", "Sparse retrieval uses TF-IDF in the local demo server and returns ranked source documents."),
    ("indexing", "Indexing Runbook", "Support Knowledge", "Index attempts progress from not_started to in_progress and then success or failed."),
    ("grounding", "Grounding Troubleshooting", "Support Knowledge", "Citations indicate retrieval grounding while answer text indicates synthesis success."),
    ("workers", "Worker Capacity", "Support Knowledge", "Queue depth and active connector counts provide a live operational snapshot."),
]
for slug, title, connector_name, contents in documents:
    connector_id = "monitoring-demo-docs" if connector_name == "Product Docs" else "monitoring-demo-support"
    store.upsert_document(
        StoredDocument(
            id=f"monitoring-demo-{slug}",
            title=title,
            contents=contents,
            url=f"https://demo.local/{slug}",
            connector_id=connector_id,
            metadata={"demo": True, "collection": connector_name},
        )
    )

attempts = [
    ("monitoring-demo-pending", "monitoring-demo-docs", "not_started", 0, 0, None),
    ("monitoring-demo-running", "monitoring-demo-support", "in_progress", 3, 12, None),
    ("monitoring-demo-success", "monitoring-demo-docs", "success", 6, 24, None),
    ("monitoring-demo-failed", "monitoring-demo-archive", "failed", 1, 2, "Source temporarily unavailable"),
]
for attempt_id, connector_id, status, total_documents, total_chunks, error in attempts:
    if store.get_index_attempt(attempt_id) is None:
        store.create_index_attempt(
            attempt_id=attempt_id,
            connector_id=connector_id,
            status=status,
            total_documents=total_documents,
            total_chunks=total_chunks,
            error=error,
            metadata={"demo": True},
        )

print({
    "db_path": str(DB_PATH.resolve()),
    "enabled_connectors": len(store.list_connectors(enabled=True)),
    "documents": len(store.list_documents()),
    "attempts": len(store.list_index_attempts()),
})
store.close()
```

- [ ] **Step 2: Run the helper from the repository root**

Run: `python /tmp/seed_monitoring_demo.py`

Expected: output includes `enabled_connectors: 2`, `documents: 6`, and `attempts: 4` on an otherwise empty database. Larger totals are valid when pre-existing records exist.

- [ ] **Step 3: Run the helper again to verify idempotence**

Run: `python /tmp/seed_monitoring_demo.py`

Expected: counts are unchanged from Step 2.

### Task 3: Launch the Three-Process Stack

**Files:**
- Runtime logs: `/tmp/agentic-search-retrieval.log`
- Runtime logs: `/tmp/agentic-search-web.log`
- Runtime logs: `/tmp/agentic-search-vite.log`

**Interfaces:**
- Consumes: `data/corpus.jsonl`, `data/agentic_search.sqlite3`, debug flags.
- Produces: retrieval at `http://127.0.0.1:8001`, web API at `http://127.0.0.1:7860`, frontend at `http://127.0.0.1:5173`.

- [ ] **Step 1: Start the demo retrieval server**

Run from the repository root in a persistent terminal session:

```bash
python -m src.internal.servers.retrieval.demo --corpus_path data/corpus.jsonl --port 8001
```

Expected: `GET http://127.0.0.1:8001/health` returns HTTP `200`.

- [ ] **Step 2: Start the web backend with SQLite and debug routing**

Run from the repository root in a persistent terminal session:

```bash
env AGENTIC_SEARCH_WEB_DB_PATH=data/agentic_search.sqlite3 AGENTIC_SEARCH_DEBUG_PANELS=1 AGENTIC_SEARCH_RETRIEVAL_URL=http://127.0.0.1:8001/retrieve SEARCH_AGENT_MODEL= OPENAI_API_KEY= uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

Expected: `GET http://127.0.0.1:7860/health` returns HTTP `200` and no local model download is attempted.

- [ ] **Step 3: Start Vite with the Console visible**

Run from `web/` in a persistent terminal session:

```bash
env VITE_DEBUG_PANELS=1 npm run dev -- --host 127.0.0.1
```

Expected: Vite reports `http://127.0.0.1:5173`.

### Task 4: Verify Monitoring APIs

**Files:**
- Read: `src/internal/servers/web/debug_router.py`

**Interfaces:**
- Consumes: running web and retrieval processes.
- Produces: captured health and worker-metric evidence.

- [ ] **Step 1: Verify server health aggregation**

Run: `curl -sS http://127.0.0.1:7860/api/debug/health`

Expected: both `web` and `retrieval` have status `up`.

- [ ] **Step 2: Verify live SQLite worker metrics**

Run: `curl -sS http://127.0.0.1:7860/api/debug/workers`

Expected: `pending_index_attempts >= 1`, `in_progress_index_attempts >= 1`, `active_connectors >= 2`, and `total_documents >= 6`.

- [ ] **Step 3: Verify the query-transform fallback**

Run: `curl -sS -X POST http://127.0.0.1:7860/api/debug/query-transform -H 'Content-Type: application/json' -d '{"query":"monitoring agent indexing health"}'`

Expected without `QT_*` configuration: `active` is `false` and `variants` contains the original query.

### Task 5: Demonstrate the Dashboard with Playwright

**Files:**
- Create runtime artifact: `/tmp/monitoring-dashboard-results.png`

**Interfaces:**
- Consumes: frontend URL and populated monitoring APIs.
- Produces: inspected Console state and a screenshot of the results.

- [ ] **Step 1: Open the frontend and inspect its accessible state**

Run: `playwright-cli -s=monitoring-demo open http://127.0.0.1:5173`

Run: `playwright-cli -s=monitoring-demo snapshot`

Expected: the top bar contains a `Console` button.

- [ ] **Step 2: Reveal Console and verify populated cards**

Run: `playwright-cli -s=monitoring-demo click "getByRole('button', { name: 'Console' })"`

Run: `playwright-cli -s=monitoring-demo snapshot`

Expected: Server Health shows web/retrieval `up`; Indexing / Workers shows at least `1` pending, `1` in progress, `6` documents, and `2` connectors.

- [ ] **Step 3: Exercise Query Transform Inspector**

Use snapshot refs to fill its query input with `monitoring agent indexing health` and submit.

Expected: the inactive, original-query fallback renders without an error.

- [ ] **Step 4: Exercise Retrieval Lab**

Use snapshot refs to fill the Retrieval Lab query with `monitoring workers indexing` and run the comparison.

Expected: supported demo-server retrieval modes show real corpus results; unsupported internal modes show explicit endpoint-unavailable states rather than a generic crash.

- [ ] **Step 5: Inspect browser diagnostics**

Run: `playwright-cli -s=monitoring-demo console`

Run: `playwright-cli -s=monitoring-demo requests`

Expected: no unexpected JavaScript errors; known 404 responses for unsupported demo retrieval modes are acceptable and visible in the UI.

- [ ] **Step 6: Capture and close**

Run: `playwright-cli -s=monitoring-demo screenshot --filename=/tmp/monitoring-dashboard-results.png --full-page`

Expected: the screenshot contains the Console panels and populated worker metrics.

Run: `playwright-cli -s=monitoring-demo close`

### Task 6: Report and Preserve the Runtime

**Files:**
- Read: `/tmp/monitoring-dashboard-results.png`
- Read: `data/agentic_search.sqlite3`

**Interfaces:**
- Consumes: API evidence, browser evidence, and process state.
- Produces: concise user-facing results and a reproducible running demo.

- [ ] **Step 1: Confirm source files remain clean**

Run: `git status --short`

Expected: no tracked source changes; the SQLite file may appear only if it is not ignored.

- [ ] **Step 2: Report exact observed values**

Include server statuses, worker counts, query-transform status, Retrieval Lab outcomes, frontend URL, database path, and screenshot path. Distinguish supported behavior from honest unavailable-mode results.

- [ ] **Step 3: Leave the local stack running**

Expected: the user can open `http://127.0.0.1:5173` and inspect the populated Console. Record process/session identifiers so they can be stopped later without broad process matching.
