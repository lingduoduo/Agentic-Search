# Generated Context Pack

# Monitoring Dashboard Demo

## Sources

- [Specification: 2026-07-02-monitoring-dashboard-demo-design.md](../specs/2026-07-02-monitoring-dashboard-demo-design.md)
- [Plan: 2026-07-02-monitoring-dashboard-demo.md](../plans/2026-07-02-monitoring-dashboard-demo.md)

## Specification Context

### Goal

Provide a repo-native command that prepares a file-backed SQLite database and matching retrieval corpus, then run the existing dashboard against that data and show verified results in the browser.

### Verification and Results

- Query `/api/debug/health` and `/api/debug/workers` directly and retain their returned values.
- Open the frontend with Playwright, reveal Console, and inspect each panel.
- Run a Retrieval Lab query against the demo corpus.
- Capture a full-page screenshot showing the populated monitoring results.
- Check browser console and failed network requests for setup defects.

## Implementation Plan Context

### Global Constraints

- Do not modify product behavior or UI code.
- Do not create or modify a repository `.env` file.
- Do not delete or replace existing SQLite rows.
- Use stable `monitoring-demo-*` IDs so setup is repeatable.
- Commit the setup CLI and focused tests; keep generated data, logs, and screenshots untracked.
- Do not expose environment secrets in output or screenshots.

---

### Task 1: Preflight the Runtime

**Files:**
- Read: `src/internal/servers/web/app.py`
- Read: `src/internal/servers/retrieval/demo.py`
- Read: `web/package.json`

**Interfaces:**
- Consumes: documented ports `8001`, `7860`, and `5173`.
- Produces: a clean port/dependency baseline for the three-process stack.

- [ ] **Step 1: Confirm frontend dependencies exist**

Run: `test -d web/node_modules`

Expected: exit status `0`.

- [ ] **Step 2: Check whether required ports are already occupied**

Run: `lsof -nP -iTCP:8001 -sTCP:LISTEN; lsof -nP -iTCP:7860 -sTCP:LISTEN; lsof -nP -iTCP:5173 -sTCP:LISTEN`

Expected: no listeners, or listeners positively identified as this worktree's intended processes. Do not kill unrelated processes.

- [ ] **Step 3: Confirm the worktree is clean before runtime data is created**

Run: `git status --short`

Expected: no output.

### Task 2: Implement the Monitoring Demo Setup CLI

**Files:**
- Create: `examples/seed_monitoring_demo.py`
- Create: `tests/unit/test_seed_monitoring_demo.py`
- Read: `src/internal/db/store.py`
- Read: `src/internal/db/models.py`

**Interfaces:**
- Consumes: `AgenticSearchStore(path)`, `ConnectorConfig`, `StoredDocument`, `AGENTIC_SEARCH_WEB_DB_PATH`, `--db-path`, and `--corpus-path`.
- Produces: `seed_monitoring_demo(db_path: str | Path, corpus_path: str | Path) -> dict[str, object]`, a file-backed SQLite dataset, and a six-row JSONL corpus.

- [ ] **Step 1: Write focused failing tests**

Create `tests/unit/test_seed_monitoring_demo.py` with tests that:

```python
def test_seed_creates_expected_database_and_corpus(tmp_path):
    summary = seed_monitoring_demo(tmp_path / "demo.sqlite3", tmp_path / "corpus.jsonl")
    assert summary["enabled_connectors"] == 2
    assert summary["documents"] == 6
    assert summary["attempts"] == 4
    assert len((tmp_path / "corpus.jsonl").read_text().splitlines()) == 6


def test_seed_is_idempotent_and_preserves_unrelated_rows(tmp_path):
    db_path = tmp_path / "demo.sqlite3"
    corpus_path = tmp_path / "corpus.jsonl"
    store = AgenticSearchStore(db_path)
    store.upsert_connector(ConnectorConfig(id="existing", name="Existing", source="test"))
    store.close()
    first = seed_monitoring_demo(db_path, corpus_path)
    second = seed_monitoring_demo(db_path, corpus_path)
    assert first == second
    store = AgenticSearchStore(db_path)
    assert store.get_connector("existing") is not None
    assert len(store.list_index_attempts()) == 4
    store.close()

_[Section compacted.]_

### Task 3: Launch the Three-Process Stack

**Files:**
- Runtime logs: `/tmp/agentic-search-retrieval.log`
- Runtime logs: `/tmp/agentic-search-web.log`
- Runtime logs: `/tmp/agentic-search-vite.log`

**Interfaces:**
- Consumes: `data/monitoring_demo_corpus.jsonl`, `data/agentic_search.sqlite3`, debug flags.
- Produces: retrieval at `http://127.0.0.1:8001`, web API at `http://127.0.0.1:7860`, frontend at `http://127.0.0.1:5173`.

- [ ] **Step 1: Start the demo retrieval server**

Run from the repository root in a persistent terminal session:

```bash
python -m src.internal.servers.retrieval.demo --corpus_path data/monitoring_demo_corpus.jsonl --port 8001
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

_[Section compacted.]_

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

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
