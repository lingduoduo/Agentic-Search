# Generated Context Pack

# Monitoring Dashboard Demo

## Sources

- [Specification: 2026-07-02-monitoring-dashboard-demo-design.md](../archive/specs/2026-07-02-monitoring-dashboard-demo-design.md)
- [Plan: 2026-07-02-monitoring-dashboard-demo.md](../archive/plans/2026-07-02-monitoring-dashboard-demo.md)

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

…

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

…

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

Expected: `GET http://127.0.0.1:8001/health` returns HTTP `200`.

- [ ] **Step 2: Start the web backend with SQLite and debug routing**

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
