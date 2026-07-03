# Monitoring Dashboard Demo Design

**Date:** 2026-07-02
**Scope:** Set up and demonstrate the existing dev-console monitoring features. No product behavior or UI changes.

## Goal

Run the existing dashboard against a file-backed SQLite database, populate realistic monitoring data without deleting existing records, and show verified results in the browser.

## Runtime

- Use `data/agentic_search.sqlite3` through `AGENTIC_SEARCH_WEB_DB_PATH`.
- Enable the backend debug router with `AGENTIC_SEARCH_DEBUG_PANELS=1`.
- Enable the frontend Console control with `VITE_DEBUG_PANELS=1`.
- Run the demo retrieval server, web backend, and Vite frontend on their documented local ports.
- Pass configuration to launched processes rather than adding a repository `.env` file.

## Data Population

Use `AgenticSearchStore` APIs against the configured database. Add stable, clearly named demo records for:

- enabled and disabled connectors;
- documents associated with enabled connectors;
- pending, in-progress, successful, and failed indexing attempts.

Connector and document upserts use stable IDs. Index-attempt records are checked by stable IDs before insertion. Re-running setup therefore updates or preserves the demo dataset rather than multiplying it. Existing non-demo rows are untouched.

## Dashboard Coverage

The demonstration covers the existing Console surfaces:

1. Request Trace waterfall and its empty state before an agent run.
2. Server Health showing web and retrieval reachability.
3. Grounding status for the latest run.
4. Indexing / Workers metrics derived live from SQLite.
5. Query Transform Inspector, including the inactive fallback when no transform pipeline is configured.
6. Retrieval Lab across sparse, dense, hybrid, and graph endpoints, including explicit unavailable-mode states.

The worker cards are expected to reflect the populated database: pending attempts, in-progress attempts, total documents, and enabled connector count.

## Verification and Results

- Query `/api/debug/health` and `/api/debug/workers` directly and retain their returned values.
- Open the frontend with Playwright, reveal Console, and inspect each panel.
- Run a Retrieval Lab query against the demo corpus.
- Capture a full-page screenshot showing the populated monitoring results.
- Check browser console and failed network requests for setup defects.

## Error Handling

- If a process fails to start, inspect its log before changing configuration.
- Retrieval modes unsupported by the demo backend are shown as unavailable; they are not treated as setup failures.
- If the configured database cannot be opened, stop rather than creating data at a different path.
- Do not expose environment secrets in logs or screenshots.

## Success Criteria

- The three-process local stack is reachable.
- Console is visible only with its two debug flags enabled.
- Worker metrics display non-zero, deterministic demo values from `data/agentic_search.sqlite3`.
- Server health correctly reflects the running stack.
- Retrieval Lab returns real demo-corpus results for supported modes and honest errors for unsupported modes.
- A screenshot and concise result summary are delivered to the user.
- Repository source code remains unchanged apart from this approved documentation; runtime database and logs stay uncommitted.
