# Generated Context Pack

# Retrieval PRD — Milestone 4: Online Signals + Production Hardening

## Sources

- [Plan: 2026-06-16-retrieval-m4-online-signals.md](../plans/2026-06-16-retrieval-m4-online-signals.md)

## Implementation Plan Context

### Task 1: `POST /api/feedback` router

**Files:**
- Create: `src/internal/servers/retrieval/feedback_router.py`
- Modify: `src/internal/servers/web/app.py` (register router)
- Test: `tests/unit/servers/retrieval/test_feedback_router.py`

- [ ] **Step 1: Write failing tests**

- [ ] **Step 2: Run tests to verify they fail**

Expected: `ImportError` — `feedback_router` not found.

- [ ] **Step 3: Implement `feedback_router.py`**

- [ ] **Step 4: Register the router in `web/app.py`**

In `create_web_app()` (around line 252, after the db is available):

- [ ] **Step 5: Run tests to verify they pass**

Expected: 4 passed.

- [ ] **Step 6: Commit**

---

### Task 2: `GET /api/admin/evals/summary` endpoint

**Files:**
- Modify: `src/internal/servers/evals/api.py` (add summary route)
- Test: `tests/unit/servers/evals/test_evals_summary.py`

The `AgenticSearchStore.get_feedback_summary()` method returns `{"thumbs_up_rate": float, "ctr": float, "rated_queries": int}`.

- [ ] **Step 1: Write failing tests**

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL — `evals_summary` endpoint not found.

- [ ] **Step 3: Add `EvalsSummary` model and endpoint to `evals/api.py`**

Add to `create_evals_router()`, gated by `_require_admin`:

The `create_evals_router` signature must accept `require_admin` as a callable or `None`:

- [ ] **Step 4: Run tests to verify they pass**

Expected: 3 passed.

…

### Task 3: Structured logging on every query

**Files:**
- Modify: `src/internal/servers/retrieval/server.py`

`retrieval_mode` and `latency_ms` must appear in every log line emitted by the `/search` handler so they can be indexed by any structured log aggregator (Datadog, CloudWatch, etc.).

- [ ] **Step 1: Write failing test**

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — log record missing `retrieval_mode` attribute.

- [ ] **Step 3: Emit structured log fields in `server.py`**

Inside the `/search` handler, after computing `latency_ms`:

- [ ] **Step 4: Run test to verify it passes**

Expected: all pass.

- [ ] **Step 5: Commit**

---

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
