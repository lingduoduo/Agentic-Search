# Retrieval PRD — Milestone 4: Online Signals + Production Hardening

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the online feedback loop: persist thumbs-up/thumbs-down signals via `POST /api/feedback`, surface aggregate metrics at `GET /api/admin/evals/summary` (admin-gated), emit structured log fields (`retrieval_mode`, `latency_ms`) on every query, and validate the system can sustain 50 QPS without P99 regression.

**Architecture:** `AgenticSearchStore` already owns the `retrieval_feedback` table (created in `_init_schema`). A thin `FeedbackRouter` writes to it; the existing `EvalsRouter` reads from it for the summary endpoint. Structured logging is emitted as `extra={}` dict on the existing `logger.info` call in `server.py`. Load validation is a Locust file that users run manually.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, SQLite (`AgenticSearchStore`), pytest, Locust.

**Spec:** `docs/superpowers/specs/2026-06-15-retrieval-prd-design.md` section 7 (Milestone 4).

**Gate to advance past M4:** Thumbs-up rate ≥ 65% over 500 rated queries. No P99 regression under 50 QPS load vs. M3 baseline.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `src/internal/servers/retrieval/feedback_router.py` | `POST /api/feedback` — validates signal, calls `store.save_retrieval_feedback()` |
| Modify | `src/internal/servers/web/app.py` | Register `create_feedback_router(db)` on the web app |
| Modify | `src/internal/servers/evals/api.py` | Add `GET /api/admin/evals/summary` endpoint (admin-gated) |
| Modify | `src/internal/servers/retrieval/server.py` | Emit `retrieval_mode` + `latency_ms` via `logger.info(..., extra={})` |
| Create | `tests/load/locustfile.py` | Locust load test: 50 QPS against `/search` and `/api/agent` |
| Create | `tests/unit/servers/retrieval/test_feedback_router.py` | 4 tests for `POST /api/feedback` |
| Create | `tests/unit/servers/evals/test_evals_summary.py` | 3 tests for `GET /api/admin/evals/summary` |

---

### Task 1: `POST /api/feedback` router

**Files:**
- Create: `src/internal/servers/retrieval/feedback_router.py`
- Modify: `src/internal/servers/web/app.py` (register router)
- Test: `tests/unit/servers/retrieval/test_feedback_router.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/servers/retrieval/test_feedback_router.py
"""Tests for POST /api/feedback router."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.db.store import AgenticSearchStore
from src.internal.servers.retrieval.feedback_router import create_feedback_router


def _app(db: AgenticSearchStore) -> TestClient:
    app = FastAPI()
    app.include_router(create_feedback_router(db))
    return TestClient(app)


def test_feedback_thumbs_up_persisted():
    db = AgenticSearchStore(":memory:")
    client = _app(db)

    resp = client.post(
        "/api/feedback", json={"session_id": "s1", "signal": "thumbs_up"}
    )

    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert db.get_feedback_summary()["rated_queries"] == 1


def test_feedback_thumbs_down_persisted():
    db = AgenticSearchStore(":memory:")
    client = _app(db)

    resp = client.post(
        "/api/feedback", json={"session_id": "s1", "signal": "thumbs_down"}
    )

    assert resp.status_code == 200
    summary = db.get_feedback_summary()
    assert summary["thumbs_up_rate"] == 0.0


def test_feedback_invalid_signal_rejected():
    db = AgenticSearchStore(":memory:")
    client = _app(db)

    resp = client.post("/api/feedback", json={"session_id": "s1", "signal": "meh"})

    assert resp.status_code == 422


def test_feedback_multiple_signals_accumulate():
    db = AgenticSearchStore(":memory:")
    client = _app(db)

    client.post("/api/feedback", json={"session_id": "s1", "signal": "thumbs_up"})
    client.post("/api/feedback", json={"session_id": "s2", "signal": "thumbs_up"})
    client.post("/api/feedback", json={"session_id": "s3", "signal": "thumbs_down"})

    summary = db.get_feedback_summary()
    assert summary["rated_queries"] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/servers/retrieval/test_feedback_router.py -v
```

Expected: `ImportError` — `feedback_router` not found.

- [ ] **Step 3: Implement `feedback_router.py`**

```python
# src/internal/servers/retrieval/feedback_router.py
"""POST /api/feedback — persists thumbs_up / thumbs_down signals."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from src.internal.db import AgenticSearchStore


class FeedbackRequest(BaseModel):
    session_id: str
    signal: Literal["thumbs_up", "thumbs_down"]


class FeedbackResponse(BaseModel):
    ok: bool


def create_feedback_router(db: AgenticSearchStore) -> APIRouter:
    """Return a router with POST /api/feedback."""
    router = APIRouter(tags=["feedback"])

    @router.post("/api/feedback", response_model=FeedbackResponse)
    def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
        db.save_retrieval_feedback(request.session_id, request.signal)
        return FeedbackResponse(ok=True)

    return router
```

- [ ] **Step 4: Register the router in `web/app.py`**

In `create_web_app()` (around line 252, after the db is available):

```python
# --- Retrieval feedback ---
from src.internal.servers.retrieval.feedback_router import create_feedback_router

app.include_router(create_feedback_router(db))
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/unit/servers/retrieval/test_feedback_router.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/internal/servers/retrieval/feedback_router.py \
        src/internal/servers/web/app.py \
        tests/unit/servers/retrieval/test_feedback_router.py
git commit -m "feat(retrieval/m4): POST /api/feedback persists thumbs_up/thumbs_down signals"
```

---

### Task 2: `GET /api/admin/evals/summary` endpoint

**Files:**
- Modify: `src/internal/servers/evals/api.py` (add summary route)
- Test: `tests/unit/servers/evals/test_evals_summary.py`

The `AgenticSearchStore.get_feedback_summary()` method returns `{"thumbs_up_rate": float, "ctr": float, "rated_queries": int}`.

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/servers/evals/test_evals_summary.py
"""Tests for GET /api/admin/evals/summary."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.configs import load_app_settings
from src.internal.db.store import AgenticSearchStore
from src.internal.servers.evals.api import create_evals_router


def _no_op_admin():
    return MagicMock()


def _client(db: AgenticSearchStore | None) -> TestClient:
    settings = load_app_settings()
    app = FastAPI()
    app.include_router(create_evals_router(settings, db=db, require_admin=_no_op_admin))
    return TestClient(app)


def test_evals_summary_returns_zeros_when_empty():
    db = AgenticSearchStore(":memory:")
    client = _client(db)

    resp = client.get("/api/admin/evals/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["thumbs_up_rate"] == 0.0
    assert data["ctr"] == 0.0
    assert data["rated_queries"] == 0


def test_evals_summary_with_no_db_returns_zeros():
    client = _client(db=None)

    resp = client.get("/api/admin/evals/summary")
    assert resp.status_code == 200
    assert resp.json()["rated_queries"] == 0


def test_evals_summary_reflects_stored_feedback():
    db = AgenticSearchStore(":memory:")
    db.save_retrieval_feedback("s1", "thumbs_up")
    db.save_retrieval_feedback("s2", "thumbs_up")
    db.save_retrieval_feedback("s3", "thumbs_down")
    client = _client(db)

    resp = client.get("/api/admin/evals/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["rated_queries"] == 3
    assert abs(data["thumbs_up_rate"] - 2 / 3) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/servers/evals/test_evals_summary.py -v
```

Expected: FAIL — `evals_summary` endpoint not found.

- [ ] **Step 3: Add `EvalsSummary` model and endpoint to `evals/api.py`**

Add to `create_evals_router()`, gated by `_require_admin`:

```python
class EvalsSummary(BaseModel):
    thumbs_up_rate: float
    ctr: float
    rated_queries: int

# Inside create_evals_router():
@router.get("/api/admin/evals/summary", response_model=EvalsSummary)
def evals_summary(
    _: AuthenticatedUser = Depends(_require_admin),
) -> EvalsSummary:
    """Return aggregate retrieval feedback metrics (admin only)."""
    if db is None:
        return EvalsSummary(thumbs_up_rate=0.0, ctr=0.0, rated_queries=0)
    summary = db.get_feedback_summary()
    return EvalsSummary(
        thumbs_up_rate=float(summary["thumbs_up_rate"]),
        ctr=float(summary["ctr"]),
        rated_queries=int(summary["rated_queries"]),
    )
```

The `create_evals_router` signature must accept `require_admin` as a callable or `None`:

```python
def create_evals_router(
    settings: AppSettings,
    *,
    search_url: str | None = None,
    db: AgenticSearchStore | None = None,
    require_admin: Callable | None = None,
) -> APIRouter:
    _require_admin = require_admin or make_require_admin(settings)
    ...
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/servers/evals/test_evals_summary.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/internal/servers/evals/api.py \
        tests/unit/servers/evals/test_evals_summary.py
git commit -m "feat(retrieval/m4): GET /api/admin/evals/summary returns thumbs-up rate (admin)"
```

---

### Task 3: Structured logging on every query

**Files:**
- Modify: `src/internal/servers/retrieval/server.py`

`retrieval_mode` and `latency_ms` must appear in every log line emitted by the `/search` handler so they can be indexed by any structured log aggregator (Datadog, CloudWatch, etc.).

- [ ] **Step 1: Write failing test**

```python
# In tests/unit/servers/retrieval/test_new_server.py — add:

def test_search_logs_retrieval_mode_and_latency(caplog):
    import logging

    svc = _make_service([_result()])
    client = TestClient(create_app(svc))

    with caplog.at_level(logging.INFO, logger="src.internal.servers.retrieval.server"):
        client.post("/search", json={"query": "test"})

    log_records = [r for r in caplog.records if "retrieval query" in r.message.lower()]
    assert log_records, "expected a log record for the query"
    record = log_records[0]
    assert hasattr(record, "retrieval_mode")
    assert hasattr(record, "latency_ms")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/servers/retrieval/test_new_server.py::test_search_logs_retrieval_mode_and_latency -v
```

Expected: FAIL — log record missing `retrieval_mode` attribute.

- [ ] **Step 3: Emit structured log fields in `server.py`**

Inside the `/search` handler, after computing `latency_ms`:

```python
logger.info(
    "retrieval query completed",
    extra={
        "retrieval_mode": mode,
        "latency_ms": latency_ms,
        "top_k": request.top_k,
    },
)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/servers/retrieval/test_new_server.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/internal/servers/retrieval/server.py \
        tests/unit/servers/retrieval/test_new_server.py
git commit -m "feat(retrieval/m4): emit retrieval_mode and latency_ms as structured log fields"
```

---

### Task 4: 50 QPS load test

**Files:**
- Create: `tests/load/locustfile.py`

This is a manual test file — not part of `pytest`. Run it after all three services are live to validate the P99 gate.

- [ ] **Step 1: Create Locust file**

```python
# tests/load/locustfile.py
"""Locust load test for the retrieval service at 50 QPS.

Usage:
    pip install locust
    locust -f tests/load/locustfile.py --host http://localhost:8000 \\
        --users 50 --spawn-rate 10 --run-time 60s --headless

To run against the web backend instead:
    locust -f tests/load/locustfile.py --host http://localhost:7860 \\
        --users 50 --spawn-rate 10 --run-time 60s --headless \\
        -t LoadTestAgent --tags agent

P99 success criteria (M4 gate): no regression vs. M3 baseline when running at 50 QPS.
"""

from __future__ import annotations

import random

from locust import HttpUser, between, tag, task

_QUERIES = [
    "What is retrieval augmented generation?",
    "Compare BM25 and dense retrieval",
    "How does hybrid search work?",
    "What are the best practices for chunking documents?",
    "Explain reciprocal rank fusion",
    "What is the difference between sparse and dense vectors?",
    "How do I set up an OpenSearch cluster?",
    "Weaviate vs Pinecone for vector search",
    "What is FAISS and how does it work?",
    "How to evaluate retrieval quality with NDCG?",
]


class RetrievalUser(HttpUser):
    """Targets the retrieval service /search endpoint (port 8000 by default)."""

    wait_time = between(0.01, 0.05)  # ~20-100 QPS per user at 50 users → ~50 QPS total

    @tag("search")
    @task(10)
    def search(self) -> None:
        query = random.choice(_QUERIES)
        self.client.post(
            "/search",
            json={"query": query, "top_k": 10},
            name="/search",
        )

    @tag("health")
    @task(1)
    def health(self) -> None:
        self.client.get("/health", name="/health")


class AgentUser(HttpUser):
    """Targets the web backend /api/agent endpoint (port 7860 by default)."""

    wait_time = between(0.1, 0.5)

    @tag("agent")
    @task
    def agent(self) -> None:
        query = random.choice(_QUERIES)
        self.client.post(
            "/api/agent",
            json={"query": query, "top_k": 5},
            name="/api/agent",
        )
```

- [ ] **Step 2: Verify the file parses**

```bash
python -c "import tests.load.locustfile"
```

Expected: no output (import succeeds; locust not required for import check).

- [ ] **Step 3: Commit**

```bash
git add tests/load/locustfile.py
git commit -m "feat(retrieval/m4): Locust load test for 50 QPS gate"
```

---

## Running the M4 Gate

After all tasks are complete and deployed:

```bash
# 1. Start retrieval service
python -m src.internal.servers.retrieval.demo --corpus_path data/corpus.jsonl

# 2. Run load test (requires pip install locust)
locust -f tests/load/locustfile.py \
    --host http://localhost:8000 \
    --users 50 --spawn-rate 10 --run-time 60s --headless

# 3. Collect feedback (minimum 500 queries rated before checking gate)
# POST /api/feedback {"session_id": "...", "signal": "thumbs_up"|"thumbs_down"}

# 4. Check thumbs-up rate (requires admin token)
# GET /api/admin/evals/summary
# → thumbs_up_rate must be >= 0.65
```

**Gate criteria:**
- `thumbs_up_rate >= 0.65` from `GET /api/admin/evals/summary`
- Locust P99 latency no worse than M3 baseline at 50 QPS
