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

```python

### tests/unit/servers/retrieval/test_feedback_router.py

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

_[Section compacted.]_

### Task 2: `GET /api/admin/evals/summary` endpoint

**Files:**
- Modify: `src/internal/servers/evals/api.py` (add summary route)
- Test: `tests/unit/servers/evals/test_evals_summary.py`

The `AgenticSearchStore.get_feedback_summary()` method returns `{"thumbs_up_rate": float, "ctr": float, "rated_queries": int}`.

- [ ] **Step 1: Write failing tests**

```python

### tests/unit/servers/evals/test_evals_summary.py

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

_[Section compacted.]_

### Task 3: Structured logging on every query

**Files:**
- Modify: `src/internal/servers/retrieval/server.py`

`retrieval_mode` and `latency_ms` must appear in every log line emitted by the `/search` handler so they can be indexed by any structured log aggregator (Datadog, CloudWatch, etc.).

- [ ] **Step 1: Write failing test**

```python

### In tests/unit/servers/retrieval/test_new_server.py — add:

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

### tests/load/locustfile.py

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

_[Section compacted.]_

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
