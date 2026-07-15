# Generated Context Pack

# Retrieval PRD — Milestone 1: BM25 Baseline + Service Skeleton

## Sources

- [Plan: 2026-06-15-retrieval-m1-bm25-service-skeleton.md](../plans/2026-06-15-retrieval-m1-bm25-service-skeleton.md)

## Implementation Plan Context

### Task 1: `RetrievalResult` dataclass + `RetrievalBackend` ABC

**Files:**
- Create: `src/internal/retrieval/__init__.py`
- Create: `src/internal/retrieval/backends/__init__.py`
- Create: `src/internal/retrieval/backends/base.py`
- Test: `tests/unit/retrieval/test_retrieval_backend.py`

- [ ] **Step 1: Write failing tests**

```python

### tests/unit/retrieval/test_retrieval_backend.py

"""Tests for RetrievalResult and RetrievalBackend ABC."""
from __future__ import annotations
import pytest
from src.internal.retrieval.backends.base import RetrievalBackend, RetrievalResult


def test_retrieval_result_defaults():
    r = RetrievalResult(doc_id="d1", title="T", text="body", url=None, score=0.5)
    assert r.metadata == {}


def test_retrieval_result_stores_fields():
    r = RetrievalResult(
        doc_id="d1", title="Title", text="body", url="https://x.com", score=0.9,
        metadata={"src": "corpus"},
    )
    assert r.doc_id == "d1"
    assert r.score == 0.9
    assert r.url == "https://x.com"
    assert r.metadata == {"src": "corpus"}


def test_retrieval_backend_is_abstract():
    """Cannot instantiate RetrievalBackend directly."""
    with pytest.raises(TypeError):
        RetrievalBackend()  # type: ignore[abstract]


class _ConcreteBackend(RetrievalBackend):
    def search_sparse(self, query: str, top_k: int) -> list[RetrievalResult]:
        return []

    def search_dense(self, query: str, top_k: int) -> list[RetrievalResult]:
        raise NotImplementedError


def test_concrete_backend_instantiates():
    b = _ConcreteBackend()
    assert b.search_sparse("q", 5) == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/retrieval/test_retrieval_backend.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.internal.retrieval'`

- [ ] **Step 3: Create package markers**

```python

### Task 2: `LocalBackend` wrapping `SparseRetriever`

**Files:**
- Create: `src/internal/retrieval/backends/local.py`
- Modify: `tests/unit/retrieval/test_retrieval_backend.py` (append)

- [ ] **Step 1: Append failing tests**

```python

### Append to tests/unit/retrieval/test_retrieval_backend.py

from unittest.mock import MagicMock
from src.internal.retrieval.backends.local import LocalBackend, _row_to_result


def _fake_sparse_retriever(rows: list[dict]):
    """Returns a mock SparseRetriever whose retrieve() returns rows."""
    m = MagicMock()
    m.retrieve.return_value = [rows]
    return m


def test_row_to_result_standard_keys():
    row = {
        "document": {"id": "d1", "title": "T1", "contents": "body text", "url": "https://x.com"},
        "score": 0.8,
    }
    r = _row_to_result(row)
    assert r.doc_id == "d1"
    assert r.title == "T1"
    assert r.text == "body text"
    assert r.url == "https://x.com"
    assert r.score == 0.8


def test_row_to_result_quoted_title_prefix_stripped():
    # Corpus format: '"Title"\nBody...' — title prefix is stripped
    row = {
        "document": {"id": "d2", "title": "T2", "contents": '"T2"\nActual body'},
        "score": 0.5,
    }
    r = _row_to_result(row)
    assert r.text == "Actual body"


def test_local_backend_search_sparse(monkeypatch):
    import src.internal.retrieval.backends.local as local_mod

    rows = [
        {"document": {"id": "d1", "title": "T1", "contents": "body", "url": None}, "score": 0.9},
        {"document": {"id": "d2", "title": "T2", "contents": "text", "url": None}, "score": 0.7},
    ]
    fake = _fake_sparse_retriever(rows)
    monkeypatch.setattr(local_mod, "_make_sparse_retriever", lambda cfg: fake)

    from src.internal.document_index.retrieval import SparseRetrieverConfig
    backend = LocalBackend(SparseRetrieverConfig(index_path="x", corpus_path="y"))

_[Section compacted.]_

### Task 3: `RetrievalService` with backend selection

**Files:**
- Create: `src/internal/retrieval/service.py`
- Create: `tests/unit/retrieval/test_service.py`

- [ ] **Step 1: Write failing tests**

```python

### tests/unit/retrieval/test_service.py

"""Tests for RetrievalService."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.service import RetrievalService


def _make_result(doc_id: str, score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(doc_id=doc_id, title="T", text="body", url=None, score=score)


def test_search_delegates_to_backend_sparse():
    backend = MagicMock()
    backend.search_sparse.return_value = [_make_result("d1")]
    service = RetrievalService(backend)

    results, mode = service.search("procurement", top_k=5)

    backend.search_sparse.assert_called_once_with("procurement", top_k=5)
    assert mode == "sparse"
    assert len(results) == 1
    assert results[0].doc_id == "d1"


def test_search_returns_empty_list_on_no_results():
    backend = MagicMock()
    backend.search_sparse.return_value = []
    service = RetrievalService(backend)

    results, mode = service.search("nothing", top_k=5)

    assert results == []
    assert mode == "sparse"


def test_from_env_raises_on_unknown_backend(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_BACKEND", "pinecone")
    with pytest.raises(ValueError, match="Unknown RETRIEVAL_BACKEND"):
        RetrievalService.from_env()


def test_from_env_local_requires_index_path(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_BACKEND", "local")
    monkeypatch.delenv("BM25_INDEX_PATH", raising=False)
    with pytest.raises(KeyError):
        RetrievalService.from_env()
```

_[Section compacted.]_

### Task 4: FastAPI server — `POST /search` and `GET /health`

**Files:**
- Create: `src/internal/servers/retrieval/server.py`
- Create: `tests/unit/servers/retrieval/test_new_server.py`

- [ ] **Step 1: Write failing tests**

```python

### tests/unit/servers/retrieval/test_new_server.py

"""Tests for the new retrieval service FastAPI app (server.py)."""
from __future__ import annotations

from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.service import RetrievalService
from src.internal.servers.retrieval.server import create_app


def _make_service(results: list[RetrievalResult], mode: str = "sparse") -> RetrievalService:
    svc = MagicMock(spec=RetrievalService)
    svc.search.return_value = (results, mode)
    return svc


def _result(doc_id: str = "d1", score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(doc_id=doc_id, title="Title", text="body", url="https://x.com", score=score)


def test_health_returns_ok():
    client = TestClient(create_app(_make_service([])))
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "backend" in data


def test_search_returns_results():
    svc = _make_service([_result("d1", 0.9), _result("d2", 0.7)])
    client = TestClient(create_app(svc))

    resp = client.post("/search", json={"query": "procurement", "top_k": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["retrieval_mode"] == "sparse"
    assert data["executed_queries"] == ["procurement"]
    assert len(data["results"]) == 2
    assert data["results"][0]["doc_id"] == "d1"
    assert "latency_ms" in data


def test_search_calls_service_with_top_k():
    svc = _make_service([])
    client = TestClient(create_app(svc))

_[Section compacted.]_

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
