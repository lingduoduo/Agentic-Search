# Retrieval PRD — Milestone 1: BM25 Baseline + Service Skeleton

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a single `RetrievalService` with a pluggable backend interface, wrap the existing Pyserini BM25 retriever behind it, expose `POST /search` and `GET /health` via FastAPI, and add offline eval metrics (Recall@K, NDCG@K, MRR) with a CLI runner.

**Architecture:** `RetrievalBackend` ABC → `LocalBackend` wraps existing `SparseRetriever` (no behavior change) → `RetrievalService` selects backend from `RETRIEVAL_BACKEND` env var → FastAPI `server.py` wraps the service. Eval metrics live in a standalone `eval_metrics.py` module called by `eval_runner.py`. No changes to existing `retrieval_server.py`.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, Pyserini BM25, pytest, argparse.

**Spec:** `docs/superpowers/specs/2026-06-15-retrieval-prd-design.md` sections 1–3.

**Gate to advance to M2:** `python -m src.internal.retrieval.eval_runner --dataset data/eval/qa_pairs.jsonl` reports `recall@10 >= 0.75`. `POST /search` P99 latency ≤ 300ms local.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `src/internal/retrieval/__init__.py` | Package marker |
| Create | `src/internal/retrieval/backends/__init__.py` | Package marker |
| Create | `src/internal/retrieval/backends/base.py` | `RetrievalResult` dataclass + `RetrievalBackend` ABC |
| Create | `src/internal/retrieval/backends/local.py` | `LocalBackend` wrapping `SparseRetriever` (M1: sparse only) |
| Create | `src/internal/retrieval/service.py` | `RetrievalService` — backend selection, `search()` method |
| Create | `src/internal/retrieval/eval_metrics.py` | Pure functions: `recall_at_k`, `ndcg_at_k`, `mrr` |
| Create | `src/internal/retrieval/eval_runner.py` | CLI: loads QA pairs, calls service, prints metrics JSON |
| Create | `src/internal/servers/retrieval/server.py` | FastAPI app: `POST /search`, `GET /health` |
| Create | `data/eval/qa_pairs.jsonl` | Sample annotated QA pairs for offline eval |
| Create | `data/eval/baseline_metrics.json` | Initial baseline snapshot (zeros until M1 gate is run) |
| Create | `tests/unit/retrieval/test_retrieval_backend.py` | Tests for `RetrievalResult`, `LocalBackend` |
| Create | `tests/unit/retrieval/test_eval_metrics.py` | Tests for `recall_at_k`, `ndcg_at_k`, `mrr` |
| Create | `tests/unit/retrieval/test_eval_runner.py` | Tests for `run_eval` with stub service |
| Create | `tests/unit/servers/retrieval/test_new_server.py` | Tests for `POST /search` and `GET /health` |

---

### Task 1: `RetrievalResult` dataclass + `RetrievalBackend` ABC

**Files:**
- Create: `src/internal/retrieval/__init__.py`
- Create: `src/internal/retrieval/backends/__init__.py`
- Create: `src/internal/retrieval/backends/base.py`
- Test: `tests/unit/retrieval/test_retrieval_backend.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/retrieval/test_retrieval_backend.py
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
# src/internal/retrieval/__init__.py
```

```python
# src/internal/retrieval/backends/__init__.py
```

- [ ] **Step 4: Implement `base.py`**

```python
# src/internal/retrieval/backends/base.py
"""Abstract base for all retrieval backends."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass
class RetrievalResult:
    doc_id: str
    title: str
    text: str
    url: str | None
    score: float
    metadata: dict = field(default_factory=dict)


class RetrievalBackend(abc.ABC):
    @abc.abstractmethod
    def search_sparse(self, query: str, top_k: int) -> list[RetrievalResult]:
        """BM25 keyword search. Must be implemented by every backend."""

    @abc.abstractmethod
    def search_dense(self, query: str, top_k: int) -> list[RetrievalResult]:
        """ANN vector search. Raise NotImplementedError if not supported."""
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/unit/retrieval/test_retrieval_backend.py -v
```

Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add src/internal/retrieval/ tests/unit/retrieval/test_retrieval_backend.py
git commit -m "feat(retrieval): add RetrievalResult dataclass and RetrievalBackend ABC"
```

---

### Task 2: `LocalBackend` wrapping `SparseRetriever`

**Files:**
- Create: `src/internal/retrieval/backends/local.py`
- Modify: `tests/unit/retrieval/test_retrieval_backend.py` (append)

- [ ] **Step 1: Append failing tests**

```python
# Append to tests/unit/retrieval/test_retrieval_backend.py

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
    results = backend.search_sparse("retrieval", top_k=5)

    assert len(results) == 2
    assert results[0].doc_id == "d1"
    assert results[0].score == 0.9
    assert results[1].doc_id == "d2"
    fake.retrieve.assert_called_once_with(["retrieval"], topk=5)


def test_local_backend_search_dense_raises():
    import src.internal.retrieval.backends.local as local_mod

    from src.internal.document_index.retrieval import SparseRetrieverConfig
    backend = LocalBackend.__new__(LocalBackend)
    backend._sparse = MagicMock()
    backend._dense = None
    with pytest.raises(NotImplementedError, match="Dense search not configured"):
        backend.search_dense("q", 5)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/retrieval/test_retrieval_backend.py -v -k "local or row_to"
```

Expected: `ImportError` on `local.py` imports

- [ ] **Step 3: Implement `local.py`**

```python
# src/internal/retrieval/backends/local.py
"""Local backend: wraps Pyserini SparseRetriever (BM25) and, in M2, DenseRetriever."""
from __future__ import annotations

from src.internal.document_index.retrieval import SparseRetriever, SparseRetrieverConfig
from .base import RetrievalBackend, RetrievalResult


def _make_sparse_retriever(config: SparseRetrieverConfig) -> SparseRetriever:
    """Thin factory — exists so tests can monkeypatch it."""
    return SparseRetriever(config)


def _row_to_result(row: dict) -> RetrievalResult:
    """Convert a raw retriever row dict into a RetrievalResult."""
    doc = row.get("document", {})
    text: str = doc.get("text") or doc.get("contents") or ""
    # Corpus stores chunks as '"Title"\nBody...' — strip the quoted title prefix.
    if text.startswith('"'):
        parts = text.split("\n", 1)
        text = parts[1] if len(parts) > 1 else text
    return RetrievalResult(
        doc_id=str(doc.get("id", "")),
        title=str(doc.get("title", "")),
        text=text,
        url=doc.get("url"),
        score=float(row.get("score", 0.0)),
    )


class LocalBackend(RetrievalBackend):
    """Backend that retrieves from a local Pyserini index and (in M2) a FAISS index."""

    def __init__(self, sparse_config: SparseRetrieverConfig):
        self._sparse = _make_sparse_retriever(sparse_config)
        self._dense = None  # wired in M2

    def search_sparse(self, query: str, top_k: int) -> list[RetrievalResult]:
        rows = self._sparse.retrieve([query], topk=top_k)
        return [_row_to_result(r) for r in rows[0]]

    def search_dense(self, query: str, top_k: int) -> list[RetrievalResult]:
        raise NotImplementedError("Dense search not configured in M1 LocalBackend")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/retrieval/test_retrieval_backend.py -v
```

Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/backends/local.py tests/unit/retrieval/test_retrieval_backend.py
git commit -m "feat(retrieval): add LocalBackend wrapping SparseRetriever"
```

---

### Task 3: `RetrievalService` with backend selection

**Files:**
- Create: `src/internal/retrieval/service.py`
- Create: `tests/unit/retrieval/test_service.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/retrieval/test_service.py
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

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/retrieval/test_service.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.internal.retrieval.service'`

- [ ] **Step 3: Implement `service.py`**

```python
# src/internal/retrieval/service.py
"""RetrievalService: selects backend from env and exposes search()."""
from __future__ import annotations

import os

from .backends.base import RetrievalBackend, RetrievalResult


def _build_local_backend() -> RetrievalBackend:
    from src.internal.document_index.retrieval import SparseRetrieverConfig
    from .backends.local import LocalBackend

    config = SparseRetrieverConfig(
        index_path=os.environ["BM25_INDEX_PATH"],
        corpus_path=os.environ.get("BM25_CORPUS_PATH", "data/corpus.jsonl"),
        topk=int(os.environ.get("BM25_TOP_K", "20")),
    )
    return LocalBackend(config)


def _build_backend() -> RetrievalBackend:
    name = os.environ.get("RETRIEVAL_BACKEND", "local").lower()
    if name == "local":
        return _build_local_backend()
    raise ValueError(
        f"Unknown RETRIEVAL_BACKEND: {name!r}. Supported values: local"
        " (opensearch and weaviate added in M3)"
    )


class RetrievalService:
    def __init__(self, backend: RetrievalBackend) -> None:
        self._backend = backend

    @classmethod
    def from_env(cls) -> "RetrievalService":
        """Construct service from environment variables."""
        return cls(_build_backend())

    def search(
        self, query: str, top_k: int = 5
    ) -> tuple[list[RetrievalResult], str]:
        """Return (results, retrieval_mode).

        retrieval_mode is 'sparse' in M1; becomes 'hybrid' in M2.
        """
        results = self._backend.search_sparse(query, top_k=top_k)
        return results, "sparse"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/retrieval/test_service.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/service.py tests/unit/retrieval/test_service.py
git commit -m "feat(retrieval): add RetrievalService with env-driven backend selection"
```

---

### Task 4: FastAPI server — `POST /search` and `GET /health`

**Files:**
- Create: `src/internal/servers/retrieval/server.py`
- Create: `tests/unit/servers/retrieval/test_new_server.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/servers/retrieval/test_new_server.py
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

    client.post("/search", json={"query": "vector search", "top_k": 10})
    svc.search.assert_called_once_with("vector search", top_k=10)


def test_search_rejects_empty_query():
    client = TestClient(create_app(_make_service([])))
    resp = client.post("/search", json={"query": "", "top_k": 5})
    assert resp.status_code == 422


def test_search_default_top_k_is_5():
    svc = _make_service([])
    client = TestClient(create_app(svc))

    client.post("/search", json={"query": "anything"})
    svc.search.assert_called_once_with("anything", top_k=5)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/servers/retrieval/test_new_server.py -v
```

Expected: `ImportError` on `src.internal.servers.retrieval.server`

- [ ] **Step 3: Implement `server.py`**

```python
# src/internal/servers/retrieval/server.py
"""FastAPI app wrapping RetrievalService.

Replaces retrieval_server.py in M3. During M1-M2 both run in parallel.
"""
from __future__ import annotations

import os
import time

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.service import RetrievalService


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    filters: dict | None = None


class SearchResultItem(BaseModel):
    doc_id: str
    title: str
    text: str
    url: str | None = None
    score: float
    metadata: dict = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    retrieval_mode: str
    executed_queries: list[str]
    latency_ms: float


def _to_item(r: RetrievalResult) -> SearchResultItem:
    return SearchResultItem(
        doc_id=r.doc_id,
        title=r.title,
        text=r.text,
        url=r.url,
        score=r.score,
        metadata=r.metadata,
    )


def create_app(service: RetrievalService | None = None) -> FastAPI:
    _service = service or RetrievalService.from_env()
    _backend_name = os.environ.get("RETRIEVAL_BACKEND", "local")
    app = FastAPI(title="Retrieval Service", version="1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "backend": _backend_name}

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        t0 = time.monotonic()
        results, mode = _service.search(request.query, top_k=request.top_k)
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        return SearchResponse(
            results=[_to_item(r) for r in results],
            retrieval_mode=mode,
            executed_queries=[request.query],
            latency_ms=latency_ms,
        )

    return app
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/servers/retrieval/test_new_server.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/internal/servers/retrieval/server.py tests/unit/servers/retrieval/test_new_server.py
git commit -m "feat(retrieval): add FastAPI server wrapping RetrievalService (POST /search, GET /health)"
```

---

### Task 5: Offline eval metrics — `recall_at_k`, `ndcg_at_k`, `mrr`

**Files:**
- Create: `src/internal/retrieval/eval_metrics.py`
- Create: `tests/unit/retrieval/test_eval_metrics.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/retrieval/test_eval_metrics.py
"""Tests for Recall@K, NDCG@K, and MRR metric functions."""
from __future__ import annotations

import math
import pytest
from src.internal.retrieval.eval_metrics import recall_at_k, ndcg_at_k, mrr


def test_recall_perfect():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0


def test_recall_partial():
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5


def test_recall_zero():
    assert recall_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0


def test_recall_k_cutoff_respected():
    # relevant doc "b" is at rank 3, k=2 so it should not count
    assert recall_at_k(["a", "x", "b"], {"a", "b"}, k=2) == 0.5


def test_recall_empty_relevant():
    assert recall_at_k(["a", "b"], set(), k=5) == 0.0


def test_ndcg_perfect():
    # Single relevant doc at rank 1: DCG = 1/log2(2) = 1.0; IDCG = 1.0
    assert ndcg_at_k(["a"], {"a"}, k=5) == pytest.approx(1.0)


def test_ndcg_relevant_at_rank_2():
    # Relevant doc at rank 2: DCG = 1/log2(3) ≈ 0.631; IDCG = 1/log2(2) = 1.0
    result = ndcg_at_k(["x", "a"], {"a"}, k=5)
    expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert result == pytest.approx(expected)


def test_ndcg_zero():
    assert ndcg_at_k(["x", "y"], {"a"}, k=5) == 0.0


def test_ndcg_empty_relevant():
    assert ndcg_at_k(["a", "b"], set(), k=5) == 0.0


def test_mrr_first_rank():
    assert mrr(["a", "b", "c"], {"a"}) == pytest.approx(1.0)


def test_mrr_second_rank():
    assert mrr(["x", "a", "b"], {"a"}) == pytest.approx(0.5)


def test_mrr_not_found():
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_mrr_empty_relevant():
    assert mrr(["a", "b"], set()) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/retrieval/test_eval_metrics.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.internal.retrieval.eval_metrics'`

- [ ] **Step 3: Implement `eval_metrics.py`**

```python
# src/internal/retrieval/eval_metrics.py
"""Retrieval evaluation metric functions: Recall@K, NDCG@K, MRR."""
from __future__ import annotations

import math


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant documents found in the top-k results."""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Binary relevance: a document is either relevant (1) or not (0).
    """
    if not relevant_ids:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, doc_id in enumerate(retrieved_ids[:k])
        if doc_id in relevant_ids
    )
    ideal_dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank in range(min(len(relevant_ids), k))
    )
    return dcg / ideal_dcg if ideal_dcg > 0.0 else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank (single-query variant; average over queries in eval_runner)."""
    if not relevant_ids:
        return 0.0
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/retrieval/test_eval_metrics.py -v
```

Expected: `12 passed`

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/eval_metrics.py tests/unit/retrieval/test_eval_metrics.py
git commit -m "feat(retrieval): add recall_at_k, ndcg_at_k, mrr metric functions"
```

---

### Task 6: `eval_runner.py` CLI + sample QA pairs + baseline snapshot

**Files:**
- Create: `src/internal/retrieval/eval_runner.py`
- Create: `tests/unit/retrieval/test_eval_runner.py`
- Create: `data/eval/qa_pairs.jsonl`
- Create: `data/eval/baseline_metrics.json`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/retrieval/test_eval_runner.py
"""Tests for eval_runner.run_eval."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.eval_runner import run_eval


def _make_service(doc_ids_per_query: list[list[str]]) -> MagicMock:
    """Stub RetrievalService that returns fixed doc_ids per call (in order)."""
    svc = MagicMock()
    side_effects = [
        ([RetrievalResult(doc_id=d, title="", text="", url=None, score=0.9 - i * 0.1) for i, d in enumerate(ids)], "sparse")
        for ids in doc_ids_per_query
    ]
    svc.search.side_effect = side_effects
    return svc


def _write_qa(qa_pairs: list[dict]) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair) + "\n")
        return f.name


def test_run_eval_perfect_recall():
    qa = [{"query": "q1", "relevant_doc_ids": ["d1"]}]
    path = _write_qa(qa)
    svc = _make_service([["d1", "d2"]])

    metrics = run_eval(path, service=svc, top_k=5)

    assert metrics["recall@5"] == pytest.approx(1.0)
    assert metrics["ndcg@5"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["num_queries"] == 1


def test_run_eval_zero_recall():
    qa = [{"query": "q1", "relevant_doc_ids": ["d1"]}]
    path = _write_qa(qa)
    svc = _make_service([["x", "y"]])

    metrics = run_eval(path, service=svc, top_k=5)

    assert metrics["recall@5"] == 0.0
    assert metrics["mrr"] == 0.0


def test_run_eval_averages_over_queries():
    qa = [
        {"query": "q1", "relevant_doc_ids": ["d1"]},
        {"query": "q2", "relevant_doc_ids": ["d2"]},
    ]
    path = _write_qa(qa)
    # q1: d1 found at rank 1 (recall=1.0); q2: d2 not found (recall=0.0)
    svc = _make_service([["d1"], ["x"]])

    metrics = run_eval(path, service=svc, top_k=5)

    assert metrics["recall@5"] == pytest.approx(0.5)
    assert metrics["num_queries"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/retrieval/test_eval_runner.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.internal.retrieval.eval_runner'`

- [ ] **Step 3: Implement `eval_runner.py`**

```python
# src/internal/retrieval/eval_runner.py
"""CLI and library for offline retrieval evaluation.

Usage:
    python -m src.internal.retrieval.eval_runner \
        --dataset data/eval/qa_pairs.jsonl \
        --top_k 10

QA pairs file format (one JSON object per line):
    {"query": "...", "relevant_doc_ids": ["doc-id-1", "doc-id-2"]}
"""
from __future__ import annotations

import argparse
import json

from .eval_metrics import mrr as mrr_score, ndcg_at_k, recall_at_k
from .service import RetrievalService


def run_eval(
    dataset_path: str,
    *,
    service: RetrievalService | None = None,
    top_k: int = 10,
) -> dict[str, float | int]:
    """Load QA pairs, run retrieval, compute and return averaged metrics."""
    _service = service or RetrievalService.from_env()

    with open(dataset_path) as f:
        qa_pairs = [json.loads(line) for line in f if line.strip()]

    recalls, ndcgs, mrrs = [], [], []
    for item in qa_pairs:
        query: str = item["query"]
        relevant: set[str] = set(item["relevant_doc_ids"])
        results, _ = _service.search(query, top_k=top_k)
        retrieved = [r.doc_id for r in results]
        recalls.append(recall_at_k(retrieved, relevant, top_k))
        ndcgs.append(ndcg_at_k(retrieved, relevant, top_k))
        mrrs.append(mrr_score(retrieved, relevant))

    n = len(qa_pairs)
    return {
        f"recall@{top_k}": sum(recalls) / n if n else 0.0,
        f"ndcg@{top_k}": sum(ndcgs) / n if n else 0.0,
        "mrr": sum(mrrs) / n if n else 0.0,
        "num_queries": n,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline retrieval evaluation")
    parser.add_argument("--dataset", required=True, help="Path to qa_pairs.jsonl")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()
    metrics = run_eval(args.dataset, top_k=args.top_k)
    print(json.dumps(metrics, indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/retrieval/test_eval_runner.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Create sample QA pairs**

```bash
mkdir -p data/eval
```

Write `data/eval/qa_pairs.jsonl` with one object per line (doc IDs from `data/corpus.jsonl`):

```
{"query": "dense retrieval FAISS index", "relevant_doc_ids": ["doc_001"]}
{"query": "BM25 sparse keyword search", "relevant_doc_ids": ["doc_002"]}
{"query": "retrieval augmented generation", "relevant_doc_ids": ["doc_003"]}
{"query": "cross encoder reranking", "relevant_doc_ids": ["doc_004"]}
{"query": "sentence transformers embeddings", "relevant_doc_ids": ["doc_005"]}
```

Write `data/eval/baseline_metrics.json`:

```json
{
  "recall@10": 0.0,
  "ndcg@10": 0.0,
  "mrr": 0.0,
  "_note": "Updated after M1 gate run. Zero until first eval completes."
}
```

- [ ] **Step 6: Commit**

```bash
git add src/internal/retrieval/eval_runner.py \
        tests/unit/retrieval/test_eval_runner.py \
        data/eval/qa_pairs.jsonl \
        data/eval/baseline_metrics.json
git commit -m "feat(retrieval): add eval_runner CLI with Recall/NDCG/MRR, sample QA pairs"
```

---

### Task 7: Full test suite pass + M1 gate verification

**Files:**
- No new files — verify everything wired together.

- [ ] **Step 1: Run all new retrieval unit tests**

```bash
pytest tests/unit/retrieval/ tests/unit/servers/retrieval/test_new_server.py -v
```

Expected: `24+ passed, 0 failed`

- [ ] **Step 2: Run the full unit test suite to check for regressions**

```bash
pytest tests/unit/ -v --tb=short 2>&1 | tail -20
```

Expected: all pre-existing tests still pass.

- [ ] **Step 3: Run linter**

```bash
ruff check src/internal/retrieval/ src/internal/servers/retrieval/server.py --fix && \
ruff format src/internal/retrieval/ src/internal/servers/retrieval/server.py
```

Expected: no errors.

- [ ] **Step 4: Verify M1 gate against real BM25 index (if index available)**

If `data/indexes/` contains a built BM25 index:

```bash
BM25_INDEX_PATH=data/indexes/bm25 \
BM25_CORPUS_PATH=data/corpus.jsonl \
RETRIEVAL_BACKEND=local \
  python -m src.internal.retrieval.eval_runner \
    --dataset data/eval/qa_pairs.jsonl \
    --top_k 10
```

Expected output shape:
```json
{
  "recall@10": 0.XX,
  "ndcg@10": 0.XX,
  "mrr": 0.XX,
  "num_queries": 3
}
```

Gate: `recall@10 >= 0.75`. If below gate, add more QA pairs to `data/eval/qa_pairs.jsonl` that match real documents in the corpus — do not lower the threshold.

- [ ] **Step 5: Update baseline snapshot**

Once gate is met, overwrite `data/eval/baseline_metrics.json` with the actual numbers:

```bash
BM25_INDEX_PATH=data/indexes/bm25 \
BM25_CORPUS_PATH=data/corpus.jsonl \
RETRIEVAL_BACKEND=local \
  python -m src.internal.retrieval.eval_runner \
    --dataset data/eval/qa_pairs.jsonl \
    --top_k 10 > data/eval/baseline_metrics.json

git add data/eval/baseline_metrics.json
git commit -m "chore(eval): record M1 baseline metrics snapshot"
```

- [ ] **Step 6: Final commit and push**

```bash
git push origin HEAD
```

---

### Task 8: Chunking + Chunk Overlap configuration

**PRD reference:** Section 3 — "Document chunking at ≤ 512 tokens, 64-token overlap. Chunk size and overlap are configurable but fixed at index time."

The existing `chunker.py` already implements chunking. This task adds typed configuration constants, env-var wiring, and unit tests that lock the behaviour so future changes don't silently alter chunk boundaries.

**Files:**
- Create: `src/internal/retrieval/chunk_config.py`
- Create: `tests/unit/retrieval/test_chunk_config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/retrieval/test_chunk_config.py
"""Unit tests for ChunkConfig — validates env-var defaults and bounds."""
from __future__ import annotations

import pytest
from src.internal.retrieval.chunk_config import ChunkConfig


def test_defaults():
    cfg = ChunkConfig()
    assert cfg.chunk_size == 512
    assert cfg.chunk_overlap == 64


def test_from_env_reads_vars(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "256")
    monkeypatch.setenv("CHUNK_OVERLAP", "32")
    cfg = ChunkConfig.from_env()
    assert cfg.chunk_size == 256
    assert cfg.chunk_overlap == 32


def test_overlap_must_be_less_than_chunk_size():
    with pytest.raises(ValueError, match="chunk_overlap"):
        ChunkConfig(chunk_size=64, chunk_overlap=64)


def test_chunk_size_must_be_positive():
    with pytest.raises(ValueError, match="chunk_size"):
        ChunkConfig(chunk_size=0)


def test_chunk_overlap_must_be_nonnegative():
    with pytest.raises(ValueError, match="chunk_overlap"):
        ChunkConfig(chunk_size=512, chunk_overlap=-1)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/retrieval/test_chunk_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.internal.retrieval.chunk_config'`

- [ ] **Step 3: Implement `chunk_config.py`**

```python
# src/internal/retrieval/chunk_config.py
"""Typed configuration for document chunking.

Values are fixed at index time — changing them requires a full re-index.
PRD defaults: chunk_size=512 tokens, chunk_overlap=64 tokens.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ChunkConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )

    @classmethod
    def from_env(cls) -> "ChunkConfig":
        return cls(
            chunk_size=int(os.environ.get("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "64")),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/retrieval/test_chunk_config.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/chunk_config.py tests/unit/retrieval/test_chunk_config.py
git commit -m "feat(retrieval): add ChunkConfig with env-var wiring (chunk_size=512, overlap=64)"
```

---

### Task 9: Metadata filtering — thread `filters` from API through service to backend

**PRD reference:** Section 7 API — `"filters": { "source": "confluence" }` in `POST /search`. The field already exists in `SearchRequest` (Task 4) but is never passed to the backend.

**Files:**
- Modify: `src/internal/retrieval/backends/base.py`
- Modify: `src/internal/retrieval/backends/local.py`
- Modify: `src/internal/retrieval/service.py`
- Modify: `src/internal/servers/retrieval/server.py`
- Modify: `tests/unit/retrieval/test_retrieval_backend.py` (append)
- Modify: `tests/unit/retrieval/test_service.py` (append)

- [ ] **Step 1: Append failing tests for backend**

```python
# Append to tests/unit/retrieval/test_retrieval_backend.py

def test_local_backend_search_sparse_filters_by_metadata(monkeypatch):
    import src.internal.retrieval.backends.local as local_mod

    rows = [
        {
            "document": {
                "id": "d1", "title": "T1", "contents": "body",
                "url": None, "source": "confluence",
            },
            "score": 0.9,
        },
        {
            "document": {
                "id": "d2", "title": "T2", "contents": "text",
                "url": None, "source": "sharepoint",
            },
            "score": 0.7,
        },
    ]
    fake = _fake_sparse_retriever(rows)
    monkeypatch.setattr(local_mod, "_make_sparse_retriever", lambda cfg: fake)

    from src.internal.document_index.retrieval import SparseRetrieverConfig
    backend = LocalBackend(SparseRetrieverConfig(index_path="x", corpus_path="y"))
    results = backend.search_sparse("q", top_k=5, filters={"source": "confluence"})

    assert len(results) == 1
    assert results[0].doc_id == "d1"
```

- [ ] **Step 2: Append failing tests for service**

```python
# Append to tests/unit/retrieval/test_service.py

def test_search_passes_filters_to_backend():
    backend = MagicMock()
    backend.search_sparse.return_value = [_make_result("d1")]
    backend.search_dense.side_effect = NotImplementedError
    service = RetrievalService(backend)

    service.search("q", top_k=5, filters={"source": "confluence"})

    backend.search_sparse.assert_called_once_with(
        "q", top_k=10, filters={"source": "confluence"}
    )
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/unit/retrieval/test_retrieval_backend.py -v -k "filters"
pytest tests/unit/retrieval/test_service.py -v -k "filters"
```

Expected: `TypeError` / `AssertionError` — `search_sparse` does not yet accept `filters`

- [ ] **Step 4: Update `base.py` — add `filters` param to ABC**

```python
# src/internal/retrieval/backends/base.py
"""Abstract base for all retrieval backends."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass
class RetrievalResult:
    doc_id: str
    title: str
    text: str
    url: str | None
    score: float
    metadata: dict = field(default_factory=dict)


class RetrievalBackend(abc.ABC):
    @abc.abstractmethod
    def search_sparse(
        self, query: str, top_k: int, filters: dict | None = None
    ) -> list[RetrievalResult]:
        """BM25 keyword search. filters: optional key/value metadata constraints."""

    @abc.abstractmethod
    def search_dense(
        self, query: str, top_k: int, filters: dict | None = None
    ) -> list[RetrievalResult]:
        """ANN vector search. Raise NotImplementedError if not supported."""
```

- [ ] **Step 5: Update `local.py` — add `_apply_filters` and thread `filters`**

Add the helper function and update both `search_sparse` and `search_dense`:

```python
def _apply_filters(
    results: list[RetrievalResult], filters: dict | None
) -> list[RetrievalResult]:
    """Post-hoc metadata filter. Pyserini does not support native filtering."""
    if not filters:
        return results
    return [
        r for r in results
        if all(r.metadata.get(k) == v for k, v in filters.items())
    ]


# In _row_to_result, populate metadata from extra document fields:
def _row_to_result(row: dict) -> RetrievalResult:
    doc = row.get("document", {})
    text: str = doc.get("text") or doc.get("contents") or ""
    if text.startswith('"'):
        parts = text.split("\n", 1)
        text = parts[1] if len(parts) > 1 else text
    # Carry all non-standard keys as metadata so filters can match them.
    known = {"id", "title", "text", "contents", "url"}
    metadata = {k: v for k, v in doc.items() if k not in known}
    return RetrievalResult(
        doc_id=str(doc.get("id", "")),
        title=str(doc.get("title", "")),
        text=text,
        url=doc.get("url"),
        score=float(row.get("score", 0.0)),
        metadata=metadata,
    )


class LocalBackend(RetrievalBackend):
    # __init__ unchanged

    def search_sparse(
        self, query: str, top_k: int, filters: dict | None = None
    ) -> list[RetrievalResult]:
        rows = self._sparse.retrieve([query], topk=top_k)
        results = [_row_to_result(r) for r in rows[0]]
        return _apply_filters(results, filters)

    def search_dense(
        self, query: str, top_k: int, filters: dict | None = None
    ) -> list[RetrievalResult]:
        if self._dense is None:
            raise NotImplementedError("Dense search not configured — set DENSE_MODEL_PATH env var")
        rows = self._dense.retrieve([query], topk=top_k)
        results = [_row_to_result(r) for r in rows[0]]
        return _apply_filters(results, filters)
```

- [ ] **Step 6: Update `service.py` — add `filters` to `search()`**

```python
def search(
    self, query: str, top_k: int = 5, filters: dict | None = None
) -> tuple[list[RetrievalResult], str]:
    """Run sparse and dense legs; fuse with RRF+MMR. filters passed to both legs."""
    over_fetch = top_k * int(os.environ.get("OVER_FETCH_MULTIPLIER", "2"))
    # ... existing fallback logic, but pass filters=filters to each backend call:
    sparse_results = self._backend.search_sparse(query, top_k=over_fetch, filters=filters)
    dense_results = self._backend.search_dense(query, top_k=over_fetch, filters=filters)
    # ... rest unchanged
```

- [ ] **Step 7: Update `server.py` — pass `request.filters` to service**

In the `search()` endpoint handler:

```python
results, mode = _service.search(request.query, top_k=request.top_k, filters=request.filters)
```

- [ ] **Step 8: Run all affected tests**

```bash
pytest tests/unit/retrieval/ tests/unit/servers/retrieval/ -v
```

Expected: all pass

- [ ] **Step 9: Commit**

```bash
git add src/internal/retrieval/backends/base.py \
        src/internal/retrieval/backends/local.py \
        src/internal/retrieval/service.py \
        src/internal/servers/retrieval/server.py \
        tests/unit/retrieval/test_retrieval_backend.py \
        tests/unit/retrieval/test_service.py
git commit -m "feat(retrieval): thread metadata filters from POST /search through to backend"
```

---

## M1 Completion Checklist

- [ ] `pytest tests/unit/retrieval/ tests/unit/servers/retrieval/test_new_server.py` — all pass
- [ ] `ruff check` and `ruff format` — clean
- [ ] `GET /health` returns `{"status": "ok", "backend": "local"}`
- [ ] `POST /search` returns `retrieval_mode`, `executed_queries`, `latency_ms`
- [ ] `POST /search` with `"filters": {"source": "confluence"}` returns only matching docs
- [ ] `python -m src.internal.retrieval.eval_runner --dataset data/eval/qa_pairs.jsonl` runs without error
- [ ] `data/eval/baseline_metrics.json` updated with real numbers (recall@10 ≥ 0.75)
- [ ] No regressions in existing `pytest tests/unit/` suite
- [ ] `CHUNK_SIZE` and `CHUNK_OVERLAP` env vars documented and validated by `ChunkConfig`

**Next:** M2 plan (`2026-06-15-retrieval-m2-dense-hybrid.md`) — dense retrieval, RRF+MMR extraction, hybrid mode, internal eval endpoints, Redis cache.
