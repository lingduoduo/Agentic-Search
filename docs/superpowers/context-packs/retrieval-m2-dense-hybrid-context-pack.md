# Generated Context Pack

# Retrieval PRD — Milestone 2: Dense Retrieval + Hybrid Fusion

## Sources

- [Plan: 2026-06-15-retrieval-m2-dense-hybrid.md](../plans/2026-06-15-retrieval-m2-dense-hybrid.md)

## Implementation Plan Context

### Task 1: `fusion.py` — `rrf_fuse` and `mmr_rerank`

**Files:**
- Create: `src/internal/retrieval/fusion.py`
- Create: `tests/unit/retrieval/test_fusion.py`

- [ ] **Step 1: Write failing tests**

```python

### tests/unit/retrieval/test_fusion.py

"""Tests for RRF fusion and MMR re-ranking over RetrievalResult objects."""
from __future__ import annotations

import pytest

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.fusion import mmr_rerank, rrf_fuse


def _r(doc_id: str, score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(doc_id=doc_id, title="T", text="body", url=None, score=score)


def test_rrf_fuse_single_set():
    results = rrf_fuse([[_r("a"), _r("b"), _r("c")]])
    ids = [r.doc_id for r in results]
    assert ids == ["a", "b", "c"]


def test_rrf_fuse_two_sets_accumulates_scores():
    # "a" appears in both sets — should outscore "b" (sparse-only) and "c" (dense-only)
    sparse = [_r("a"), _r("b")]
    dense = [_r("a"), _r("c")]
    results = rrf_fuse([sparse, dense])
    assert results[0].doc_id == "a"


def test_rrf_fuse_deduplicates():
    sparse = [_r("a"), _r("b")]
    dense = [_r("a"), _r("b")]
    results = rrf_fuse([sparse, dense])
    assert len(results) == 2


def test_rrf_fuse_empty_sets():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[]]) == []


def test_rrf_fuse_score_is_rrf():
    results = rrf_fuse([[_r("a")]], rrf_k=60)
    assert results[0].score == pytest.approx(1.0 / (60 + 1))


def test_mmr_rerank_top_k_respected():
    results = [_r(f"d{i}", score=1.0 - i * 0.1) for i in range(5)]
    reranked = mmr_rerank(results, top_k=3)
    assert len(reranked) == 3


def test_mmr_rerank_lambda_1_preserves_order():
    results = [_r("a", 0.9), _r("b", 0.7), _r("c", 0.5)]
    reranked = mmr_rerank(results, top_k=3, mmr_lambda=1.0)

_[Section compacted.]_

### Task 2: Add dense leg to `LocalBackend`

**Files:**
- Modify: `src/internal/retrieval/backends/local.py`
- Modify: `tests/unit/retrieval/test_retrieval_backend.py` (append)

- [ ] **Step 1: Append failing tests**

```python

### Append to tests/unit/retrieval/test_retrieval_backend.py

from src.internal.document_index.retrieval import DenseRetrieverConfig  # noqa: E402


def _fake_dense_retriever(rows: list[dict]) -> MagicMock:
    m = MagicMock()
    m.retrieve.return_value = [rows]
    return m


def test_local_backend_search_dense(monkeypatch):
    import src.internal.retrieval.backends.local as local_mod

    dense_rows = [
        {"document": {"id": "d3", "title": "T3", "contents": "dense body", "url": None}, "score": 0.95},
    ]
    monkeypatch.setattr(local_mod, "_make_sparse_retriever", lambda cfg: MagicMock())
    monkeypatch.setattr(local_mod, "_make_dense_retriever", lambda cfg: _fake_dense_retriever(dense_rows))

    backend = LocalBackend(
        SparseRetrieverConfig(index_path="x", corpus_path="y"),
        dense_config=DenseRetrieverConfig.for_e5_base_v2(
            index_path="z", corpus_path="y"
        ),
    )
    results = backend.search_dense("embedding", top_k=5)
    assert len(results) == 1
    assert results[0].doc_id == "d3"
    assert results[0].score == pytest.approx(0.95)


def test_local_backend_search_dense_raises_when_not_configured(monkeypatch):
    import src.internal.retrieval.backends.local as local_mod

    monkeypatch.setattr(local_mod, "_make_sparse_retriever", lambda cfg: MagicMock())

    backend = LocalBackend(SparseRetrieverConfig(index_path="x", corpus_path="y"))
    with pytest.raises(NotImplementedError, match="Dense search not configured"):
        backend.search_dense("q", 5)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/retrieval/test_retrieval_backend.py -v -k "dense"

_[Section compacted.]_

### Task 3: Upgrade `RetrievalService.search()` to hybrid mode

**Files:**
- Modify: `src/internal/retrieval/service.py`
- Modify: `tests/unit/retrieval/test_service.py` (append)

- [ ] **Step 1: Append failing tests**

```python

### Append to tests/unit/retrieval/test_service.py

import logging


def _backend_with_both(sparse_results, dense_results):
    backend = MagicMock()
    backend.search_sparse.return_value = sparse_results
    backend.search_dense.return_value = dense_results
    return backend


def test_search_hybrid_when_both_legs_succeed():
    backend = _backend_with_both(
        [_make_result("s1", 0.9), _make_result("s2", 0.7)],
        [_make_result("d1", 0.8), _make_result("s1", 0.6)],
    )
    service = RetrievalService(backend)
    results, mode = service.search("q", top_k=3)
    assert mode == "hybrid"
    # s1 appears in both sets → highest RRF score
    assert results[0].doc_id == "s1"


def test_search_falls_back_to_sparse_when_dense_raises_not_implemented():
    backend = MagicMock()
    backend.search_sparse.return_value = [_make_result("s1")]
    backend.search_dense.side_effect = NotImplementedError("no dense")
    service = RetrievalService(backend)
    results, mode = service.search("q", top_k=5)
    assert mode == "sparse_only"
    assert results[0].doc_id == "s1"


def test_search_falls_back_to_dense_when_sparse_raises(caplog):
    backend = MagicMock()
    backend.search_sparse.side_effect = RuntimeError("BM25 down")
    backend.search_dense.return_value = [_make_result("d1")]
    service = RetrievalService(backend)
    with caplog.at_level(logging.WARNING):
        results, mode = service.search("q", top_k=5)
    assert mode == "dense_only"
    assert results[0].doc_id == "d1"


def test_search_raises_when_both_legs_fail():
    backend = MagicMock()

_[Section compacted.]_

### Task 4: Internal eval router

**Files:**
- Create: `src/internal/servers/retrieval/eval_router.py`
- Modify: `src/internal/servers/retrieval/server.py` (mount router)
- Create: `tests/unit/servers/retrieval/test_eval_router.py`

- [ ] **Step 1: Write failing tests**

```python

### tests/unit/servers/retrieval/test_eval_router.py

"""Tests for internal eval endpoints."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.service import RetrievalService
from src.internal.servers.retrieval.eval_router import create_eval_router


def _make_backend(sparse=None, dense=None):
    backend = MagicMock()
    backend.search_sparse.return_value = sparse or []
    if dense is None:
        backend.search_dense.side_effect = NotImplementedError
    else:
        backend.search_dense.return_value = dense
    return backend


def _result(doc_id="d1"):
    return RetrievalResult(doc_id=doc_id, title="T", text="b", url=None, score=0.9)


def _app_with_router(backend):
    svc = RetrievalService(backend)
    app = FastAPI()
    # Mount without auth for testing (pass require_admin=False stub)
    app.include_router(create_eval_router(svc, require_admin=None))
    return TestClient(app)


def test_sparse_endpoint_returns_results():
    client = _app_with_router(_make_backend(sparse=[_result("s1")]))
    resp = client.post("/internal/search/sparse", json={"query": "q", "top_k": 5})
    assert resp.status_code == 200
    assert resp.json()["results"][0]["doc_id"] == "s1"
    assert resp.json()["retrieval_mode"] == "sparse"


def test_dense_endpoint_returns_results():
    client = _app_with_router(_make_backend(dense=[_result("d1")]))
    resp = client.post("/internal/search/dense", json={"query": "q", "top_k": 5})

_[Section compacted.]_

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
