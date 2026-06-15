"""Tests for RetrievalService."""

from __future__ import annotations

import logging

import pytest
from unittest.mock import MagicMock

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.service import RetrievalService


def _make_result(doc_id: str, score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(doc_id=doc_id, title="T", text="body", url=None, score=score)


def _sparse_only_backend(results):
    """Backend where dense raises NotImplementedError (no dense configured)."""
    backend = MagicMock()
    backend.search_sparse.return_value = results
    backend.search_dense.side_effect = NotImplementedError
    return backend


def test_search_delegates_to_backend_sparse():
    backend = _sparse_only_backend([_make_result("d1")])
    service = RetrievalService(backend)

    results, mode = service.search("procurement", top_k=5)

    # over_fetch multiplier=2 → top_k * 2 = 10
    backend.search_sparse.assert_called_once_with("procurement", top_k=10)
    assert mode == "sparse_only"
    assert len(results) == 1
    assert results[0].doc_id == "d1"


def test_search_returns_empty_list_on_no_results():
    backend = _sparse_only_backend([])
    service = RetrievalService(backend)

    results, mode = service.search("nothing", top_k=5)

    assert results == []
    assert mode == "sparse_only"


def test_from_env_raises_on_unknown_backend(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_BACKEND", "pinecone")
    with pytest.raises(ValueError, match="Unknown RETRIEVAL_BACKEND"):
        RetrievalService.from_env()


def test_from_env_local_requires_index_path(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_BACKEND", "local")
    monkeypatch.delenv("BM25_INDEX_PATH", raising=False)
    with pytest.raises(KeyError):
        RetrievalService.from_env()


def test_search_hybrid_when_both_legs_succeed():
    backend = MagicMock()
    backend.search_sparse.return_value = [
        _make_result("s1", 0.9),
        _make_result("s2", 0.7),
    ]
    backend.search_dense.return_value = [
        _make_result("d1", 0.8),
        _make_result("s1", 0.6),
    ]
    service = RetrievalService(backend)

    results, mode = service.search("q", top_k=3)

    assert mode == "hybrid"
    # s1 appears in both sets — highest RRF score
    assert results[0].doc_id == "s1"


def test_search_falls_back_to_sparse_when_dense_raises_not_implemented():
    backend = _sparse_only_backend([_make_result("s1")])
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
    backend.search_sparse.side_effect = RuntimeError("sparse down")
    backend.search_dense.side_effect = RuntimeError("dense down")
    service = RetrievalService(backend)

    with pytest.raises(RuntimeError, match="Both retrieval legs failed"):
        service.search("q", top_k=5)
