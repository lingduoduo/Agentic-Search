"""Tests for RetrievalResult and RetrievalBackend ABC."""

from __future__ import annotations

import pytest

from src.internal.retrieval.backends.base import RetrievalBackend, RetrievalResult


def test_retrieval_result_defaults():
    r = RetrievalResult(doc_id="d1", title="T", text="body", url=None, score=0.5)
    assert r.metadata == {}


def test_retrieval_result_stores_fields():
    r = RetrievalResult(
        doc_id="d1",
        title="Title",
        text="body",
        url="https://x.com",
        score=0.9,
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
