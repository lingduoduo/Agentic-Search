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
    assert [r.doc_id for r in reranked] == ["a", "b", "c"]


def test_mmr_rerank_penalises_same_source():
    # "chunk-1" and "chunk-2" share prefix "chunk"; "other-1" does not
    results = [
        _r("chunk-1", 0.9),
        _r("chunk-2", 0.8),
        _r("other-1", 0.7),
    ]
    # lambda=0.0 → maximum diversity, so "other-1" beats "chunk-2"
    reranked = mmr_rerank(results, top_k=2, mmr_lambda=0.0)
    ids = [r.doc_id for r in reranked]
    assert "chunk-1" in ids
    assert "other-1" in ids


def test_mmr_rerank_empty():
    assert mmr_rerank([], top_k=5) == []
