"""Unit tests for hybrid retrieval result fusion."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    HybridRetrieverConfig,
    combine_retrieval_results,
    maximal_marginal_relevance,
)
from src.retrieval.dense_retriever import DenseRetrieverConfig
from src.retrieval.sparse_retriever import SparseRetrieverConfig


# ---------------------------------------------------------------------------
# combine_retrieval_results
# ---------------------------------------------------------------------------


def _result(doc_id: str, score: float) -> dict:
    return {"document": {"id": doc_id, "contents": "text"}, "score": score}


def test_combine_empty_sets():
    assert combine_retrieval_results([]) == []
    assert combine_retrieval_results([[], []]) == []


def test_combine_single_set_preserves_order():
    results = [_result("a", 0.9), _result("b", 0.5), _result("c", 0.1)]
    fused = combine_retrieval_results([results])
    # RRF ranks: a=1, b=2, c=3 → scores 1/61, 1/62, 1/63
    assert [r["document"]["id"] for r in fused] == ["a", "b", "c"]


def test_combine_deduplicates_and_accumulates_rrf():
    dense = [_result("a", 0.9), _result("b", 0.5)]
    sparse = [_result("b", 10.0), _result("c", 8.0)]
    fused = combine_retrieval_results([dense, sparse])
    ids = [r["document"]["id"] for r in fused]
    # "b" gets RRF from both rank-2 positions — should beat "c" which appears only once
    assert ids.index("b") < ids.index("c")
    assert set(ids) == {"a", "b", "c"}


def test_combine_higher_rrf_k_reduces_score_gap():
    results_a = [_result("a", 1.0)]
    results_b = [_result("b", 1.0)]
    fused_low_k = combine_retrieval_results([results_a, results_b], rrf_k=1)
    fused_high_k = combine_retrieval_results([results_a, results_b], rrf_k=1000)
    # With rrf_k=1: 1/(1+1)=0.5;  rrf_k=1000: 1/1001 ≈ 0.001 — both scores equal
    # but high-k scores should be smaller
    assert fused_high_k[0]["score"] < fused_low_k[0]["score"]


def test_combine_all_overlap_keeps_best_ranked_doc_on_top():
    # doc "x" is rank 1 in both sets — accumulates highest RRF score
    dense = [_result("x", 0.9), _result("y", 0.3)]
    sparse = [_result("x", 5.0), _result("z", 4.0)]
    fused = combine_retrieval_results([dense, sparse])
    assert fused[0]["document"]["id"] == "x"


# ---------------------------------------------------------------------------
# HybridRetrieverConfig validation
# ---------------------------------------------------------------------------


def _dense_cfg(**kwargs):
    defaults = dict(
        model_path="m", index_path="i", corpus_path="c", retrieval_method="e5"
    )
    return DenseRetrieverConfig(**{**defaults, **kwargs})


def _sparse_cfg(**kwargs):
    defaults = dict(index_path="i", corpus_path="c")
    return SparseRetrieverConfig(**{**defaults, **kwargs})


def test_hybrid_config_requires_sparse_when_alpha_lt_one():
    cfg = HybridRetrieverConfig(dense=_dense_cfg(), sparse=None, hybrid_alpha=0.5)
    with pytest.raises(ValueError, match="sparse config is required"):
        cfg.validate()


def test_hybrid_config_allows_pure_dense_without_sparse():
    cfg = HybridRetrieverConfig(dense=_dense_cfg(), sparse=None, hybrid_alpha=1.0)
    cfg.validate()  # should not raise


def test_hybrid_config_rejects_out_of_range_alpha():
    for alpha in (-0.1, 1.1):
        cfg = HybridRetrieverConfig(dense=_dense_cfg(), hybrid_alpha=alpha)
        with pytest.raises(ValueError, match="hybrid_alpha"):
            cfg.validate()


# ---------------------------------------------------------------------------
# HybridRetriever — mode dispatch (retrievers mocked; no real index needed)
# ---------------------------------------------------------------------------


def _make_hybrid(alpha: float, *, with_sparse: bool = True) -> HybridRetriever:
    """Build a HybridRetriever with patched underlying retrievers."""
    dense_cfg = _dense_cfg()
    sparse_cfg = _sparse_cfg() if with_sparse else None
    cfg = HybridRetrieverConfig(dense=dense_cfg, sparse=sparse_cfg, hybrid_alpha=alpha)

    with (
        patch("src.retrieval.hybrid_retriever.DenseRetriever") as mock_dense_cls,
        patch("src.retrieval.hybrid_retriever.SparseRetriever") as mock_sparse_cls,
    ):
        mock_dense_cls.return_value = MagicMock()
        mock_sparse_cls.return_value = MagicMock()
        retriever = HybridRetriever(cfg)
        retriever._dense = mock_dense_cls.return_value
        retriever._sparse = mock_sparse_cls.return_value if with_sparse else None

    return retriever


def test_pure_dense_only_calls_dense():
    retriever = _make_hybrid(1.0, with_sparse=False)
    retriever._dense.retrieve.return_value = [[_result("a", 0.9)]]

    results = retriever.retrieve(["q"])

    retriever._dense.retrieve.assert_called_once_with(["q"], None)
    assert results[0][0]["document"]["id"] == "a"


def test_pure_sparse_only_calls_sparse():
    retriever = _make_hybrid(0.0)
    retriever._dense = None
    retriever._sparse.retrieve.return_value = [[_result("b", 5.0)]]

    results = retriever.retrieve(["q"])

    retriever._sparse.retrieve.assert_called_once_with(["q"], None)
    assert results[0][0]["document"]["id"] == "b"


def test_hybrid_calls_both_and_fuses():
    retriever = _make_hybrid(0.5)
    retriever._dense.retrieve.return_value = [[_result("a", 0.9), _result("b", 0.4)]]
    retriever._sparse.retrieve.return_value = [[_result("b", 8.0), _result("c", 6.0)]]

    results = retriever.retrieve(["q"])

    retriever._dense.retrieve.assert_called_once()
    retriever._sparse.retrieve.assert_called_once()
    ids = [r["document"]["id"] for r in results[0]]
    assert set(ids) == {"a", "b", "c"}
    # "b" ranks 2nd in both → higher RRF than "c" (only rank 2 in sparse)
    assert ids.index("b") < ids.index("c")


def test_batch_search_without_scores():
    retriever = _make_hybrid(1.0, with_sparse=False)
    retriever._dense.retrieve.return_value = [[_result("a", 0.9)]]

    docs = retriever.batch_search(["q"])

    assert docs == [[{"id": "a", "contents": "text"}]]


def test_batch_search_with_scores():
    retriever = _make_hybrid(1.0, with_sparse=False)
    retriever._dense.retrieve.return_value = [[_result("a", 0.9)]]

    docs, scores = retriever.batch_search(["q"], return_score=True)

    assert docs == [[{"id": "a", "contents": "text"}]]
    assert len(scores[0]) == 1


# ---------------------------------------------------------------------------
# maximal_marginal_relevance
# ---------------------------------------------------------------------------


def test_mmr_lambda_one_preserves_relevance_order():
    """At lambda=1.0 MMR is pure relevance — order matches input."""
    results = [_result("a", 0.9), _result("b", 0.5), _result("c", 0.1)]
    out = maximal_marginal_relevance(results, topk=3, mmr_lambda=1.0)
    assert [r["document"]["id"] for r in out] == ["a", "b", "c"]


def test_mmr_lambda_zero_maximises_diversity():
    """At lambda=0.0 MMR picks items maximally distant from what's already selected."""
    results = [
        {"document": {"id": "src-1-chunk-0", "contents": "text"}, "score": 0.9},
        {"document": {"id": "src-1-chunk-1", "contents": "text"}, "score": 0.8},
        {"document": {"id": "src-2-chunk-0", "contents": "text"}, "score": 0.3},
    ]
    out = maximal_marginal_relevance(results, topk=2, mmr_lambda=0.0)
    ids = [r["document"]["id"] for r in out]
    assert ids[0] == "src-1-chunk-0"
    assert ids[1] == "src-2-chunk-0"


def test_mmr_returns_topk_results():
    results = [_result(str(i), 1.0 / (i + 1)) for i in range(10)]
    out = maximal_marginal_relevance(results, topk=3, mmr_lambda=0.5)
    assert len(out) == 3


def test_mmr_returns_all_when_topk_exceeds_input():
    results = [_result("a", 0.9), _result("b", 0.5)]
    out = maximal_marginal_relevance(results, topk=10, mmr_lambda=0.5)
    assert len(out) == 2


def test_mmr_empty_input_returns_empty():
    assert maximal_marginal_relevance([], topk=5, mmr_lambda=0.5) == []


def test_hybrid_retriever_applies_mmr_when_configured():
    retriever = _make_hybrid(0.5)
    retriever._dense.retrieve.return_value = [
        [_result("a", 0.9), _result("b", 0.8), _result("c", 0.3)]
    ]
    retriever._sparse.retrieve.return_value = [
        [_result("b", 8.0), _result("c", 6.0), _result("d", 2.0)]
    ]
    retriever.config = retriever.config.__class__(
        **{**retriever.config.__dict__, "mmr_lambda": 0.7, "mmr_topk": 2}
    )
    results = retriever.retrieve(["q"])
    assert len(results[0]) == 2
