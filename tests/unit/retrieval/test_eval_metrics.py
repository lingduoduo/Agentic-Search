"""Tests for Recall@K, NDCG@K, and MRR metric functions."""

from __future__ import annotations

import math

import pytest

from src.internal.retrieval.eval_metrics import (
    hit_rate_at_k,
    map_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reranker_improvement_ratio,
)


def test_recall_perfect():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0


def test_recall_partial():
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5


def test_recall_zero():
    assert recall_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0


def test_recall_k_cutoff_respected():
    assert recall_at_k(["a", "x", "b"], {"a", "b"}, k=2) == 0.5


def test_recall_empty_relevant():
    assert recall_at_k(["a", "b"], set(), k=5) == 0.0


def test_ndcg_perfect():
    assert ndcg_at_k(["a"], {"a"}, k=5) == pytest.approx(1.0)


def test_ndcg_relevant_at_rank_2():
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


def test_precision_at_k_perfect():
    assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == pytest.approx(1.0)


def test_precision_at_k_half():
    assert precision_at_k(["a", "x", "b", "y"], {"a", "b"}, 4) == pytest.approx(0.5)


def test_precision_at_k_empty_relevant():
    assert precision_at_k(["a", "b"], set(), 5) == 0.0


def test_hit_rate_at_1_hit():
    assert hit_rate_at_k(["a", "b"], {"a"}, 1) == 1.0


def test_hit_rate_at_1_miss():
    assert hit_rate_at_k(["b", "c"], {"a"}, 1) == 0.0


def test_hit_rate_at_k_any_relevant():
    assert hit_rate_at_k(["x", "y", "a"], {"a"}, 3) == 1.0


# --- map_at_k and reranker_improvement_ratio tests ---


def test_map_at_k_perfect():
    assert map_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)


def test_map_at_k_partial():
    # Only "a" is relevant, found at rank 1 → AP = 1/1 / 1 = 1.0
    assert map_at_k(["a", "x", "y"], {"a"}, k=3) == pytest.approx(1.0)


def test_map_at_k_second_rank():
    # "a" at rank 2, "b" at rank 1 (not relevant) → precision when "a" found = 1/2
    assert map_at_k(["x", "a"], {"a"}, k=2) == pytest.approx(0.5)


def test_map_at_k_none_found():
    assert map_at_k(["x", "y"], {"a", "b"}, k=2) == pytest.approx(0.0)


def test_map_at_k_empty_relevant():
    assert map_at_k(["a", "b"], set(), k=5) == 0.0


def test_reranker_improvement_positive():
    ratio = reranker_improvement_ratio(pre_ndcg=0.5, post_ndcg=0.6)
    assert ratio == pytest.approx(0.2)


def test_reranker_improvement_negative():
    ratio = reranker_improvement_ratio(pre_ndcg=0.6, post_ndcg=0.5)
    assert ratio == pytest.approx(-1 / 6, rel=1e-4)


def test_reranker_improvement_zero_pre():
    assert reranker_improvement_ratio(pre_ndcg=0.0, post_ndcg=0.5) == 0.0
