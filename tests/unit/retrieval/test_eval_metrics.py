"""Tests for Recall@K, NDCG@K, and MRR metric functions."""

from __future__ import annotations

import math

import pytest

from src.internal.retrieval.eval_metrics import mrr, ndcg_at_k, recall_at_k


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
