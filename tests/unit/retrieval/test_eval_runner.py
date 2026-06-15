"""Tests for eval_runner.run_eval."""

from __future__ import annotations

import json
import tempfile
from unittest.mock import MagicMock

import pytest

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.eval_runner import run_eval


def _make_service(doc_ids_per_query: list[list[str]]) -> MagicMock:
    svc = MagicMock()
    svc.search.side_effect = [
        (
            [
                RetrievalResult(
                    doc_id=d, title="", text="", url=None, score=0.9 - i * 0.1
                )
                for i, d in enumerate(ids)
            ],
            "sparse",
        )
        for ids in doc_ids_per_query
    ]
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
    svc = _make_service([["d1"], ["x"]])

    metrics = run_eval(path, service=svc, top_k=5)

    assert metrics["recall@5"] == pytest.approx(0.5)
    assert metrics["num_queries"] == 2
