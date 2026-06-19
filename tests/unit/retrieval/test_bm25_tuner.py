"""Tests for BM25Tuner grid search."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.internal.retrieval.bm25_tuner import BM25Params, BM25Tuner


def _write_qa(path: str, pairs: list[dict]) -> None:
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")


def test_grid_search_returns_bm25params(tmp_path):
    qa = tmp_path / "qa.jsonl"
    _write_qa(str(qa), [{"query": "q1", "relevant_doc_ids": ["d1"]}])

    def factory(k1, b):
        svc = MagicMock()
        svc.search.return_value = ([MagicMock(doc_id="d1")], "hybrid")
        return svc

    tuner = BM25Tuner(factory)
    result = tuner.grid_search(
        str(qa),
        k1_range=[1.0, 1.2],
        b_range=[0.5, 0.75],
    )
    assert isinstance(result, BM25Params)
    assert result.k1 in (1.0, 1.2)
    assert result.b in (0.5, 0.75)


def test_grid_search_picks_best_params(tmp_path):
    qa = tmp_path / "qa.jsonl"
    _write_qa(str(qa), [{"query": "q", "relevant_doc_ids": ["good"]}])

    def factory(k1, b):
        svc = MagicMock()
        if k1 == 0.8 and b == 0.5:
            svc.search.return_value = ([MagicMock(doc_id="good")], "hybrid")
        else:
            svc.search.return_value = ([MagicMock(doc_id="bad")], "hybrid")
        return svc

    tuner = BM25Tuner(factory)
    result = tuner.grid_search(str(qa), k1_range=[0.8, 1.2], b_range=[0.5, 0.75])
    assert result.k1 == 0.8
    assert result.b == 0.5
    assert result.score == pytest.approx(1.0)


def test_grid_search_saves_to_output_path(tmp_path):
    qa = tmp_path / "qa.jsonl"
    out = tmp_path / "params.json"
    _write_qa(str(qa), [{"query": "q", "relevant_doc_ids": ["d1"]}])

    def factory(k1, b):
        svc = MagicMock()
        svc.search.return_value = ([MagicMock(doc_id="d1")], "hybrid")
        return svc

    tuner = BM25Tuner(factory)
    result = tuner.grid_search(
        str(qa), k1_range=[1.2], b_range=[0.75], output_path=str(out)
    )
    assert out.exists()
    saved = json.loads(out.read_text())
    assert saved["k1"] == result.k1
