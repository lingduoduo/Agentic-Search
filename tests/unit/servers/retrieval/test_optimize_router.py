"""Tests for /internal/optimize/* admin endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.servers.retrieval.optimize_router import create_optimize_router


def _make_app() -> TestClient:
    app = FastAPI()
    router = create_optimize_router()
    app.include_router(router)
    return TestClient(app)


def _write_qa(path: Path) -> None:
    path.write_text(json.dumps({"query": "q", "relevant_doc_ids": ["d1"]}) + "\n")


def test_bm25_tune_returns_params(tmp_path):
    qa = tmp_path / "qa.jsonl"
    _write_qa(qa)
    client = _make_app()

    mock_svc = MagicMock()
    mock_svc.search.return_value = ([MagicMock(doc_id="d1")], "hybrid")

    with patch(
        "src.internal.servers.retrieval.optimize_router._make_service",
        return_value=mock_svc,
    ):
        resp = client.post(
            "/internal/optimize/bm25-tune",
            json={"qa_pairs_path": str(qa), "k1_range": [1.2], "b_range": [0.75]},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "k1" in data and "b" in data and "score" in data


def test_fusion_weights_endpoint(tmp_path):
    qa = tmp_path / "qa.jsonl"
    _write_qa(qa)
    client = _make_app()

    mock_svc = MagicMock()
    mock_svc.search.return_value = ([MagicMock(doc_id="d1")], "hybrid")

    with patch(
        "src.internal.servers.retrieval.optimize_router._make_service",
        return_value=mock_svc,
    ):
        resp = client.post(
            "/internal/optimize/fusion-weights",
            json={"qa_pairs_path": str(qa), "w_sparse_range": [0.4, 0.6]},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "w_sparse" in data and "w_dense" in data
    assert abs(data["w_sparse"] + data["w_dense"] - 1.0) < 1e-4
