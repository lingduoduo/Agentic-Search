from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

DOC_A = {
    "id": "a",
    "title": "Alpha",
    "contents": '"Alpha"\nFirst doc.',
    "url": "https://a.com",
}
DOC_B = {
    "id": "b",
    "title": "Beta",
    "contents": '"Beta"\nSecond doc.',
    "url": "https://b.com",
}
DOC_C = {
    "id": "c",
    "title": "Gamma",
    "contents": '"Gamma"\nThird doc.',
    "url": "https://c.com",
}

RERANKED = [
    [
        {"document": DOC_A, "score": 0.95},
        {"document": DOC_C, "score": 0.72},
    ]
]


def _make_app(retriever_results, reranker_results):
    mock_retriever = MagicMock()
    mock_retriever.batch_search.return_value = retriever_results

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = reranker_results

    with (
        patch(
            "src.backend.servers.retrieval.hybrid_rerank.HybridRetriever",
            return_value=mock_retriever,
        ),
        patch(
            "src.backend.servers.retrieval.hybrid_rerank.get_reranker",
            return_value=mock_reranker,
        ),
    ):
        from src.backend.servers.retrieval.hybrid_rerank import (
            HybridRerankConfig,
            create_app,
        )
        from src.backend.document_index.hybrid_retriever import HybridRetrieverConfig
        from src.backend.document_index.retrieval import DenseRetrieverConfig
        from src.backend.servers.retrieval.rerank import RerankerConfig

        config = HybridRerankConfig(
            retriever=HybridRetrieverConfig(
                dense=DenseRetrieverConfig(
                    model_path="intfloat/e5-base-v2",
                    index_path="indexes/e5.index",
                    corpus_path="data/corpus.jsonl",
                    retrieval_method="e5",
                    topk=10,
                ),
                hybrid_alpha=1.0,  # pure dense — no sparse needed
            ),
            reranker=RerankerConfig(rerank_topk=2),
        )
        app = create_app(config)

    return TestClient(app), mock_retriever, mock_reranker


def test_returns_reranked_documents():
    client, mock_retriever, mock_reranker = _make_app([[DOC_A, DOC_B, DOC_C]], RERANKED)
    resp = client.post("/retrieve", json={"queries": ["what is alpha"]})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert len(result) == 1
    assert result[0][0]["document"]["id"] == "a"
    assert result[0][0]["score"] == pytest.approx(0.95)


def test_calls_retriever_with_topk_retrieval():
    client, mock_retriever, mock_reranker = _make_app([[DOC_A, DOC_B]], RERANKED)
    client.post("/retrieve", json={"queries": ["q"], "topk_retrieval": 7})
    mock_retriever.batch_search.assert_called_once_with(
        ["q"], num=7, return_score=False
    )


def test_calls_reranker_with_topk_rerank():
    client, mock_retriever, mock_reranker = _make_app([[DOC_A, DOC_B]], RERANKED)
    client.post("/retrieve", json={"queries": ["q"], "topk_rerank": 1})
    mock_reranker.rerank.assert_called_once_with(["q"], [[DOC_A, DOC_B]], topk=1)


def test_empty_retrieval_returns_empty_results():
    client, _, _ = _make_app([[]], [[]])
    resp = client.post("/retrieve", json={"queries": ["obscure"]})
    assert resp.status_code == 200
    assert resp.json()["result"] == [[]]


def test_retriever_exception_returns_500():
    mock_retriever = MagicMock()
    mock_retriever.batch_search.side_effect = RuntimeError("index not loaded")
    mock_reranker = MagicMock()

    with (
        patch(
            "src.backend.servers.retrieval.hybrid_rerank.HybridRetriever",
            return_value=mock_retriever,
        ),
        patch(
            "src.backend.servers.retrieval.hybrid_rerank.get_reranker",
            return_value=mock_reranker,
        ),
    ):
        from src.backend.servers.retrieval.hybrid_rerank import (
            HybridRerankConfig,
            create_app,
        )
        from src.backend.document_index.hybrid_retriever import HybridRetrieverConfig
        from src.backend.document_index.retrieval import DenseRetrieverConfig
        from src.backend.servers.retrieval.rerank import RerankerConfig

        config = HybridRerankConfig(
            retriever=HybridRetrieverConfig(
                dense=DenseRetrieverConfig(
                    model_path="m",
                    index_path="i",
                    corpus_path="c",
                    retrieval_method="e5",
                    topk=5,
                ),
                hybrid_alpha=1.0,
            ),
            reranker=RerankerConfig(rerank_topk=3),
        )
        client = TestClient(create_app(config))

    resp = client.post("/retrieve", json={"queries": ["q"]})
    assert resp.status_code == 500
