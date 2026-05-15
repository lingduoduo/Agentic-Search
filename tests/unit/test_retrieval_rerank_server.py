"""Unit tests for the combined retrieval+rerank server."""

from __future__ import annotations

import sys

from fastapi.testclient import TestClient

from src.retrieval.dense_retriever import DenseRetrieverConfig
from src.retrieval.rerank import RerankerConfig
from src.retrieval.servers.retrieval_rerank import RetrievalRerankConfig, create_app
from src.retrieval.sparse_retriever import SparseRetrieverConfig


class _FakeRetriever:
    def __init__(self, config):
        self.config = config
        self.batch_search_calls = []
        self.retrieve_calls = []

    def batch_search(self, query_list, num=None, return_score=False):
        self.batch_search_calls.append((list(query_list), num, return_score))
        return [
            [
                {"id": f"{query}-low", "contents": f'"Low"\n{query} low'},
                {"id": f"{query}-high", "contents": f'"High"\n{query} high'},
            ]
            for query in query_list
        ]

    def retrieve(self, queries, topk=None):
        self.retrieve_calls.append((list(queries), topk))
        return []


class _FakeReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, queries, documents, topk=None):
        self.calls.append((list(queries), documents, topk))
        rows = []
        for row in documents:
            scored = [
                {"document": document, "score": float(index)}
                for index, document in enumerate(row)
            ]
            scored.sort(key=lambda item: item["score"], reverse=True)
            rows.append(scored[:topk] if topk is not None else scored)
        return rows


def _dense_config() -> RetrievalRerankConfig:
    return RetrievalRerankConfig(
        retriever=DenseRetrieverConfig(
            model_path="/model",
            index_path="/index.faiss",
            corpus_path="/corpus.jsonl",
            retrieval_method="e5",
            topk=10,
        ),
        reranker=RerankerConfig(rerank_topk=3),
    )


def _sparse_config() -> RetrievalRerankConfig:
    return RetrievalRerankConfig(
        retriever=SparseRetrieverConfig(
            index_path="/bm25",
            corpus_path="/corpus.jsonl",
            retrieval_method="bm25",
            topk=10,
        ),
        reranker=RerankerConfig(rerank_topk=3),
    )


def test_retrieval_rerank_batches_retrieval_once(monkeypatch):
    retrievers = []
    rerankers = []

    def _retriever_factory(config):
        retriever = _FakeRetriever(config)
        retrievers.append(retriever)
        return retriever

    def _reranker_factory(config):
        reranker = _FakeReranker()
        rerankers.append(reranker)
        return reranker

    monkeypatch.setattr("src.retrieval.DenseRetriever", _retriever_factory)
    monkeypatch.setattr(
        "src.retrieval.servers.retrieval_rerank.get_reranker", _reranker_factory
    )

    client = TestClient(create_app(_dense_config()))
    response = client.post(
        "/retrieve",
        json={"queries": ["alpha", "beta"], "topk_retrieval": 8, "topk_rerank": 1},
    )

    assert response.status_code == 200
    assert retrievers[0].batch_search_calls == [(["alpha", "beta"], 8, False)]
    assert retrievers[0].retrieve_calls == []
    assert rerankers[0].calls[0][0] == ["alpha", "beta"]
    assert len(response.json()["result"]) == 2
    assert response.json()["result"][0] == [
        {
            "document": {"id": "alpha-high", "contents": '"High"\nalpha high'},
            "score": 1.0,
        }
    ]


def test_retrieval_rerank_can_return_rerank_scores(monkeypatch):
    monkeypatch.setattr("src.retrieval.DenseRetriever", _FakeRetriever)
    monkeypatch.setattr(
        "src.retrieval.servers.retrieval_rerank.get_reranker",
        lambda config: _FakeReranker(),
    )

    client = TestClient(create_app(_dense_config()))
    response = client.post("/retrieve", json={"queries": ["alpha"], "topk_rerank": 2})

    assert response.status_code == 200
    result = response.json()["result"][0]
    assert result[0]["document"]["id"] == "alpha-high"
    assert result[0]["score"] == 1.0


def test_retrieval_rerank_supports_bm25_retriever(monkeypatch):
    dense_calls = []
    sparse_calls = []

    def _dense_factory(config):
        dense_calls.append(config)
        return _FakeRetriever(config)

    def _sparse_factory(config):
        sparse_calls.append(config)
        return _FakeRetriever(config)

    monkeypatch.setattr("src.retrieval.DenseRetriever", _dense_factory)
    monkeypatch.setattr("src.retrieval.SparseRetriever", _sparse_factory)
    monkeypatch.setattr(
        "src.retrieval.servers.retrieval_rerank.get_reranker",
        lambda config: _FakeReranker(),
    )

    client = TestClient(create_app(_sparse_config()))
    response = client.post("/retrieve", json={"queries": ["bm25"], "topk_rerank": 1})

    assert response.status_code == 200
    assert dense_calls == []
    assert len(sparse_calls) == 1
    assert response.json()["result"][0][0]["document"]["id"] == "bm25-high"


def test_retrieval_rerank_parse_args_allows_bm25_without_model():
    from src.retrieval.servers.retrieval_rerank import parse_args

    saved = sys.argv
    sys.argv = [
        "retrieval_rerank",
        "--index_path",
        "/idx/bm25",
        "--corpus_path",
        "/corpus",
        "--retrieval_method",
        "bm25",
    ]
    try:
        args = parse_args()
    finally:
        sys.argv = saved

    assert args.retriever_model is None
    assert args.retrieval_method == "bm25"


def test_retrieval_rerank_parse_args_exposes_batch_and_device_flags():
    from src.retrieval.servers.retrieval_rerank import parse_args

    saved = sys.argv
    sys.argv = [
        "retrieval_rerank",
        "--index_path",
        "/idx",
        "--corpus_path",
        "/corpus",
        "--retrieval_method",
        "e5",
        "--retriever_model",
        "intfloat/e5-base-v2",
        "--retrieval_query_batch_size",
        "64",
        "--device",
        "cuda:1",
        "--faiss_gpu",
        "--hnsw_ef_search",
        "128",
    ]
    try:
        args = parse_args()
    finally:
        sys.argv = saved

    assert args.retrieval_query_batch_size == 64
    assert args.device == "cuda:1"
    assert args.faiss_gpu is True
    assert args.hnsw_ef_search == 128
