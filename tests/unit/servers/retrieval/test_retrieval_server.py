"""Unit tests for the local document-index retrieval server."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.backend.document_index.retrieval import (
    DenseRetriever,
    DenseRetrieverConfig,
    SparseRetrieverConfig,
)
from src.backend.servers.retrieval_server import RetrievalServerConfig, create_app


class _FakeDenseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieve_calls = []
        self.batch_search_calls = []

    def retrieve(self, queries, topk=None):
        self.retrieve_calls.append((list(queries), topk))
        return [
            [
                {
                    "document": {
                        "id": f"doc-{i}",
                        "title": f"Title {i}",
                        "contents": f'"Title {i}"\nBody {query}',
                        "url": f"https://example.com/{i}",
                    },
                    "score": 0.9 - i * 0.1,
                }
                for i in range(2)
            ]
            for query in queries
        ]

    def batch_search(self, queries, num=None, return_score=False):
        self.batch_search_calls.append((list(queries), num, return_score))
        return [
            [
                {
                    "id": f"doc-{i}",
                    "title": f"Title {i}",
                    "contents": f'"Title {i}"\nBody {query}',
                    "url": f"https://example.com/{i}",
                }
                for i in range(2)
            ]
            for query in queries
        ]


class _FakeSparseRetriever(_FakeDenseRetriever):
    pass


def _server_config() -> RetrievalServerConfig:
    return RetrievalServerConfig(
        retriever=DenseRetrieverConfig(
            model_path="/fake/model",
            index_path="/fake/index.faiss",
            corpus_path="/fake/corpus.jsonl",
            retrieval_method="e5",
            topk=5,
        )
    )


def _bm25_server_config() -> RetrievalServerConfig:
    return RetrievalServerConfig(
        retriever=SparseRetrieverConfig(
            index_path="/fake/bm25",
            corpus_path="/fake/corpus.jsonl",
            retrieval_method="bm25",
            topk=5,
        )
    )


def test_retrieve_single_query_returns_trainer_friendly_shape(monkeypatch):
    monkeypatch.setattr(
        "src.backend.document_index.retrieval.DenseRetriever",
        _FakeDenseRetriever,
    )
    client = TestClient(create_app(_server_config()))

    response = client.post("/retrieve", json={"query": "what is faiss", "top_k": 3})

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "what is faiss"
    assert data["top_k"] == 3
    assert len(data["results"]) == 2
    assert data["results"][0]["doc_id"] == "doc-0"
    assert data["results"][0]["title"] == "Title 0"
    assert "Body what is faiss" in data["results"][0]["text"]
    assert data["result"][0][0]["title"] == "Title 0"


def test_retrieve_batch_queries_keeps_legacy_result_shape(monkeypatch):
    retrievers = []

    def _factory(config):
        retriever = _FakeDenseRetriever(config)
        retrievers.append(retriever)
        return retriever

    monkeypatch.setattr(
        "src.backend.document_index.retrieval.DenseRetriever",
        _factory,
    )
    client = TestClient(create_app(_server_config()))

    response = client.post("/retrieve", json={"queries": ["q1", "q2"], "topk": 4})

    assert response.status_code == 200
    data = response.json()
    assert data["queries"] == ["q1", "q2"]
    assert data["top_k"] == 4
    assert len(data["results"]) == 2
    assert len(data["result"]) == 2
    assert data["result"][1][1]["title"] == "Title 1"
    assert retrievers[0].retrieve_calls == [(["q1", "q2"], 4)]


def test_retrieve_single_query_with_scores_preserves_score_information(monkeypatch):
    retrievers = []

    def _factory(config):
        retriever = _FakeDenseRetriever(config)
        retrievers.append(retriever)
        return retriever

    monkeypatch.setattr(
        "src.backend.document_index.retrieval.DenseRetriever",
        _factory,
    )
    client = TestClient(create_app(_server_config()))

    response = client.post(
        "/retrieve",
        json={"query": "faiss index", "top_k": 2, "return_scores": True},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["results"][0]["score"] == 0.9
    assert data["result"][0][0]["score"] == 0.9
    assert retrievers[0].retrieve_calls == [(["faiss index"], 2)]


def test_bm25_config_uses_sparse_retriever(monkeypatch):
    dense_calls = []
    sparse_calls = []

    def _dense_factory(config):
        dense_calls.append(config)
        return _FakeDenseRetriever(config)

    def _sparse_factory(config):
        sparse_calls.append(config)
        return _FakeSparseRetriever(config)

    monkeypatch.setattr(
        "src.backend.document_index.retrieval.DenseRetriever", _dense_factory
    )
    monkeypatch.setattr(
        "src.backend.document_index.retrieval.SparseRetriever", _sparse_factory
    )

    client = TestClient(create_app(_bm25_server_config()))
    response = client.post("/retrieve", json={"queries": ["bm25 query"], "topk": 2})

    assert response.status_code == 200
    assert dense_calls == []
    assert len(sparse_calls) == 1
    assert response.json()["result"][0][0]["title"] == "Title 0"


def test_dense_retriever_config_defaults_device_to_cpu():
    """CPU default is the whole point: retrieval service must not steal trainer VRAM."""
    cfg = DenseRetrieverConfig(
        model_path="/m",
        index_path="/i",
        corpus_path="/c",
        retrieval_method="e5",
    )
    assert cfg.device == "cpu"
    assert cfg.query_batch_size == 128
    assert cfg.faiss_gpu is False


def test_dense_retriever_config_rejects_zero_query_batch_size():
    from src.backend.document_index.retrieval import DenseRetrieverConfig

    cfg = DenseRetrieverConfig(
        model_path="/m",
        index_path="/i",
        corpus_path="/c",
        retrieval_method="e5",
        query_batch_size=0,
    )

    import pytest

    with pytest.raises(ValueError, match="query_batch_size"):
        cfg.validate()


def test_dense_retriever_config_rejects_zero_hnsw_ef_search():
    from src.backend.document_index.retrieval import DenseRetrieverConfig

    cfg = DenseRetrieverConfig(
        model_path="/m",
        index_path="/i",
        corpus_path="/c",
        retrieval_method="e5",
        hnsw_ef_search=0,
    )

    import pytest

    with pytest.raises(ValueError, match="hnsw_ef_search"):
        cfg.validate()


def test_dense_retriever_retrieve_batches_queries_and_preserves_empty_rows():
    import numpy as np

    class _FakeIndex:
        def search(self, embeddings, k):
            return (
                np.ones((embeddings.shape[0], k), dtype=np.float32),
                np.zeros((embeddings.shape[0], k), dtype=np.int64),
            )

    retriever = DenseRetriever.__new__(DenseRetriever)
    retriever.config = SimpleNamespace(
        topk=1,
        query_batch_size=2,
        retrieval_method="contriever",
    )
    retriever.index = _FakeIndex()
    retriever.corpus = [{"id": "doc-0", "contents": "body"}]
    calls = []

    def _encode_queries(queries):
        calls.append(list(queries))
        return np.ones((len(queries), 2), dtype=np.float32)

    retriever.encode_queries = _encode_queries

    rows = retriever.retrieve(["alpha", " ", "beta", "gamma"], topk=1)

    assert calls == [["alpha", "beta"], ["gamma"]]
    assert rows[1] == []
    assert rows[0][0]["document"]["id"] == "doc-0"
    assert rows[2][0]["score"] == 1.0


def test_dense_retriever_deduplicates_queries_within_batch():
    import numpy as np

    class _FakeIndex:
        def search(self, embeddings, k):
            return (
                np.ones((embeddings.shape[0], k), dtype=np.float32),
                np.zeros((embeddings.shape[0], k), dtype=np.int64),
            )

    retriever = DenseRetriever.__new__(DenseRetriever)
    retriever.config = SimpleNamespace(
        topk=1,
        query_batch_size=3,
        retrieval_method="contriever",
    )
    retriever.index = _FakeIndex()
    retriever.corpus = [{"id": "doc-0", "contents": "body"}]
    calls = []

    def _encode_queries(queries):
        calls.append(list(queries))
        return np.ones((len(queries), 2), dtype=np.float32)

    retriever.encode_queries = _encode_queries

    rows = retriever.retrieve(["alpha", " ", "alpha", "gamma"], topk=1)

    assert calls == [["alpha", "gamma"]]
    assert rows[1] == []
    assert len(rows[0]) == len(rows[2]) == len(rows[3]) == 1


def test_dense_retriever_config_for_e5_base_v2_sets_e5_method_and_cpu():
    from src.backend.document_index.retrieval import DenseRetrieverConfig

    cfg = DenseRetrieverConfig.for_e5_base_v2(index_path="/idx", corpus_path="/corpus")
    assert cfg.retrieval_method == "e5"
    assert cfg.model_path == "intfloat/e5-base-v2"
    assert cfg.device == "cpu"


def test_dense_retriever_config_for_e5_base_v2_accepts_custom_device():
    from src.backend.document_index.retrieval import DenseRetrieverConfig

    cfg = DenseRetrieverConfig.for_e5_base_v2(
        index_path="/idx", corpus_path="/corpus", device="cuda:1"
    )
    assert cfg.device == "cuda:1"


def test_parse_args_device_defaults_to_cpu():
    """Ensure the CLI default keeps retrieval on CPU even without explicit flag."""
    import sys
    from src.backend.servers.retrieval_server import parse_args

    saved = sys.argv
    sys.argv = [
        "retrieval_server",
        "--index_path",
        "/idx",
        "--corpus_path",
        "/corpus",
        "--retrieval_method",
        "e5",
        "--model_path",
        "intfloat/e5-base-v2",
    ]
    try:
        args = parse_args()
    finally:
        sys.argv = saved

    assert args.device == "cpu"
    assert args.workers == 1
    assert args.query_batch_size == 128
    assert args.faiss_gpu is False
    assert args.hnsw_ef_search is None


def test_parse_args_allows_bm25_without_model_path():
    """BM25 retrieval should not require a dense embedding model."""
    import sys
    from src.backend.servers.retrieval_server import parse_args

    saved = sys.argv
    sys.argv = [
        "retrieval_server",
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

    assert args.model_path is None
    assert args.retrieval_method == "bm25"


def test_health_endpoint_returns_ok(monkeypatch):
    monkeypatch.setattr(
        "src.backend.document_index.retrieval.DenseRetriever", _FakeDenseRetriever
    )
    client = TestClient(create_app(_server_config()))
    assert client.get("/health").json() == {"status": "ok"}
