"""Unit tests for src.search.retrieval_server."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.search.retrieval import DenseRetrieverConfig
from src.search.retrieval_server import RetrievalServerConfig, create_app


class _FakeDenseRetriever:
    def __init__(self, config):
        self.config = config

    def retrieve(self, queries, topk=None):
        del topk
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
        del num, return_score
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


def test_retrieve_single_query_returns_trainer_friendly_shape(monkeypatch):
    monkeypatch.setattr(
        "src.search.retrieval_server.DenseRetriever",
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
    monkeypatch.setattr(
        "src.search.retrieval_server.DenseRetriever",
        _FakeDenseRetriever,
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


def test_retrieve_single_query_with_scores_preserves_score_information(monkeypatch):
    monkeypatch.setattr(
        "src.search.retrieval_server.DenseRetriever",
        _FakeDenseRetriever,
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


def test_dense_retriever_config_defaults_device_to_cpu():
    """CPU default is the whole point: retrieval service must not steal trainer VRAM."""
    cfg = DenseRetrieverConfig(
        model_path="/m",
        index_path="/i",
        corpus_path="/c",
        retrieval_method="e5",
    )
    assert cfg.device == "cpu"


def test_dense_retriever_config_for_e5_base_v2_sets_e5_method_and_cpu():
    from src.search.retrieval import DenseRetrieverConfig

    cfg = DenseRetrieverConfig.for_e5_base_v2(index_path="/idx", corpus_path="/corpus")
    assert cfg.retrieval_method == "e5"
    assert cfg.model_path == "intfloat/e5-base-v2"
    assert cfg.device == "cpu"


def test_dense_retriever_config_for_e5_base_v2_accepts_custom_device():
    from src.search.retrieval import DenseRetrieverConfig

    cfg = DenseRetrieverConfig.for_e5_base_v2(
        index_path="/idx", corpus_path="/corpus", device="cuda:1"
    )
    assert cfg.device == "cuda:1"


def test_parse_args_device_defaults_to_cpu():
    """Ensure the CLI default keeps retrieval on CPU even without explicit flag."""
    import sys
    from src.search.retrieval_server import parse_args

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


def test_health_endpoint_returns_ok(monkeypatch):
    monkeypatch.setattr(
        "src.search.retrieval_server.DenseRetriever", _FakeDenseRetriever
    )
    client = TestClient(create_app(_server_config()))
    assert client.get("/health").json() == {"status": "ok"}
