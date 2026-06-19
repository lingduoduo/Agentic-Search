"""Tests for Reranker (local + Cohere dispatch)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.reranker import Reranker, RerankerConfig


def _result(doc_id: str, score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(
        doc_id=doc_id,
        title=f"Title {doc_id}",
        text=f"Body of {doc_id}",
        url=None,
        score=score,
    )


# --- Config validation ---


def test_config_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        RerankerConfig(provider="pinecone").validate()


def test_config_requires_api_key_for_cohere():
    with pytest.raises(ValueError, match="api_key"):
        RerankerConfig(provider="cohere", api_key=None).validate()


def test_config_rejects_zero_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        RerankerConfig(provider="local", batch_size=0).validate()


# --- Local provider ---


def test_local_reranker_reorders_by_score():
    fake_reranker = MagicMock()
    # Returns scored results: d2 higher than d1
    fake_reranker.rerank.return_value = [
        [
            {
                "document": {"contents": "Title d1\nBody of d1", "doc_id": "d1"},
                "score": 0.3,
            },
            {
                "document": {"contents": "Title d2\nBody of d2", "doc_id": "d2"},
                "score": 0.9,
            },
        ]
    ]

    with patch(
        "src.internal.retrieval.reranker.SentenceTransformerReranker.load",
        return_value=fake_reranker,
    ):
        ranker = Reranker(RerankerConfig(provider="local"))
        results = ranker.rerank(
            "query", [_result("d1", 0.8), _result("d2", 0.2)], top_k=2
        )

    assert results[0].doc_id == "d2"
    assert results[1].doc_id == "d1"
    assert results[0].score == pytest.approx(0.9)


def test_local_reranker_respects_top_k():
    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = [
        [
            {
                "document": {"contents": "Title d1\nBody of d1", "doc_id": "d1"},
                "score": 0.9,
            },
            {
                "document": {"contents": "Title d2\nBody of d2", "doc_id": "d2"},
                "score": 0.7,
            },
            {
                "document": {"contents": "Title d3\nBody of d3", "doc_id": "d3"},
                "score": 0.5,
            },
        ]
    ]

    with patch(
        "src.internal.retrieval.reranker.SentenceTransformerReranker.load",
        return_value=fake_reranker,
    ):
        ranker = Reranker(RerankerConfig(provider="local"))
        results = ranker.rerank(
            "query",
            [_result("d1"), _result("d2"), _result("d3")],
            top_k=2,
        )

    assert len(results) == 2


def test_local_reranker_empty_results():
    with patch("src.internal.retrieval.reranker.SentenceTransformerReranker.load"):
        ranker = Reranker(RerankerConfig(provider="local"))
        assert ranker.rerank("q", [], top_k=5) == []


# --- Cohere provider ---


def test_cohere_reranker_reorders_by_score():
    async def fake_cohere(query, passages, model, api_key):
        # Return high score for "Body of d2", low for "Body of d1"
        return [0.2, 0.9]  # d1=0.2, d2=0.9 (preserves input order)

    with patch(
        "src.internal.retrieval.reranker.cohere_rerank_api",
        side_effect=fake_cohere,
    ):
        ranker = Reranker(
            RerankerConfig(
                provider="cohere",
                model="rerank-english-v3.0",
                api_key="test-key",
            )
        )
        results = ranker.rerank("q", [_result("d1", 0.8), _result("d2", 0.1)], top_k=2)

    assert results[0].doc_id == "d2"
    assert results[0].score == pytest.approx(0.9)


# --- from_env ---


def test_from_env_returns_none_when_provider_unset(monkeypatch):
    monkeypatch.delenv("RERANKER_PROVIDER", raising=False)
    assert Reranker.from_env() is None


def test_from_env_builds_local_reranker(monkeypatch):
    monkeypatch.setenv("RERANKER_PROVIDER", "local")
    monkeypatch.setenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    with patch("src.internal.retrieval.reranker.SentenceTransformerReranker.load"):
        ranker = Reranker.from_env()
    assert ranker is not None
    assert ranker._config.provider == "local"
    assert ranker._config.model == "BAAI/bge-reranker-base"


def test_from_env_builds_cohere_reranker(monkeypatch):
    monkeypatch.setenv("RERANKER_PROVIDER", "cohere")
    monkeypatch.setenv("RERANKER_MODEL", "rerank-english-v3.0")
    monkeypatch.setenv("COHERE_API_KEY", "ck-test")
    ranker = Reranker.from_env()
    assert ranker is not None
    assert ranker._config.provider == "cohere"
    assert ranker._config.api_key == "ck-test"


def test_from_env_cohere_without_api_key_raises(monkeypatch):
    monkeypatch.setenv("RERANKER_PROVIDER", "cohere")
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="api_key"):
        Reranker.from_env()


# --- _cohere_documents tests ---


def test_cohere_documents_v4_returns_dicts():
    from unittest.mock import patch, MagicMock
    import sys

    fake_cohere = MagicMock()
    fake_cohere.__version__ = "4.0.0"
    with patch.dict(sys.modules, {"cohere": fake_cohere}):
        from src.internal.retrieval.reranker import _cohere_documents

        result = _cohere_documents(["text1", "text2"])
    assert result == [{"text": "text1"}, {"text": "text2"}]


def test_cohere_documents_v3_returns_strings():
    from unittest.mock import patch, MagicMock
    import sys

    fake_cohere = MagicMock()
    fake_cohere.__version__ = "3.9.0"
    with patch.dict(sys.modules, {"cohere": fake_cohere}):
        from src.internal.retrieval.reranker import _cohere_documents

        result = _cohere_documents(["text1", "text2"])
    assert result == ["text1", "text2"]


def test_local_rerank_20_candidates_under_5s():
    import time

    candidates = [_result(f"d{i}", score=float(i)) for i in range(20)]
    fake_local = MagicMock()
    fake_local.rerank.return_value = [
        [{"document": {"doc_id": f"d{i}"}, "score": float(20 - i)} for i in range(20)]
    ]
    cfg = RerankerConfig(provider="local")
    with patch(
        "src.internal.retrieval.reranker.SentenceTransformerReranker.load",
        return_value=fake_local,
    ):
        reranker = Reranker(cfg)
    start = time.monotonic()
    result = reranker.rerank("query", candidates, top_k=10)
    elapsed = time.monotonic() - start
    assert elapsed < 5.0
    assert len(result) == 10
