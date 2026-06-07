"""Unit tests for the Redis embedding cache."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from src.backend.document_index.embedding_cache import (
    EmbeddingCache,
    OpenAIEmbedder,
    _pack,
    _unpack,
)


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------


def test_pack_unpack_roundtrip():
    arr = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    assert np.allclose(_unpack(_pack(arr)), arr)


def test_pack_unpack_1d():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert np.allclose(_unpack(_pack(arr)), arr)


# ---------------------------------------------------------------------------
# EmbeddingCache — Redis unavailable (graceful degradation)
# ---------------------------------------------------------------------------


def _make_unavailable_cache() -> EmbeddingCache:
    with patch("redis.Redis.from_url", side_effect=ConnectionRefusedError("no redis")):
        return EmbeddingCache(redis_url="redis://localhost:6379/0")


def test_cache_get_returns_none_when_unavailable():
    cache = _make_unavailable_cache()
    assert cache.get("any query") is None


def test_cache_set_is_noop_when_unavailable():
    cache = _make_unavailable_cache()
    cache.set("any query", np.zeros(4, dtype=np.float32))  # must not raise


def test_cache_get_batch_returns_all_none_when_unavailable():
    cache = _make_unavailable_cache()
    result = cache.get_batch(["a", "b", "c"])
    assert result == [None, None, None]


# ---------------------------------------------------------------------------
# EmbeddingCache — Redis available (happy path)
# ---------------------------------------------------------------------------


def _make_cache_with_mock_redis() -> tuple[EmbeddingCache, MagicMock]:
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True

    with patch("redis.Redis.from_url", return_value=mock_redis):
        cache = EmbeddingCache(redis_url="redis://localhost:6379/0")

    return cache, mock_redis


def test_cache_miss_returns_none():
    cache, mock_redis = _make_cache_with_mock_redis()
    mock_redis.get.return_value = None
    assert cache.get("unseen query") is None


def test_cache_hit_returns_embedding():
    cache, mock_redis = _make_cache_with_mock_redis()
    arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    mock_redis.get.return_value = _pack(arr)
    result = cache.get("some query")
    assert result is not None
    assert np.allclose(result, arr)


def test_cache_set_calls_setex():
    cache, mock_redis = _make_cache_with_mock_redis()
    arr = np.array([0.5, 0.6], dtype=np.float32)
    cache.set("my query", arr)
    mock_redis.setex.assert_called_once()
    key, ttl, data = mock_redis.setex.call_args[0]
    assert key.startswith("emb:")
    assert ttl == cache.ttl_seconds
    assert np.allclose(_unpack(data), arr)


def test_cache_key_is_text_based():
    cache, _ = _make_cache_with_mock_redis()
    assert cache._key("hello") == cache._key("hello")
    assert cache._key("hello") != cache._key("world")


def test_get_batch_uses_mget():
    cache, mock_redis = _make_cache_with_mock_redis()
    arr = np.array([1.0, 2.0], dtype=np.float32)
    mock_redis.mget.return_value = [_pack(arr), None]
    result = cache.get_batch(["hit", "miss"])
    assert result[0] is not None
    assert np.allclose(result[0], arr)
    assert result[1] is None


def test_set_batch_uses_pipeline():
    cache, mock_redis = _make_cache_with_mock_redis()
    pipe = MagicMock()
    mock_redis.pipeline.return_value = pipe
    embeddings = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    cache.set_batch(["a", "b"], embeddings)
    assert pipe.setex.call_count == 2
    pipe.execute.assert_called_once()


# ---------------------------------------------------------------------------
# OpenAIEmbedder
# ---------------------------------------------------------------------------


def _fake_openai_response(texts: list[str], dim: int = 4) -> MagicMock:
    """Build a mock openai.embeddings.create() response."""
    response = MagicMock()
    response.data = [
        MagicMock(index=i, embedding=[float(i) * 0.1] * dim) for i in range(len(texts))
    ]
    return response


def test_openai_embedder_calls_api_on_cache_miss():
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _fake_openai_response(["hello"])

    with patch("openai.OpenAI", return_value=mock_client):
        embedder = OpenAIEmbedder(model="text-embedding-3-small")
        result = embedder.embed(["hello"])

    assert result.shape == (1, 4)
    mock_client.embeddings.create.assert_called_once()


def test_openai_embedder_uses_redis_cache_on_hit():
    cached_vec = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    mock_redis.mget.return_value = [_pack(cached_vec)]

    with patch("redis.Redis.from_url", return_value=mock_redis):
        with patch("openai.OpenAI") as mock_openai_cls:
            embedder = OpenAIEmbedder(
                model="text-embedding-3-small",
                redis_url="redis://localhost:6379/0",
            )
            result = embedder.embed(["hello"])

    assert np.allclose(result[0], cached_vec)
    mock_openai_cls.assert_not_called()  # OpenAI never instantiated


def test_openai_embedder_writes_misses_to_redis():
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    mock_redis.mget.return_value = [None]  # cache miss
    pipe = MagicMock()
    mock_redis.pipeline.return_value = pipe

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _fake_openai_response(["hello"])

    with patch("redis.Redis.from_url", return_value=mock_redis):
        with patch("openai.OpenAI", return_value=mock_client):
            embedder = OpenAIEmbedder(
                model="text-embedding-3-small",
                redis_url="redis://localhost:6379/0",
            )
            embedder.embed(["hello"])

    pipe.setex.assert_called_once()  # new embedding written to Redis
    pipe.execute.assert_called_once()


def test_openai_embedder_partial_cache_hit():
    """Only misses should reach OpenAI; hits come from Redis."""
    hit_vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    mock_redis.mget.return_value = [_pack(hit_vec), None]  # "a" hits, "b" misses
    pipe = MagicMock()
    mock_redis.pipeline.return_value = pipe

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _fake_openai_response(["b"])

    with patch("redis.Redis.from_url", return_value=mock_redis):
        with patch("openai.OpenAI", return_value=mock_client):
            embedder = OpenAIEmbedder(
                model="text-embedding-3-small",
                redis_url="redis://localhost:6379/0",
            )
            result = embedder.embed(["a", "b"])

    assert result.shape == (2, 4)
    assert np.allclose(result[0], hit_vec)
    # OpenAI called with only the miss
    call_args = mock_client.embeddings.create.call_args
    assert call_args[1]["input"] == ["b"]


def test_openai_embedder_empty_input():
    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    result = embedder.embed([])
    assert result.shape[0] == 0
