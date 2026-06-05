"""Unit tests for the Redis embedding cache."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from src.retrieval.embedding_cache import EmbeddingCache, _pack, _unpack


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
        return EmbeddingCache("model", redis_url="redis://localhost:6379/0")


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
        cache = EmbeddingCache("mymodel", redis_url="redis://localhost:6379/0")

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


def test_cache_key_is_model_scoped():
    cache1, _ = _make_cache_with_mock_redis()
    cache2 = EmbeddingCache.__new__(EmbeddingCache)
    cache2.model_path = "other_model"
    cache2._unavailable = False
    cache2._client = None

    key1 = cache1._key("hello")
    key2 = cache2._key("hello")
    assert key1 != key2


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
