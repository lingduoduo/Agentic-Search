"""Tests for Redis embedding cache."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.internal.retrieval.embedding_cache import CachedEmbedder, _cache_key


def _fake_base_embedder(vec: list[float]) -> MagicMock:
    m = MagicMock()
    m.encode.return_value = np.array(vec, dtype=np.float32)
    return m


def test_cache_miss_calls_embedder():
    embedder = _fake_base_embedder([0.1, 0.2, 0.3])
    redis = MagicMock()
    redis.get.return_value = None  # cache miss

    cached = CachedEmbedder(embedder, redis_client=redis)
    result = cached.embed("hello")

    embedder.encode.assert_called_once()
    redis.setex.assert_called_once()
    assert result == pytest.approx([0.1, 0.2, 0.3])


def test_cache_hit_skips_embedder():
    embedder = _fake_base_embedder([0.1, 0.2, 0.3])
    redis = MagicMock()
    redis.get.return_value = json.dumps([0.4, 0.5, 0.6]).encode()

    cached = CachedEmbedder(embedder, redis_client=redis)
    result = cached.embed("hello")

    embedder.encode.assert_not_called()
    assert result == pytest.approx([0.4, 0.5, 0.6])


def test_no_redis_passes_through():
    embedder = _fake_base_embedder([0.1, 0.2])
    cached = CachedEmbedder(embedder, redis_client=None)
    result = cached.embed("hello")
    embedder.encode.assert_called_once()
    assert result == pytest.approx([0.1, 0.2])


def test_cache_key_is_deterministic():
    assert _cache_key("hello") == _cache_key("hello")
    assert _cache_key("hello") != _cache_key("world")


def test_cache_key_has_emb_prefix():
    assert _cache_key("anything").startswith("emb:")


def test_cache_miss_sets_ttl_of_one_hour():
    embedder = _fake_base_embedder([0.0])
    redis = MagicMock()
    redis.get.return_value = None

    CachedEmbedder(embedder, redis_client=redis).embed("q")

    args = redis.setex.call_args
    ttl = (
        args[0][1]
        if args[0]
        else args[1].get("time") or args[1].get("ex") or args[0][1]
    )
    assert ttl == 3600


def test_embed_returns_list_not_ndarray():
    embedder = _fake_base_embedder([0.5, 0.6])
    cached = CachedEmbedder(embedder, redis_client=None)
    result = cached.embed("q")
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# EmbeddingBatcher tests
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402

from src.internal.retrieval.embedding_cache import EmbeddingBatcher  # noqa: E402


class _SimpleEmbedder:
    def encode_batch(self, texts):
        return np.zeros((len(texts), 8), dtype="float32")


def test_batcher_returns_vector():
    batcher = EmbeddingBatcher(_SimpleEmbedder(), max_batch=4, wait_ms=1.0)
    result = asyncio.run(batcher.embed("hello world"))
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1


def test_batcher_coalesces_concurrent_calls():
    call_log: list[int] = []

    class TrackingEmbedder:
        def encode_batch(self, texts):
            call_log.append(len(texts))
            return np.zeros((len(texts), 8), dtype="float32")

    batcher = EmbeddingBatcher(TrackingEmbedder(), max_batch=8, wait_ms=20.0)

    async def _run():
        return await asyncio.gather(*[batcher.embed(f"q{i}") for i in range(6)])

    results = asyncio.run(_run())
    assert len(results) == 6
    assert sum(call_log) == 6
    assert max(call_log) > 1  # at least one batch had >1 item


def test_batcher_respects_max_batch():
    class CountingEmbedder:
        def encode_batch(self, texts):
            assert len(texts) <= 4, f"Batch too large: {len(texts)}"
            return np.zeros((len(texts), 8), dtype="float32")

    batcher = EmbeddingBatcher(CountingEmbedder(), max_batch=4, wait_ms=5.0)

    async def _run():
        return await asyncio.gather(*[batcher.embed(f"q{i}") for i in range(8)])

    asyncio.run(_run())
