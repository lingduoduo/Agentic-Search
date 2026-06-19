"""Tests for Redis-backed result cache."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.result_cache import ResultCache


def _result(doc_id: str) -> RetrievalResult:
    return RetrievalResult(doc_id=doc_id, title="t", text="c", url=None, score=1.0)


def _make_redis() -> MagicMock:
    store: dict = {}
    redis = MagicMock()
    redis.get.side_effect = lambda k: store.get(k)
    redis.setex.side_effect = lambda k, ttl, v: store.update({k: v})
    return redis


def test_cache_miss_returns_none():
    cache = ResultCache(MagicMock(get=lambda k: None))
    assert cache.get("query", None, 5) is None


def test_cache_hit_returns_results():
    redis = _make_redis()
    cache = ResultCache(redis, ttl_seconds=60)
    results = [_result("d1"), _result("d2")]
    cache.set("q", None, 5, results)
    retrieved = cache.get("q", None, 5)
    assert retrieved is not None
    assert [r.doc_id for r in retrieved] == ["d1", "d2"]


def test_cache_key_includes_top_k():
    redis = _make_redis()
    cache = ResultCache(redis, ttl_seconds=60)
    cache.set("q", None, 5, [_result("d1")])
    assert cache.get("q", None, 10) is None


def test_cache_key_normalises_query_case():
    redis = _make_redis()
    cache = ResultCache(redis, ttl_seconds=60)
    cache.set("Hello World", None, 5, [_result("d1")])
    retrieved = cache.get("hello world", None, 5)
    assert retrieved is not None


def test_none_redis_disables_cache():
    cache = ResultCache(None)
    cache.set("q", None, 5, [_result("d1")])  # no-op
    assert cache.get("q", None, 5) is None


def test_stats_tracks_hits_and_misses():
    redis = _make_redis()
    cache = ResultCache(redis, ttl_seconds=60)
    cache.get("q", None, 5)  # miss
    cache.set("q", None, 5, [_result("d1")])
    cache.get("q", None, 5)  # hit
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(0.5)
