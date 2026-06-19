from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.cached_reranker import CachedReranker


def _result(doc_id: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(doc_id=doc_id, title="t", text="c", url=None, score=score)


def _make_redis():
    store: dict = {}
    redis = MagicMock()
    redis.get.side_effect = lambda k: store.get(k)
    redis.setex.side_effect = lambda k, ttl, v: store.update({k: v})
    return redis


def test_cache_miss_calls_base():
    base = MagicMock()
    base.rerank.return_value = [_result("d1")]
    cr = CachedReranker(base, _make_redis(), ttl_seconds=60)
    cr.rerank("q", [_result("d1")], top_k=1)
    base.rerank.assert_called_once()


def test_cache_hit_skips_base():
    base = MagicMock()
    base.rerank.return_value = [_result("d1")]
    redis = _make_redis()
    cr = CachedReranker(base, redis, ttl_seconds=60)
    # Populate cache on first call
    cr.rerank("q", [_result("d1")], top_k=1)
    # Second call — base should NOT be called again
    result = cr.rerank("q", [_result("d1")], top_k=1)
    assert base.rerank.call_count == 1
    assert result[0].doc_id == "d1"


def test_cache_key_includes_doc_order_invariant():
    base = MagicMock()
    base.rerank.return_value = [_result("d1")]
    redis = _make_redis()
    cr = CachedReranker(base, redis, ttl_seconds=60)
    cr.rerank("q", [_result("a"), _result("b")], top_k=1)
    # Reversed order of inputs — same doc_ids, should hit cache
    cr.rerank("q", [_result("b"), _result("a")], top_k=1)
    assert base.rerank.call_count == 1


def test_cache_key_normalises_query_case():
    base = MagicMock()
    base.rerank.return_value = [_result("d1")]
    redis = _make_redis()
    cr = CachedReranker(base, redis, ttl_seconds=60)
    cr.rerank("Hello World", [_result("d1")], top_k=1)
    cr.rerank("hello world", [_result("d1")], top_k=1)
    assert base.rerank.call_count == 1


def test_none_redis_disables_cache():
    base = MagicMock()
    base.rerank.return_value = [_result("d1")]
    cr = CachedReranker(base, None, ttl_seconds=60)
    cr.rerank("q", [_result("d1")], top_k=1)
    cr.rerank("q", [_result("d1")], top_k=1)
    assert base.rerank.call_count == 2


def test_stats_tracks_hits_and_misses():
    base = MagicMock()
    base.rerank.return_value = [_result("d1")]
    redis = _make_redis()
    cr = CachedReranker(base, redis, ttl_seconds=60)
    cr.rerank("q", [_result("d1")], top_k=1)  # miss
    cr.rerank("q", [_result("d1")], top_k=1)  # hit
    stats = cr.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(0.5)


def test_from_env_returns_base_when_no_redis_url(monkeypatch):
    monkeypatch.delenv("RERANKER_CACHE_REDIS_URL", raising=False)
    base = MagicMock()
    result = CachedReranker.from_env(base)
    assert result is base


def test_cache_key_includes_top_k():
    base = MagicMock()
    base.rerank.return_value = [_result("d1")]
    redis = _make_redis()
    cr = CachedReranker(base, redis, ttl_seconds=60)
    cr.rerank("q", [_result("d1")], top_k=5)
    # Different top_k — should NOT hit cache
    result = cr.rerank("q", [_result("d1")], top_k=1)
    assert base.rerank.call_count == 2
    assert result[0].doc_id == "d1"
