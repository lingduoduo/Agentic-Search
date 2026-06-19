from __future__ import annotations

from unittest.mock import MagicMock

from src.context.query_transform import QueryTransformConfig, QueryTransformPipeline
from src.internal.retrieval.cached_query_transform import CachedQueryTransformPipeline


class FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, k):
        return self.store.get(k)

    def setex(self, k, ttl, v):
        self.store[k] = v


def _fake_llm(text="broad"):
    llm = MagicMock()
    llm.complete.return_value = type("R", (), {"text": text})()
    return llm


def test_second_call_is_cache_hit():
    leaf = QueryTransformPipeline(
        QueryTransformConfig(step_back=True), _fake_llm("broad")
    )
    redis = FakeRedis()
    pipe = CachedQueryTransformPipeline(leaf, redis)
    b1 = pipe.transform("q")
    b2 = pipe.transform("q")
    assert b1.step_back == b2.step_back == "broad"
    assert pipe.stats() == {"hits": 1, "misses": 1, "hit_rate": 0.5}


def test_disabled_without_redis_passes_through():
    leaf = QueryTransformPipeline(QueryTransformConfig(step_back=True), _fake_llm())
    pipe = CachedQueryTransformPipeline(leaf, None)
    assert pipe.transform("q").original == "q"
    assert pipe.stats()["hits"] == 0


def test_cache_hit_remerges_caller_filters():
    from src.context.query_transform import QueryTransformConfig, QueryTransformPipeline

    leaf = QueryTransformPipeline(QueryTransformConfig(), _fake_llm())
    redis = FakeRedis()
    pipe = CachedQueryTransformPipeline(leaf, redis)
    pipe.transform("q")  # populate cache (miss)
    b = pipe.transform("q", filters={"lang": "en"})  # hit with caller filters
    assert b.merged_filters.get("lang") == "en"


def test_caller_filters_not_leaked_across_callers():
    from src.context.query_transform import QueryTransformConfig, QueryTransformPipeline

    leaf = QueryTransformPipeline(QueryTransformConfig(), _fake_llm())
    redis = FakeRedis()
    pipe = CachedQueryTransformPipeline(leaf, redis)
    pipe.transform("q", filters={"source": "web"})  # caller A, miss
    b = pipe.transform("q")  # caller B, hit, no filters
    assert "source" not in b.merged_filters  # A's filters must not leak to B
