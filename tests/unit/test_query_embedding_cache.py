"""Unit tests for src.backend.natural_language_processing.query_embedding_cache."""

from __future__ import annotations

import pytest
from shared_configs.enums import EmbeddingProvider

from src.backend.cache.interface import InMemoryCache, set_cache_backend
from src.backend.natural_language_processing.query_embedding_cache import (
    _build_key,
    _safe_pack_or_none,
    _safe_unpack_or_none,
    cache_query_embeddings,
    get_cached_query_embeddings,
    record_cache_skipped,
)


@pytest.fixture(autouse=True)
def fresh_cache():
    """Isolate each test with a clean InMemoryCache."""
    set_cache_backend(InMemoryCache())


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_build_key_stable():
    assert _build_key("hello", 1) == _build_key("hello", 1)


def test_build_key_differs_by_query():
    assert _build_key("hello", 1) != _build_key("world", 1)


def test_build_key_differs_by_settings_id():
    assert _build_key("hello", 1) != _build_key("hello", 2)


def test_pack_roundtrip():
    v = [0.1, 0.2, 0.3]
    result = _safe_unpack_or_none(_safe_pack_or_none(v))
    assert result == pytest.approx(v, abs=1e-6)


def test_pack_single_float():
    v = [42.0]
    assert _safe_unpack_or_none(_safe_pack_or_none(v)) == pytest.approx(v)


def test_unpack_empty_bytes_returns_none():
    assert _safe_unpack_or_none(b"") is None


def test_unpack_misaligned_bytes_returns_none():
    assert _safe_unpack_or_none(b"\x00\x01\x02") is None  # 3 bytes, not multiple of 4


# ---------------------------------------------------------------------------
# get_cached_query_embeddings — miss
# ---------------------------------------------------------------------------


def test_empty_query_list_returns_empty():
    assert get_cached_query_embeddings([], 1, None, 60) == []


def test_all_misses_on_cold_cache():
    results = get_cached_query_embeddings(["a", "b"], 1, None, 60)
    assert results == [None, None]


# ---------------------------------------------------------------------------
# cache_query_embeddings → get_cached_query_embeddings — hit path
# ---------------------------------------------------------------------------


def test_write_then_read_hit():
    embedding = [0.1, 0.2, 0.3]
    cache_query_embeddings(
        ["hello"], [embedding], search_settings_id=7, provider_type=None, ttl_seconds=60
    )
    results = get_cached_query_embeddings(["hello"], 7, None, 60)
    assert results[0] == pytest.approx(embedding, abs=1e-6)


def test_write_then_read_multiple():
    embeddings = [[0.1, 0.2], [0.3, 0.4]]
    queries = ["foo", "bar"]
    cache_query_embeddings(queries, embeddings, 1, None, 60)
    results = get_cached_query_embeddings(queries, 1, None, 60)
    assert results[0] == pytest.approx(embeddings[0], abs=1e-6)
    assert results[1] == pytest.approx(embeddings[1], abs=1e-6)


def test_partial_hit():
    cache_query_embeddings(["cached"], [[0.5, 0.6]], 1, None, 60)
    results = get_cached_query_embeddings(["cached", "missing"], 1, None, 60)
    assert results[0] == pytest.approx([0.5, 0.6], abs=1e-6)
    assert results[1] is None


def test_settings_id_isolates_entries():
    cache_query_embeddings(
        ["q"], [[1.0]], search_settings_id=1, provider_type=None, ttl_seconds=60
    )
    results = get_cached_query_embeddings(
        ["q"], search_settings_id=2, provider_type=None, ttl_seconds=60
    )
    assert results == [None]


def test_provider_type_label_does_not_affect_key():
    """provider_type is observability-only; the same query should hit regardless."""
    cache_query_embeddings(["q"], [[0.1]], 1, EmbeddingProvider.OPENAI, 60)
    results = get_cached_query_embeddings(["q"], 1, EmbeddingProvider.COHERE, 60)
    assert results[0] == pytest.approx([0.1], abs=1e-6)


# ---------------------------------------------------------------------------
# cache_query_embeddings — validation
# ---------------------------------------------------------------------------


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        cache_query_embeddings(["a", "b"], [[0.1]], 1, None, 60)


def test_empty_queries_is_noop():
    cache_query_embeddings([], [], 1, None, 60)  # must not raise


# ---------------------------------------------------------------------------
# Fail-open on backend errors
# ---------------------------------------------------------------------------


def test_get_fails_open_when_backend_raises(monkeypatch):
    from src.backend.cache import interface

    monkeypatch.setattr(interface, "_default_cache", None)

    class BrokenCache(InMemoryCache):
        def get(self, key: str):
            raise RuntimeError("redis down")

    set_cache_backend(BrokenCache())
    results = get_cached_query_embeddings(["q"], 1, None, 60)
    assert results == [None]


def test_set_fails_open_when_backend_raises(monkeypatch):
    class BrokenCache(InMemoryCache):
        def set(self, key, value, ex=None, ttl=None):
            raise RuntimeError("redis down")

    set_cache_backend(BrokenCache())
    cache_query_embeddings(["q"], [[0.1]], 1, None, 60)  # must not raise


# ---------------------------------------------------------------------------
# record_cache_skipped — smoke test
# ---------------------------------------------------------------------------


def test_record_cache_skipped_does_not_raise():
    record_cache_skipped(provider_type=None, count=3)
    record_cache_skipped(provider_type=EmbeddingProvider.OPENAI, count=0)
