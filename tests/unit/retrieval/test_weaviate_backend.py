"""Tests for WeaviateBackend — all tests use a monkeypatched client, no live server."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.internal.retrieval.backends.weaviate import WeaviateBackend, _obj_to_result


def _weaviate_obj(
    uuid: str = "uuid-1",
    title: str = "T",
    content: str = "body",
    url: str | None = "https://x.com",
    score: float | None = 0.9,
    distance: float | None = None,
) -> MagicMock:
    obj = MagicMock()
    obj.uuid = uuid
    obj.properties = {"title": title, "content": content, "source_links": url}
    obj.metadata = MagicMock()
    obj.metadata.score = score
    obj.metadata.distance = distance
    return obj


def _make_client(sparse_objs=None, dense_objs=None) -> MagicMock:
    """Return a mock weaviate client whose collection responds to bm25 / near_vector."""
    client = MagicMock()
    collection = MagicMock()
    collection.query.bm25.return_value = MagicMock(objects=sparse_objs or [])
    collection.query.near_vector.return_value = MagicMock(objects=dense_objs or [])
    client.collections.get.return_value = collection
    return client


def test_obj_to_result_maps_fields():
    obj = _weaviate_obj("u1", "Title", "text body", "https://x.com", score=0.7)
    r = _obj_to_result(obj)
    assert r.doc_id == "u1"
    assert r.title == "Title"
    assert r.text == "text body"
    assert r.url == "https://x.com"
    assert r.score == pytest.approx(0.7)


def test_obj_to_result_uses_distance_when_score_is_none():
    obj = _weaviate_obj(score=None, distance=0.3)
    r = _obj_to_result(obj)
    assert r.score == pytest.approx(-0.3)


def test_search_sparse_calls_bm25():
    client = _make_client(sparse_objs=[_weaviate_obj("u1")])
    backend = WeaviateBackend("Documents", client=client)

    results = backend.search_sparse("neural IR", top_k=5)

    assert len(results) == 1
    assert results[0].doc_id == "u1"
    client.collections.get.return_value.query.bm25.assert_called_once_with(
        query="neural IR", limit=5
    )


def test_search_dense_raises_without_embedder():
    client = _make_client()
    backend = WeaviateBackend("Documents", client=client)
    with pytest.raises(NotImplementedError, match="Dense search not configured"):
        backend.search_dense("q", 5)


def test_search_dense_calls_near_vector():
    client = _make_client(dense_objs=[_weaviate_obj("u2")])
    embedder = MagicMock(return_value=[0.4, 0.5])
    backend = WeaviateBackend("Documents", embedder=embedder, client=client)

    results = backend.search_dense("embedding query", top_k=3)

    embedder.assert_called_once_with("embedding query")
    assert results[0].doc_id == "u2"
    client.collections.get.return_value.query.near_vector.assert_called_once_with(
        near_vector=[0.4, 0.5], limit=3
    )


def test_search_sparse_empty():
    client = _make_client()
    backend = WeaviateBackend("Docs", client=client)
    assert backend.search_sparse("q", 5) == []


def _make_fake_collection(objs) -> MagicMock:
    collection = MagicMock()
    collection.query.bm25.return_value = MagicMock(objects=objs)
    collection.query.near_vector.return_value = MagicMock(objects=objs)
    return collection


def test_bm25_passes_filter_object_when_filters_set(monkeypatch):
    import src.internal.retrieval.backends.weaviate as wv_mod

    fake_collection = _make_fake_collection([])
    monkeypatch.setattr(wv_mod, "_make_collection", lambda url, name: fake_collection)
    backend = WeaviateBackend.from_env()
    backend.search_sparse("q", top_k=5, filters={"source": "confluence"})
    kw = fake_collection.query.bm25.call_args[1]
    assert kw.get("filters") is not None


def test_bm25_no_filter_kwarg_when_filters_none(monkeypatch):
    import src.internal.retrieval.backends.weaviate as wv_mod

    fake_collection = _make_fake_collection([])
    monkeypatch.setattr(wv_mod, "_make_collection", lambda url, name: fake_collection)
    backend = WeaviateBackend.from_env()
    backend.search_sparse("q", top_k=5)
    kw = fake_collection.query.bm25.call_args[1]
    assert kw.get("filters") is None
