"""Tests for OpenSearchBackend — all tests use a monkeypatched client, no live server."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.internal.retrieval.backends.opensearch import OpenSearchBackend, _hit_to_result


def _make_client(hits: list[dict]) -> MagicMock:
    """Return a mock client whose .search() returns the given hit list."""
    client = MagicMock()
    client.search.return_value = {"hits": {"hits": hits}}
    return client


def _hit(
    doc_id: str = "d1", score: float = 0.9, title: str = "T", content: str = "body"
) -> dict:
    return {
        "_id": doc_id,
        "_score": score,
        "_source": {
            "document_id": doc_id,
            "title": title,
            "content": content,
            "source_links": "https://x.com",
        },
    }


def test_hit_to_result_maps_fields():
    r = _hit_to_result(
        _hit("d1", 0.8),
        content_field="content",
        doc_id_field="document_id",
        title_field="title",
        url_field="source_links",
    )
    assert r.doc_id == "d1"
    assert r.score == pytest.approx(0.8)
    assert r.text == "body"
    assert r.url == "https://x.com"


def test_search_sparse_issues_match_query():
    client = _make_client([_hit("d1"), _hit("d2")])
    backend = OpenSearchBackend("my_index", client=client)

    results = backend.search_sparse("neural retrieval", top_k=5)

    assert len(results) == 2
    assert results[0].doc_id == "d1"
    call_body = client.search.call_args.kwargs["body"]
    assert call_body["query"]["match"]["content"] == "neural retrieval"
    assert call_body["size"] == 5


def test_search_sparse_uses_custom_content_field():
    client = _make_client([])
    backend = OpenSearchBackend("idx", content_field="text_body", client=client)
    backend.search_sparse("q", top_k=3)

    call_body = client.search.call_args.kwargs["body"]
    assert "text_body" in call_body["query"]["match"]


def test_search_dense_raises_without_embedder():
    client = _make_client([])
    backend = OpenSearchBackend("idx", client=client)
    with pytest.raises(NotImplementedError, match="Dense search not configured"):
        backend.search_dense("q", 5)


def test_search_dense_issues_knn_query():
    client = _make_client([_hit("d1")])
    embedder = MagicMock(return_value=[0.1, 0.2, 0.3])
    backend = OpenSearchBackend("idx", embedder=embedder, client=client)

    results = backend.search_dense("query text", top_k=4)

    embedder.assert_called_once_with("query text")
    assert len(results) == 1
    call_body = client.search.call_args.kwargs["body"]
    assert "knn" in call_body["query"]
    assert call_body["query"]["knn"]["content_vector"]["vector"] == [0.1, 0.2, 0.3]
    assert call_body["query"]["knn"]["content_vector"]["k"] == 4


def test_search_sparse_empty_results():
    client = _make_client([])
    backend = OpenSearchBackend("idx", client=client)
    assert backend.search_sparse("q", 5) == []
