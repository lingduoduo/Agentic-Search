"""Tests for OpenSearchIndexClient.msearch."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import src.backend.document_index.opensearch.client as _client_mod
from src.backend.document_index.opensearch.client import OpenSearchIndexClient

_PATCH_TARGET = f"{_client_mod.__name__}.OpenSearch"


def _make_source(doc_id: str, chunk_index: int) -> dict:
    return {
        "document_id": doc_id,
        "chunk_index": chunk_index,
        "blurb": "blurb",
        "content": "content",
        "source_type": "web",
        "semantic_identifier": "test",
        "title": "Test Doc",
        "global_boost": 0,
        "hidden": False,
        "last_updated": None,
        "public": True,
        "access_control_list": [],
        "metadata_list": None,
        "metadata_suffix": "",
        "source_links": None,
        "image_file_id": None,
        "doc_summary": "",
        "chunk_context": "",
        "document_sets": None,
        "user_projects": None,
        "personas": None,
        "primary_owners": None,
        "secondary_owners": None,
        "tenant_id": None,
        "ancestor_hierarchy_node_ids": None,
    }


def test_msearch_issues_single_http_call():
    """msearch must call self._client.msearch once regardless of query count."""
    mock_os_instance = MagicMock()
    mock_os_instance.msearch.return_value = {
        "responses": [
            {
                "hits": {
                    "hits": [
                        {"_id": "a", "_score": 1.0, "_source": _make_source("doc-1", 0)}
                    ]
                }
            },
            {
                "hits": {
                    "hits": [
                        {
                            "_id": "b",
                            "_score": 1.0,
                            "_source": _make_source("doc-2", 0),
                        },
                        {
                            "_id": "c",
                            "_score": 1.0,
                            "_source": _make_source("doc-2", 1),
                        },
                    ]
                }
            },
        ]
    }

    with patch(_PATCH_TARGET, return_value=mock_os_instance):
        client = OpenSearchIndexClient(index_name="test-index")
        queries = [
            {"query": {"term": {"document_id": "doc-1"}}},
            {"query": {"term": {"document_id": "doc-2"}}},
        ]
        result = client.msearch(queries)

    mock_os_instance.msearch.assert_called_once()
    expected_body = [
        {"index": "test-index"},
        {"query": {"term": {"document_id": "doc-1"}}},
        {"index": "test-index"},
        {"query": {"term": {"document_id": "doc-2"}}},
    ]
    mock_os_instance.msearch.assert_called_once_with(body=expected_body)
    assert len(result) == 2
    assert len(result[0]) == 1
    assert len(result[1]) == 2


def test_msearch_raises_on_per_response_error():
    """msearch must raise RuntimeError when a sub-response contains an 'error' key."""
    mock_os_instance = MagicMock()
    mock_os_instance.msearch.return_value = {
        "responses": [
            {
                "error": {
                    "type": "index_not_found_exception",
                    "reason": "no such index",
                },
                "status": 404,
            },
        ]
    }

    with patch(_PATCH_TARGET, return_value=mock_os_instance):
        client = OpenSearchIndexClient(index_name="test-index")
        import pytest

        with pytest.raises(RuntimeError, match="msearch sub-request failed"):
            client.msearch([{"query": {"match_all": {}}}])


def test_msearch_returns_empty_for_empty_queries():
    """msearch with no queries must not hit OpenSearch at all."""
    mock_os_instance = MagicMock()

    with patch(_PATCH_TARGET, return_value=mock_os_instance):
        client = OpenSearchIndexClient(index_name="test-index")
        result = client.msearch([])

    mock_os_instance.msearch.assert_not_called()
    assert result == []
