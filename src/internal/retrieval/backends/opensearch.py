"""OpenSearch backend: BM25 (match) + kNN (content_vector) via opensearch-py."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from .base import RetrievalBackend, RetrievalResult

# Field names match the schema in src/internal/document_index/opensearch/schema.py
_DEFAULT_CONTENT_FIELD = "content"
_DEFAULT_VECTOR_FIELD = "content_vector"
_DEFAULT_DOC_ID_FIELD = "document_id"
_DEFAULT_TITLE_FIELD = "title"
_DEFAULT_URL_FIELD = "source_links"


def _make_client(url: str) -> Any:
    """Thin factory — exists so tests can monkeypatch it."""
    try:
        from opensearchpy import OpenSearch
    except ImportError as exc:
        raise ImportError(
            "OpenSearch backend requires 'opensearch-py'. Install with: pip install opensearch-py"
        ) from exc
    return OpenSearch(url)


def _hit_to_result(
    hit: dict,
    *,
    content_field: str,
    doc_id_field: str,
    title_field: str,
    url_field: str,
) -> RetrievalResult:
    src = hit.get("_source", {})
    return RetrievalResult(
        doc_id=str(src.get(doc_id_field, hit.get("_id", ""))),
        title=str(src.get(title_field, "")),
        text=str(src.get(content_field, "")),
        url=src.get(url_field),
        score=float(hit.get("_score") or 0.0),
    )


def _build_sparse_body(query: str, content_field: str, filters: dict | None) -> dict:
    match_clause = {"match": {content_field: query}}
    if not filters:
        return {"query": match_clause}
    return {
        "query": {
            "bool": {
                "must": [match_clause],
                "filter": [{"term": {k: v}} for k, v in filters.items()],
            }
        }
    }


def _build_dense_body(
    vec: list[float], vector_field: str, top_k: int, filters: dict | None
) -> dict:
    knn_clause: dict = {"vector": vec, "k": top_k}
    if filters:
        knn_clause["filter"] = {
            "bool": {"filter": [{"term": {k: v}} for k, v in filters.items()]}
        }
    return {"query": {"knn": {vector_field: knn_clause}}}


class OpenSearchBackend(RetrievalBackend):
    """Retrieves from an OpenSearch index via BM25 (sparse) and kNN (dense)."""

    def __init__(
        self,
        index_name: str,
        *,
        content_field: str = _DEFAULT_CONTENT_FIELD,
        vector_field: str = _DEFAULT_VECTOR_FIELD,
        doc_id_field: str = _DEFAULT_DOC_ID_FIELD,
        title_field: str = _DEFAULT_TITLE_FIELD,
        url_field: str = _DEFAULT_URL_FIELD,
        embedder: Callable[[str], list[float]] | None = None,
        client: Any = None,
    ) -> None:
        self._index = index_name
        self._content_field = content_field
        self._vector_field = vector_field
        self._doc_id_field = doc_id_field
        self._title_field = title_field
        self._url_field = url_field
        self._embedder = embedder
        self._client = client or _make_client(
            os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
        )

    @classmethod
    def from_env(cls) -> "OpenSearchBackend":
        return cls(
            index_name=os.environ.get("OPENSEARCH_INDEX", "documents"),
            content_field=os.environ.get("OPENSEARCH_CONTENT_FIELD", "content"),
            vector_field=os.environ.get("OPENSEARCH_VECTOR_FIELD", "content_vector"),
            doc_id_field=os.environ.get("OPENSEARCH_DOC_ID_FIELD", "document_id"),
            title_field=os.environ.get("OPENSEARCH_TITLE_FIELD", "title"),
            url_field=os.environ.get("OPENSEARCH_URL_FIELD", "source_links"),
        )

    def _convert(self, hit: dict) -> RetrievalResult:
        return _hit_to_result(
            hit,
            content_field=self._content_field,
            doc_id_field=self._doc_id_field,
            title_field=self._title_field,
            url_field=self._url_field,
        )

    def search_sparse(
        self, query: str, top_k: int, filters: dict | None = None
    ) -> list[RetrievalResult]:
        body = _build_sparse_body(query, self._content_field, filters)
        body["size"] = top_k
        resp = self._client.search(index=self._index, body=body)
        return [self._convert(h) for h in resp["hits"]["hits"]]

    def search_dense(
        self, query: str, top_k: int, filters: dict | None = None
    ) -> list[RetrievalResult]:
        if self._embedder is None:
            raise NotImplementedError(
                "Dense search not configured — provide an embedder or set DENSE_MODEL_PATH"
            )
        vector = self._embedder(query)
        body = _build_dense_body(vector, self._vector_field, top_k, filters)
        body["size"] = top_k
        resp = self._client.search(index=self._index, body=body)
        return [self._convert(h) for h in resp["hits"]["hits"]]
