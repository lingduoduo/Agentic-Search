"""Weaviate backend: BM25 + nearVector via weaviate-client v4."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from .base import RetrievalBackend, RetrievalResult


def _make_weaviate_client(url: str) -> Any:
    """Thin factory — exists so tests can monkeypatch it."""
    try:
        import weaviate
    except ImportError as exc:
        raise ImportError(
            "Weaviate backend requires 'weaviate-client>=4.9'. Install with: pip install 'weaviate-client>=4.9'"
        ) from exc
    parsed = url.replace("http://", "").replace("https://", "")
    host, _, port_str = parsed.partition(":")
    port = int(port_str) if port_str else 8080
    return weaviate.connect_to_custom(
        http_host=host, http_port=port, grpc_host=host, grpc_port=50051
    )


def _make_collection(url: str, collection_name: str) -> Any:
    """Thin factory — exists so tests can monkeypatch it."""
    client = _make_weaviate_client(url)
    return client.collections.get(collection_name)


def _obj_to_result(obj: Any) -> RetrievalResult:
    """Convert a Weaviate v4 result object to RetrievalResult."""
    props = obj.properties if hasattr(obj, "properties") else {}
    meta = obj.metadata if hasattr(obj, "metadata") else None
    score = 0.0
    if meta is not None:
        if getattr(meta, "score", None) is not None:
            score = float(meta.score)
        elif getattr(meta, "distance", None) is not None:
            # nearVector returns distance (lower = better); negate so higher = better
            score = -float(meta.distance)
    return RetrievalResult(
        doc_id=str(getattr(obj, "uuid", "") or props.get("document_id", "")),
        title=str(props.get("title", "")),
        text=str(props.get("content", "")),
        url=props.get("source_links"),
        score=score,
    )


def _weaviate_filter(filters: dict | None) -> Any:
    """Build a Weaviate Filter from a key/value dict, or return None."""
    if not filters:
        return None
    try:
        from weaviate.classes.query import Filter as WvFilter
    except ImportError:
        return None
    clauses = [WvFilter.by_property(k).equal(v) for k, v in filters.items()]
    return clauses[0] if len(clauses) == 1 else WvFilter.all_of(clauses)


class WeaviateBackend(RetrievalBackend):
    """Retrieves from a Weaviate collection via BM25 (sparse) and nearVector (dense)."""

    def __init__(
        self,
        collection_name: str,
        *,
        embedder: Callable[[str], list[float]] | None = None,
        client: Any = None,
        collection: Any = None,
    ) -> None:
        self._collection_name = collection_name
        self._embedder = embedder
        if collection is not None:
            self._collection_obj = collection
        elif client is not None:
            self._collection_obj = client.collections.get(collection_name)
        else:
            url = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
            self._collection_obj = _make_collection(url, collection_name)

    @classmethod
    def from_env(cls) -> "WeaviateBackend":
        return cls(
            collection_name=os.environ.get("WEAVIATE_COLLECTION", "Document"),
        )

    @property
    def _collection(self) -> Any:
        return self._collection_obj

    def search_sparse(
        self, query: str, top_k: int, filters: dict | None = None
    ) -> list[RetrievalResult]:
        kwargs: dict = {"query": query, "limit": top_k}
        wv_filter = _weaviate_filter(filters)
        if wv_filter is not None:
            kwargs["filters"] = wv_filter
        resp = self._collection.query.bm25(**kwargs)
        return [_obj_to_result(obj) for obj in resp.objects]

    def search_dense(
        self, query: str, top_k: int, filters: dict | None = None
    ) -> list[RetrievalResult]:
        if self._embedder is None:
            raise NotImplementedError(
                "Dense search not configured — provide an embedder or set DENSE_MODEL_PATH"
            )
        vector = self._embedder(query)
        kwargs: dict = {"near_vector": vector, "limit": top_k}
        wv_filter = _weaviate_filter(filters)
        if wv_filter is not None:
            kwargs["filters"] = wv_filter
        resp = self._collection.query.near_vector(**kwargs)
        return [_obj_to_result(obj) for obj in resp.objects]
