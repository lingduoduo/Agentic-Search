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


class WeaviateBackend(RetrievalBackend):
    """Retrieves from a Weaviate collection via BM25 (sparse) and nearVector (dense)."""

    def __init__(
        self,
        collection_name: str,
        *,
        embedder: Callable[[str], list[float]] | None = None,
        client: Any = None,
    ) -> None:
        self._collection_name = collection_name
        self._embedder = embedder
        self._client = client or _make_weaviate_client(
            os.environ.get("WEAVIATE_URL", "http://localhost:8080")
        )

    @property
    def _collection(self) -> Any:
        return self._client.collections.get(self._collection_name)

    def search_sparse(self, query: str, top_k: int) -> list[RetrievalResult]:
        resp = self._collection.query.bm25(query=query, limit=top_k)
        return [_obj_to_result(obj) for obj in resp.objects]

    def search_dense(self, query: str, top_k: int) -> list[RetrievalResult]:
        if self._embedder is None:
            raise NotImplementedError(
                "Dense search not configured — provide an embedder or set DENSE_MODEL_PATH"
            )
        vector = self._embedder(query)
        resp = self._collection.query.near_vector(near_vector=vector, limit=top_k)
        return [_obj_to_result(obj) for obj in resp.objects]
