"""RetrievalService: selects backend from env and exposes search()."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

from .backends.base import RetrievalBackend, RetrievalResult
from .fusion import mmr_rerank, rrf_fuse

logger = logging.getLogger(__name__)


def _build_local_backend() -> RetrievalBackend:
    from src.internal.document_index.retrieval import (
        DenseRetrieverConfig,
        SparseRetrieverConfig,
    )

    from .backends.local import LocalBackend

    sparse_config = SparseRetrieverConfig(
        index_path=os.environ["BM25_INDEX_PATH"],
        corpus_path=os.environ.get("BM25_CORPUS_PATH", "data/corpus.jsonl"),
        topk=int(os.environ.get("BM25_TOP_K", "20")),
        k1=float(os.environ.get("BM25_K1", "1.2")),
        b=float(os.environ.get("BM25_B", "0.75")),
    )
    dense_config: DenseRetrieverConfig | None = None
    if os.environ.get("DENSE_MODEL_PATH"):
        dense_config = DenseRetrieverConfig.for_e5_base_v2(
            model_path=os.environ["DENSE_MODEL_PATH"],
            index_path=os.environ["DENSE_INDEX_PATH"],
            corpus_path=os.environ.get("DENSE_CORPUS_PATH", "data/corpus.jsonl"),
            topk=int(os.environ.get("BM25_TOP_K", "20")),
            device=os.environ.get("DENSE_DEVICE", "cpu"),
            redis_url=os.environ.get("DENSE_REDIS_URL"),
        )
    return LocalBackend(sparse_config, dense_config=dense_config)


def _build_opensearch_backend() -> RetrievalBackend:
    from .backends.opensearch import OpenSearchBackend

    return OpenSearchBackend(
        index_name=os.environ["OPENSEARCH_INDEX"],
        content_field=os.environ.get("OPENSEARCH_CONTENT_FIELD", "content"),
        vector_field=os.environ.get("OPENSEARCH_VECTOR_FIELD", "content_vector"),
        doc_id_field=os.environ.get("OPENSEARCH_DOC_ID_FIELD", "document_id"),
        title_field=os.environ.get("OPENSEARCH_TITLE_FIELD", "title"),
        url_field=os.environ.get("OPENSEARCH_URL_FIELD", "source_links"),
    )


def _build_weaviate_backend() -> RetrievalBackend:
    from .backends.weaviate import WeaviateBackend

    return WeaviateBackend(
        collection_name=os.environ["WEAVIATE_COLLECTION"],
    )


def _build_backend() -> RetrievalBackend:
    name = os.environ.get("RETRIEVAL_BACKEND", "local").lower()
    if name == "local":
        return _build_local_backend()
    if name == "opensearch":
        return _build_opensearch_backend()
    if name == "weaviate":
        return _build_weaviate_backend()
    raise ValueError(
        f"Unknown RETRIEVAL_BACKEND: {name!r}. Supported values: local, opensearch, weaviate"
    )


class RetrievalService:
    def __init__(self, backend: RetrievalBackend) -> None:
        self._backend = backend

    @classmethod
    def from_env(cls) -> "RetrievalService":
        """Construct service from environment variables."""
        return cls(_build_backend())

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> tuple[list[RetrievalResult], str]:
        """Run sparse and dense legs, fuse with RRF+MMR, fall back gracefully.

        filters: optional key/value pairs applied by each backend before returning results.
        Returns (results, retrieval_mode) where mode is 'hybrid' | 'sparse_only' | 'dense_only'.
        """
        over_fetch = top_k * int(os.environ.get("OVER_FETCH_MULTIPLIER", "2"))

        sparse_results: list[RetrievalResult] = []
        dense_results: list[RetrievalResult] = []
        sparse_ok = dense_ok = False

        with ThreadPoolExecutor(max_workers=2) as executor:
            sparse_future = executor.submit(
                self._backend.search_sparse, query, top_k=over_fetch, filters=filters
            )
            dense_future = executor.submit(
                self._backend.search_dense, query, top_k=over_fetch, filters=filters
            )
        # Both futures are complete once the with-block exits.

        try:
            sparse_results = sparse_future.result()
            sparse_ok = True
        except Exception as exc:
            logger.warning("Sparse retrieval leg failed: %s", exc)

        try:
            dense_results = dense_future.result()
            dense_ok = True
        except NotImplementedError:
            pass  # dense not configured — silent fallback
        except Exception as exc:
            logger.warning("Dense retrieval leg failed: %s", exc)

        if not sparse_ok and not dense_ok:
            raise RuntimeError("Both retrieval legs failed")

        if not dense_ok:
            return sparse_results[:top_k], "sparse_only"
        if not sparse_ok:
            return dense_results[:top_k], "dense_only"

        fused = rrf_fuse([sparse_results, dense_results])
        reranked = mmr_rerank(fused, top_k=top_k)
        return reranked, "hybrid"

    def graph_search(
        self,
        query: str,
        top_k: int = 10,
        initial_k: int = 5,
        max_entity_queries: int = 3,
    ) -> list[RetrievalResult]:
        """Graph-augmented retrieval: seed search → entity expansion → RRF fusion."""
        from .graph_rag import graph_rag_search

        return graph_rag_search(
            query,
            service=self,
            top_k=top_k,
            initial_k=initial_k,
            max_entity_queries=max_entity_queries,
        )
