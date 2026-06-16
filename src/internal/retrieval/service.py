"""RetrievalService: selects backend from env and exposes search()."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from .backends.base import RetrievalBackend, RetrievalResult
from .fusion import mmr_rerank, rrf_fuse

if TYPE_CHECKING:
    from src.internal.retrieval.reranker import Reranker
    from src.context.query_transform import QueryTransformPipeline

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


def _build_llm() -> object:
    """Build an LLM client from GEN_AI_* environment variables."""
    from src.internal.llm.interfaces import LLMConfig
    from src.internal.llm.providers import OpenAICompatibleLLM

    return OpenAICompatibleLLM(
        LLMConfig(
            model_provider=os.environ.get("GEN_AI_MODEL_PROVIDER", "openai"),
            model_name=os.environ.get("GEN_AI_MODEL_VERSION", "gpt-4o-mini"),
            api_key=os.environ.get("GEN_AI_API_KEY")
            or os.environ.get("OPENAI_API_KEY"),
            api_base=os.environ.get("GEN_AI_API_BASE"),
            max_input_tokens=int(os.environ.get("GEN_AI_MAX_INPUT_TOKENS", "8192")),
        )
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
    def __init__(
        self,
        backend: RetrievalBackend,
        reranker: "Reranker | None" = None,
        pipeline: "QueryTransformPipeline | None" = None,
    ) -> None:
        self._backend = backend
        self._reranker = reranker
        self._pipeline = pipeline

    @classmethod
    def from_env(cls) -> "RetrievalService":
        """Construct service from environment variables."""
        from src.internal.retrieval.reranker import Reranker

        pipeline = None
        _qt_flags = (
            "QT_DECOMPOSE",
            "QT_HYDE",
            "QT_STEP_BACK",
            "QT_KEYWORDS",
            "QT_CONSTRUCT_FILTERS",
        )
        if any(
            os.environ.get(v, "").lower() in ("1", "true", "yes") for v in _qt_flags
        ):
            from src.context.query_transform import QueryTransformPipeline

            pipeline = QueryTransformPipeline.from_env(_build_llm())

        return cls(_build_backend(), reranker=Reranker.from_env(), pipeline=pipeline)

    def _search_one(
        self,
        query: str,
        over_fetch: int,
        filters: dict | None,
    ) -> tuple[list[RetrievalResult], str]:
        """Run sparse+dense retrieval for one query variant. Returns (results, base_mode)."""
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

        try:
            sparse_results = sparse_future.result()
            sparse_ok = True
        except Exception as exc:
            logger.warning("Sparse retrieval leg failed: %s", exc)

        try:
            dense_results = dense_future.result()
            dense_ok = True
        except NotImplementedError:
            pass
        except Exception as exc:
            logger.warning("Dense retrieval leg failed: %s", exc)

        if not sparse_ok and not dense_ok:
            raise RuntimeError("Both retrieval legs failed")

        if not dense_ok:
            return sparse_results, "sparse_only"
        elif not sparse_ok:
            return dense_results, "dense_only"
        else:
            return rrf_fuse([sparse_results, dense_results]), "hybrid"

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> tuple[list[RetrievalResult], str]:
        """Run retrieval, fuse with RRF+MMR, fall back gracefully.

        When pipeline is set: generates query variants, retrieves per variant in
        parallel, fuses all result sets via RRF → mode gains '+rag_fusion' suffix.
        Without pipeline: single query, identical behaviour to previous releases.
        """
        over_fetch = top_k * int(os.environ.get("OVER_FETCH_MULTIPLIER", "2"))

        if self._pipeline:
            bundle = self._pipeline.transform(query, filters)
            variants = bundle.retrieval_variants(self._pipeline._config.max_variants)
            active_filters: dict | None = bundle.merged_filters or None
        else:
            variants = [query]
            active_filters = filters

        if (
            not variants
        ):  # guard: retrieval_variants() returned [] (shouldn't happen but be safe)
            variants = [query]

        max_workers = min(len(variants), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._search_one, v, over_fetch, active_filters)
                for v in variants
            ]
        result_sets_with_modes = [f.result() for f in futures]
        all_result_sets = [rs for rs, _ in result_sets_with_modes]
        base_mode = (
            result_sets_with_modes[0][1] if result_sets_with_modes else "sparse_only"
        )

        if len(all_result_sets) > 1:
            fused = rrf_fuse(all_result_sets)
            fused = mmr_rerank(fused, top_k=top_k)
            mode = f"{base_mode}+rag_fusion"
        else:
            raw = all_result_sets[0] if all_result_sets else []
            if base_mode == "hybrid":
                fused = mmr_rerank(raw, top_k=top_k)
            else:
                fused = raw[:top_k]
            mode = base_mode

        if self._reranker:
            fused = self._reranker.rerank(query, fused, top_k)
            mode = f"{mode}+reranked"

        return fused, mode

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
