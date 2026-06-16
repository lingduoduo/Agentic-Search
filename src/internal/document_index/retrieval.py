"""Local dense and sparse document-index readers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.internal.document_index.embedding_cache import EmbeddingCache, OpenAIEmbedder
from src.internal.document_index.index_builder import (
    _encode_batch,
    _normalize_embedding_rows,
    _require_faiss,
    _require_torch,
    load_corpus,
    load_model,
    prepare_texts,
    resolve_pooling_method,
    set_hnsw_ef_search,
)

# Must be set before torch/faiss are imported to prevent an OpenMP conflict on macOS
# when both libraries bundle their own libomp.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _format_batch_results(
    results: list[list[dict[str, Any]]],
    *,
    return_score: bool,
) -> list[list[dict[str, Any]]] | tuple[list[list[dict[str, Any]]], list[list[float]]]:
    documents = [[item["document"] for item in row] for row in results]
    if not return_score:
        return documents
    return documents, [[float(item["score"]) for item in row] for row in results]


@dataclass(frozen=True)
class DenseRetrieverConfig:
    model_path: str
    index_path: str
    corpus_path: str
    retrieval_method: str
    topk: int = 5
    max_length: int = 180
    query_batch_size: int = 128
    use_fp16: bool = False
    pooling_method: str | None = None
    faiss_gpu: bool = False
    hnsw_ef_search: int | None = None
    normalize_query_embeddings: bool = False
    query_prefix: str | None = None
    passage_prefix: str | None = None
    # Default to CPU so the retrieval service never competes with the trainer
    # for GPU memory.  Set to "cuda" (or "cuda:N") only when the retrieval
    # server is deployed on a dedicated GPU that the trainer does not use.
    device: str = "cpu"
    # Optional Redis URL for query-embedding cache.  None disables caching.
    redis_url: str | None = None
    # When set, use OpenAI embeddings instead of the local model.
    # Requires OPENAI_API_KEY in env (or pass via openai_api_key).
    openai_embedding_model: str | None = None
    openai_api_key: str | None = None

    def validate(self) -> None:
        if not self.model_path:
            raise ValueError("model_path is required.")
        if not self.index_path:
            raise ValueError("index_path is required.")
        if not self.corpus_path:
            raise ValueError("corpus_path is required.")
        if not self.retrieval_method:
            raise ValueError("retrieval_method is required.")
        if self.topk < 1:
            raise ValueError("topk must be at least 1.")
        if self.query_batch_size < 1:
            raise ValueError("query_batch_size must be at least 1.")
        if self.hnsw_ef_search is not None and self.hnsw_ef_search < 1:
            raise ValueError("hnsw_ef_search must be at least 1.")

    @classmethod
    def for_e5_base_v2(
        cls,
        model_path: str = "intfloat/e5-base-v2",
        *,
        index_path: str,
        corpus_path: str,
        topk: int = 5,
        device: str = "cpu",
        **kwargs: object,
    ) -> "DenseRetrieverConfig":
        """Preset for the intfloat/e5-base-v2 retriever.

        Runs on CPU by default so the trainer GPU is not affected.
        Pass ``device="cuda:1"`` when the retrieval server has a
        dedicated GPU separate from the trainer.
        """
        return cls(
            model_path=model_path,
            index_path=index_path,
            corpus_path=corpus_path,
            retrieval_method="e5",
            topk=topk,
            device=device,
            **kwargs,
        )


class DenseRetriever:
    """Loads a FAISS index and returns the top matching corpus documents."""

    def __init__(self, config: DenseRetrieverConfig):
        config.validate()
        self.config = config
        _require_torch()
        faiss = _require_faiss()

        # Use the configured device.  The default is "cpu" so that a
        # standalone retrieval service never competes with the trainer for
        # GPU memory.  The trainer calls this service over HTTP and never
        # loads the embedding model or FAISS index in-process.
        self.device = config.device
        self.pooling_method = resolve_pooling_method(
            config.retrieval_method,
            config.pooling_method,
        )
        self.model, self.tokenizer = load_model(
            model_path=config.model_path,
            use_fp16=config.use_fp16,
            device=self.device,
        )
        self.index = faiss.read_index(config.index_path)
        if config.hnsw_ef_search is not None:
            set_hnsw_ef_search(self.index, config.hnsw_ef_search)
        if config.faiss_gpu:
            if (
                not hasattr(faiss, "GpuMultipleClonerOptions")
                or not getattr(faiss, "get_num_gpus", lambda: 0)()
            ):
                raise RuntimeError(
                    "faiss_gpu was requested, but GPU FAISS support is not available."
                )
            clone_options = faiss.GpuMultipleClonerOptions()
            clone_options.useFloat16 = True
            clone_options.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, clone_options)
        self.corpus = load_corpus(config.corpus_path)
        if config.openai_embedding_model:
            self._openai_embedder = OpenAIEmbedder(
                model=config.openai_embedding_model,
                redis_url=config.redis_url,
                api_key=config.openai_api_key,
            )
            self._cache = None
        else:
            self._openai_embedder = None
            self._cache = (
                EmbeddingCache(redis_url=config.redis_url) if config.redis_url else None
            )

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        torch = _require_torch()
        non_empty = [q.strip() for q in queries if q.strip()]
        if not non_empty:
            return np.empty((0, 0), dtype=np.float32)

        # OpenAI path: Redis cache is handled inside OpenAIEmbedder.
        if self._openai_embedder is not None:
            return self._openai_embedder.embed(non_empty)

        # --- Redis cache: resolve hits, collect misses ---
        if self._cache is not None:
            cached = self._cache.get_batch(non_empty)
            miss_indices = [i for i, v in enumerate(cached) if v is None]
            miss_texts = [non_empty[i] for i in miss_indices]
        else:
            cached = [None] * len(non_empty)
            miss_indices = list(range(len(non_empty)))
            miss_texts = non_empty

        # --- Encode only cache misses ---
        if miss_texts:
            texts = prepare_texts(
                miss_texts,
                self.config.retrieval_method,
                is_query=True,
                query_prefix=self.config.query_prefix,
                passage_prefix=self.config.passage_prefix,
            )
            with torch.no_grad():
                fresh = _encode_batch(
                    self.model,
                    self.tokenizer,
                    texts,
                    self.config.retrieval_method.lower(),
                    self.config.max_length,
                    self.pooling_method,
                    self.device,
                )
            if self.config.normalize_query_embeddings:
                fresh = _normalize_embedding_rows(fresh)
            if self._cache is not None:
                self._cache.set_batch(miss_texts, fresh)
            for i, row in zip(miss_indices, fresh):
                cached[i] = row

        return np.stack(cached)  # type: ignore[arg-type]

    def _encode_query_batch(
        self,
        query_batch: list[tuple[int, str]],
    ) -> tuple[list[int], np.ndarray]:
        """Encode unique queries once while preserving result row order."""

        unique_queries: list[str] = []
        query_to_unique_index: dict[str, int] = {}
        row_to_unique_index: list[int] = []
        for _, query in query_batch:
            unique_index = query_to_unique_index.get(query)
            if unique_index is None:
                unique_index = len(unique_queries)
                query_to_unique_index[query] = unique_index
                unique_queries.append(query)
            row_to_unique_index.append(unique_index)

        unique_embeddings = self.encode_queries(unique_queries)
        if unique_embeddings.shape[0] != len(unique_queries):
            raise ValueError("encode_queries returned an unexpected number of rows.")
        return (
            [query_index for query_index, _ in query_batch],
            unique_embeddings[row_to_unique_index],
        )

    def retrieve(
        self, queries: list[str], topk: int | None = None
    ) -> list[list[dict[str, Any]]]:
        resolved_topk = topk if topk is not None else self.config.topk
        clean_queries = [query.strip() for query in queries]
        results: list[list[dict[str, Any]]] = [[] for _ in clean_queries]
        non_empty = [
            (index, query) for index, query in enumerate(clean_queries) if query
        ]
        for start in range(0, len(non_empty), self.config.query_batch_size):
            query_batch = non_empty[start : start + self.config.query_batch_size]
            query_indices, batch_embeddings = self._encode_query_batch(query_batch)
            if batch_embeddings.size == 0:
                continue
            scores, indices = self.index.search(batch_embeddings, resolved_topk)
            for query_index, row_scores, row_indices in zip(
                query_indices, scores, indices
            ):
                query_results: list[dict[str, Any]] = []
                for score, idx in zip(row_scores, row_indices):
                    if idx < 0:
                        continue
                    query_results.append(
                        {
                            "document": self.corpus[int(idx)],
                            "score": float(score),
                        }
                    )
                results[query_index] = query_results

        return results

    def batch_search(
        self,
        query_list: list[str],
        num: int | None = None,
        return_score: bool = False,
    ) -> (
        list[list[dict[str, Any]]]
        | tuple[list[list[dict[str, Any]]], list[list[float]]]
    ):
        return _format_batch_results(
            self.retrieve(query_list, topk=num),
            return_score=return_score,
        )


def _require_lucene_searcher():
    from pyserini.search.lucene import LuceneSearcher

    return LuceneSearcher


@dataclass(frozen=True)
class SparseRetrieverConfig:
    index_path: str
    corpus_path: str
    retrieval_method: str = "bm25"
    topk: int = 5
    k1: float = 1.2
    b: float = 0.75

    def validate(self) -> None:
        if not self.index_path:
            raise ValueError("index_path is required.")
        if not self.corpus_path:
            raise ValueError("corpus_path is required.")
        if self.retrieval_method.lower() != "bm25":
            raise ValueError(
                "SparseRetriever currently supports retrieval_method='bm25'."
            )
        if self.topk < 1:
            raise ValueError("topk must be at least 1.")


class SparseRetriever:
    """Loads a local Pyserini BM25 index and returns matching corpus documents."""

    def __init__(self, config: SparseRetrieverConfig):
        config.validate()
        self.config = config
        self.searcher = _require_lucene_searcher()(config.index_path)
        self.searcher.set_bm25(config.k1, config.b)
        self._corpus_by_id: dict[str, dict[str, Any]] | None = None

    def _ensure_corpus_by_id(self) -> dict[str, dict[str, Any]]:
        if self._corpus_by_id is None:
            self._corpus_by_id = {
                str(item["id"]): item
                for item in load_corpus(self.config.corpus_path)
                if item.get("id") is not None
            }
        return self._corpus_by_id

    def _document_for_hit(self, hit: Any) -> dict[str, Any]:
        raw_doc = self.searcher.doc(hit.docid)
        raw = raw_doc.raw() if raw_doc is not None else ""
        if raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return {"id": hit.docid, "contents": raw}
            if isinstance(parsed, dict):
                return parsed
        return self._ensure_corpus_by_id().get(
            str(hit.docid), {"id": hit.docid, "contents": ""}
        )

    def retrieve(
        self, queries: list[str], topk: int | None = None
    ) -> list[list[dict[str, Any]]]:
        resolved_topk = topk if topk is not None else self.config.topk
        return [
            [
                {
                    "document": self._document_for_hit(hit),
                    "score": float(hit.score),
                }
                for hit in self.searcher.search(query.strip(), k=resolved_topk)
            ]
            if query.strip()
            else []
            for query in queries
        ]

    def batch_search(
        self,
        query_list: list[str],
        num: int | None = None,
        return_score: bool = False,
    ) -> (
        list[list[dict[str, Any]]]
        | tuple[list[list[dict[str, Any]]], list[list[float]]]
    ):
        return _format_batch_results(
            self.retrieve(query_list, topk=num),
            return_score=return_score,
        )


def build_retriever(config: Any) -> DenseRetriever | SparseRetriever | Any:
    """Instantiate the local retriever matching the supplied configuration."""
    from .hybrid_retriever import HybridRetriever, HybridRetrieverConfig

    if isinstance(config, HybridRetrieverConfig):
        return HybridRetriever(config)
    if isinstance(config, SparseRetrieverConfig):
        return SparseRetriever(config)
    if config.retrieval_method.lower() == "bm25":
        return SparseRetriever(
            SparseRetrieverConfig(
                index_path=config.index_path,
                corpus_path=config.corpus_path,
                retrieval_method=config.retrieval_method,
                topk=config.topk,
            )
        )
    return DenseRetriever(config)


__all__ = [
    "build_retriever",
    "DenseRetriever",
    "DenseRetrieverConfig",
    "SparseRetriever",
    "SparseRetrieverConfig",
]
