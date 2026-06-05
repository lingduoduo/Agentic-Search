"""Utilities for querying dense retrieval indexes."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .embedding_cache import EmbeddingCache
from .index_builder import (
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
        self._cache = (
            EmbeddingCache(redis_url=config.redis_url) if config.redis_url else None
        )

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        torch = _require_torch()
        non_empty = [q.strip() for q in queries if q.strip()]
        if not non_empty:
            return np.empty((0, 0), dtype=np.float32)

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
        results = self.retrieve(query_list, topk=num)
        if not return_score:
            return [[item["document"] for item in row] for row in results]
        documents, scores = [], []
        for row in results:
            documents.append([item["document"] for item in row])
            scores.append([float(item["score"]) for item in row])
        return documents, scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a dense retrieval index.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--retrieval_method", type=str, required=True)
    parser.add_argument("--queries", nargs="+", required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=180)
    parser.add_argument("--query_batch_size", type=int, default=128)
    parser.add_argument("--use_fp16", default=False, action="store_true")
    parser.add_argument("--pooling_method", type=str, default=None)
    parser.add_argument("--faiss_gpu", default=False, action="store_true")
    parser.add_argument(
        "--normalize_query_embeddings", default=False, action="store_true"
    )
    parser.add_argument("--query_prefix", type=str, default=None)
    parser.add_argument("--passage_prefix", type=str, default=None)
    parser.add_argument(
        "--hnsw_ef_search",
        type=int,
        default=None,
        help="Optional FAISS HNSW efSearch value for CPU ANN retrieval.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = DenseRetriever(
        DenseRetrieverConfig(
            model_path=args.model_path,
            index_path=args.index_path,
            corpus_path=args.corpus_path,
            retrieval_method=args.retrieval_method,
            topk=args.topk,
            max_length=args.max_length,
            query_batch_size=args.query_batch_size,
            use_fp16=args.use_fp16,
            pooling_method=args.pooling_method,
            faiss_gpu=args.faiss_gpu,
            normalize_query_embeddings=args.normalize_query_embeddings,
            query_prefix=args.query_prefix,
            passage_prefix=args.passage_prefix,
            hnsw_ef_search=args.hnsw_ef_search,
        )
    )
    print(json.dumps(retriever.retrieve(args.queries), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
