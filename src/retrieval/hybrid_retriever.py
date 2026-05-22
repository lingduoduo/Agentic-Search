"""Hybrid retrieval: fuses dense and sparse results with Reciprocal Rank Fusion."""

from __future__ import annotations

import concurrent.futures
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .dense_retriever import DenseRetriever, DenseRetrieverConfig
from .sparse_retriever import SparseRetriever, SparseRetrieverConfig

# Standard RRF constant — larger k reduces sensitivity to rank differences at the top.
_RRF_K = 60


def combine_retrieval_results(
    result_sets: list[list[dict[str, Any]]],
    *,
    rrf_k: int = _RRF_K,
) -> list[dict[str, Any]]:
    """Merge multiple ranked result sets into one list via Reciprocal Rank Fusion.

    Each result in a set must be a dict with a "document" key (containing at least
    an "id" field) and a "score" key.  Duplicate documents across sets accumulate
    RRF scores; the returned list is sorted by fused score descending.

    Borrows the dedup+merge shape from Danswer's combine_retrieval_results but uses
    rank-based RRF fusion instead of raw-score max to avoid scale differences between
    dense (cosine ~[-1,1]) and BM25 (unbounded positive) scores.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, dict[str, Any]] = {}

    for result_set in result_sets:
        for rank, result in enumerate(result_set, 1):
            doc_id = str(result["document"].get("id", ""))
            rrf_scores[doc_id] += 1.0 / (rrf_k + rank)
            if doc_id not in first_seen:
                first_seen[doc_id] = result

    return sorted(
        [{**first_seen[doc_id], "score": rrf_scores[doc_id]} for doc_id in rrf_scores],
        key=lambda r: r["score"],
        reverse=True,
    )


@dataclass(frozen=True)
class HybridRetrieverConfig:
    """Config for a retriever that fuses dense and sparse search.

    hybrid_alpha controls which backends are active:
        0.0  — pure BM25 (sparse only, no embedding computed)
        1.0  — pure dense (embedding only, BM25 index not loaded)
        0 < alpha < 1 — both run in parallel and results are fused with RRF
    """

    dense: DenseRetrieverConfig
    sparse: SparseRetrieverConfig | None = None
    hybrid_alpha: float = 0.5
    rrf_k: int = _RRF_K

    def validate(self) -> None:
        if not 0.0 <= self.hybrid_alpha <= 1.0:
            raise ValueError("hybrid_alpha must be between 0.0 and 1.0.")
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be at least 1.")
        if self.hybrid_alpha < 1.0 and self.sparse is None:
            raise ValueError(
                "sparse config is required when hybrid_alpha < 1.0 "
                "(pure-dense mode requires hybrid_alpha=1.0)."
            )
        self.dense.validate()
        if self.sparse is not None:
            self.sparse.validate()


class HybridRetriever:
    """Runs dense and/or BM25 search and fuses results with Reciprocal Rank Fusion.

    Borrows the parallel-dispatch and alpha-gating pattern from Danswer's
    search_chunks / _embed_and_hybrid_search design: at alpha=0 only BM25 runs
    (no embedding computed), at alpha=1 only dense runs, in between both execute
    concurrently in a thread pool and their ranked lists are merged with RRF.
    """

    def __init__(self, config: HybridRetrieverConfig) -> None:
        config.validate()
        self.config = config
        self._dense: DenseRetriever | None = None
        self._sparse: SparseRetriever | None = None
        if config.hybrid_alpha > 0.0:
            self._dense = DenseRetriever(config.dense)
        if config.hybrid_alpha < 1.0:
            assert config.sparse is not None
            self._sparse = SparseRetriever(config.sparse)

    def retrieve(
        self,
        queries: list[str],
        topk: int | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Return one ranked result list per query.

        Each result is {"document": dict, "score": float}.  In pure-dense or
        pure-BM25 mode scores are the retriever's native scores; in hybrid mode
        scores are fused RRF values (not directly comparable to either native scale).
        """
        # Pure modes — single retriever, no fusion overhead.
        if self._sparse is None:
            assert self._dense is not None
            return self._dense.retrieve(queries, topk)
        if self._dense is None:
            return self._sparse.retrieve(queries, topk)

        # Hybrid — dispatch both retrievers concurrently then fuse per query.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            dense_fut = pool.submit(self._dense.retrieve, queries, topk)
            sparse_fut = pool.submit(self._sparse.retrieve, queries, topk)
            dense_results = dense_fut.result()
            sparse_results = sparse_fut.result()

        return [
            combine_retrieval_results(
                [dense_results[i], sparse_results[i]],
                rrf_k=self.config.rrf_k,
            )
            for i in range(len(queries))
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
        """Same interface as DenseRetriever.batch_search / SparseRetriever.batch_search."""
        results = self.retrieve(query_list, topk=num)
        if not return_score:
            return [[item["document"] for item in row] for row in results]
        documents, scores = [], []
        for row in results:
            documents.append([item["document"] for item in row])
            scores.append([float(item["score"]) for item in row])
        return documents, scores
