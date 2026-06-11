"""Hybrid retrieval: fuses dense and sparse results with Reciprocal Rank Fusion."""

from __future__ import annotations

import concurrent.futures
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from src.internal.document_index.retrieval import (
    DenseRetriever,
    DenseRetrieverConfig,
    SparseRetriever,
    SparseRetrieverConfig,
)

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


def _doc_source_prefix(doc_id: str) -> str:
    """Return the source-level prefix of a document id for similarity estimation.

    Convention used in the demo corpus: ids are 'source-name-chunk-N'.
    Prefix = everything before the last '-' separator.  When the id has no
    separator the full id is used (no similarity penalty applied).
    """
    sep = doc_id.rfind("-")
    return doc_id[:sep] if sep > 0 else doc_id


def maximal_marginal_relevance(
    results: list[dict[str, Any]],
    *,
    topk: int,
    mmr_lambda: float = 0.5,
) -> list[dict[str, Any]]:
    """Re-rank `results` with Maximal Marginal Relevance to balance relevance and diversity.

    Uses the RRF/retrieval score as relevance and source-prefix matching as a cheap
    proxy for inter-document similarity (no embeddings required).

    Args:
        results: Ranked list of {"document": dict, "score": float} items.
        topk: Maximum number of results to return.
        mmr_lambda: 1.0 = pure relevance order; 0.0 = maximum diversity.
    """
    if not results:
        return []
    if mmr_lambda == 1.0:
        return results[:topk]

    max_score = max(r["score"] for r in results) or 1.0
    normalized = [(r, r["score"] / max_score) for r in results]

    selected: list[dict[str, Any]] = []
    selected_prefixes: list[str] = []
    remaining = list(normalized)

    while remaining and len(selected) < topk:
        if not selected:
            best = max(remaining, key=lambda x: x[1])
        else:

            def mmr_score(item: tuple[dict[str, Any], float]) -> float:
                result, rel = item
                doc_id = str(result["document"].get("id", ""))
                prefix = _doc_source_prefix(doc_id)
                sim = 1.0 if prefix in selected_prefixes else 0.0
                return mmr_lambda * rel - (1.0 - mmr_lambda) * sim

            best = max(remaining, key=mmr_score)

        result, _ = best
        selected.append(result)
        selected_prefixes.append(
            _doc_source_prefix(str(result["document"].get("id", "")))
        )
        remaining.remove(best)

    return selected


@dataclass(frozen=True)
class HybridRetrieverConfig:
    """Config for a retriever that fuses dense and sparse search.

    hybrid_alpha controls which backends are active:
        0.0  — pure BM25 (sparse only, no embedding computed)
        1.0  — pure dense (embedding only, BM25 index not loaded)
        0 < alpha < 1 — both run in parallel and results are fused with RRF

    mmr_lambda: 1.0 = pure relevance (no diversity), 0.5 = balanced.
    mmr_topk: number of results to return after MMR (None = return all fused results).
    """

    dense: DenseRetrieverConfig
    sparse: SparseRetrieverConfig | None = None
    hybrid_alpha: float = 0.5
    rrf_k: int = _RRF_K
    mmr_lambda: float = 1.0
    mmr_topk: int | None = None

    def validate(self) -> None:
        if not 0.0 <= self.hybrid_alpha <= 1.0:
            raise ValueError("hybrid_alpha must be between 0.0 and 1.0.")
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be at least 1.")
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be between 0.0 and 1.0.")
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
        """Return one ranked result list per query, optionally MMR-diversified."""
        # Pure modes — single retriever, no fusion overhead.
        if self._sparse is None:
            assert self._dense is not None
            raw_results = self._dense.retrieve(queries, topk)
        elif self._dense is None:
            raw_results = self._sparse.retrieve(queries, topk)
        else:
            # Hybrid — dispatch both retrievers concurrently then fuse per query.
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                dense_fut = pool.submit(self._dense.retrieve, queries, topk)
                sparse_fut = pool.submit(self._sparse.retrieve, queries, topk)
                dense_results = dense_fut.result()
                sparse_results = sparse_fut.result()

            raw_results = [
                combine_retrieval_results(
                    [dense_results[i], sparse_results[i]],
                    rrf_k=self.config.rrf_k,
                )
                for i in range(len(queries))
            ]

        # Apply MMR when lambda < 1.0 or an explicit mmr_topk is set.
        if self.config.mmr_lambda < 1.0 or self.config.mmr_topk is not None:
            mmr_topk = self.config.mmr_topk or (
                topk or len(raw_results[0] if raw_results else [])
            )
            return [
                maximal_marginal_relevance(
                    result_list,
                    topk=mmr_topk,
                    mmr_lambda=self.config.mmr_lambda,
                )
                for result_list in raw_results
            ]

        return raw_results

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
