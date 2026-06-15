"""Retrieval evaluation metric functions: Recall@K, NDCG@K, MRR."""

from __future__ import annotations

import math


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant documents found in the top-k results."""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k (binary relevance)."""
    if not relevant_ids:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, doc_id in enumerate(retrieved_ids[:k])
        if doc_id in relevant_ids
    )
    ideal_dcg = sum(
        1.0 / math.log2(rank + 2) for rank in range(min(len(relevant_ids), k))
    )
    return dcg / ideal_dcg if ideal_dcg > 0.0 else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant document (0.0 if none found)."""
    if not relevant_ids:
        return 0.0
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0
