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


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-k retrieved documents that are relevant."""
    if not relevant_ids or k == 0:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return hits / k


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """1.0 if any relevant document appears in the top-k results, else 0.0."""
    if not relevant_ids:
        return 0.0
    return 1.0 if any(doc_id in relevant_ids for doc_id in retrieved_ids[:k]) else 0.0


def map_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Mean average precision at k (binary relevance)."""
    if not relevant_ids:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], 1):
        if doc_id in relevant_ids:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant_ids)


def reranker_improvement_ratio(pre_ndcg: float, post_ndcg: float) -> float:
    """(post / pre) - 1; negative means reranker hurt quality. Returns 0.0 if pre is 0."""
    if pre_ndcg == 0.0:
        return 0.0
    return post_ndcg / pre_ndcg - 1.0


def routing_accuracy(predictions: list[str], labels: list[str]) -> float:
    """Top-1 routing accuracy: fraction of predicted retrievers matching labels."""
    if not labels:
        return 0.0
    correct = sum(1 for p, t in zip(predictions, labels) if p == t)
    return round(correct / len(labels), 4)
