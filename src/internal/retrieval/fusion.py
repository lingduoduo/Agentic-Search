"""RRF fusion and MMR re-ranking over RetrievalResult objects."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .backends.base import RetrievalResult


@dataclass
class FusionWeights:
    w_sparse: float = 0.5
    w_dense: float = 0.5


_RRF_K = 60


def _source_prefix(doc_id: str) -> str:
    """Source-level prefix of doc_id used as a cheap similarity proxy for MMR."""
    sep = doc_id.rfind("-")
    return doc_id[:sep] if sep > 0 else doc_id


def rrf_fuse(
    result_sets: list[list[RetrievalResult]],
    *,
    rrf_k: int = _RRF_K,
) -> list[RetrievalResult]:
    """Merge ranked result sets via Reciprocal Rank Fusion.

    Score formula: score(doc) = Σ 1 / (k + rank)  for each set the doc appears in.
    Scale-invariant — no normalisation of raw BM25 or cosine scores required.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, RetrievalResult] = {}

    for result_set in result_sets:
        for rank, result in enumerate(result_set, 1):
            rrf_scores[result.doc_id] += 1.0 / (rrf_k + rank)
            if result.doc_id not in first_seen:
                first_seen[result.doc_id] = result

    return sorted(
        [
            RetrievalResult(
                doc_id=doc_id,
                title=first_seen[doc_id].title,
                text=first_seen[doc_id].text,
                url=first_seen[doc_id].url,
                score=rrf_scores[doc_id],
                metadata=first_seen[doc_id].metadata,
            )
            for doc_id in rrf_scores
        ],
        key=lambda r: r.score,
        reverse=True,
    )


def weighted_rrf_fuse(
    result_sets: list[list[RetrievalResult]],
    weights: FusionWeights | None = None,
    *,
    rrf_k: int = _RRF_K,
) -> list[RetrievalResult]:
    """RRF fusion with per-source weights (defaults to uniform = standard RRF).

    Assumes result_sets[0] = sparse, result_sets[1] = dense.
    Falls back to rrf_fuse() when weights is None or len(result_sets) != 2.
    """
    if weights is None or len(result_sets) != 2:
        return rrf_fuse(result_sets, rrf_k=rrf_k)

    w = [weights.w_sparse, weights.w_dense]
    rrf_scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, RetrievalResult] = {}

    for i, result_set in enumerate(result_sets):
        for rank, result in enumerate(result_set, 1):
            rrf_scores[result.doc_id] += w[i] * (1.0 / (rrf_k + rank))
            if result.doc_id not in first_seen:
                first_seen[result.doc_id] = result

    return sorted(
        [
            RetrievalResult(
                doc_id=doc_id,
                title=first_seen[doc_id].title,
                text=first_seen[doc_id].text,
                url=first_seen[doc_id].url,
                score=rrf_scores[doc_id],
                metadata=first_seen[doc_id].metadata,
            )
            for doc_id in rrf_scores
        ],
        key=lambda r: r.score,
        reverse=True,
    )


def mmr_rerank(
    results: list[RetrievalResult],
    *,
    top_k: int,
    mmr_lambda: float = 0.5,
) -> list[RetrievalResult]:
    """Re-rank with Maximal Marginal Relevance.

    Uses source-prefix matching as a cheap inter-document similarity proxy —
    no embeddings required at re-rank time.

    mmr_lambda=1.0 → pure relevance order (no diversity penalty).
    mmr_lambda=0.0 → maximum diversity.
    """
    if not results:
        return []
    if mmr_lambda == 1.0:
        return results[:top_k]

    max_score = max(r.score for r in results) or 1.0
    normalized = [(r, r.score / max_score) for r in results]

    selected: list[RetrievalResult] = []
    selected_prefixes: list[str] = []
    remaining = list(normalized)

    while remaining and len(selected) < top_k:
        if not selected:
            best = max(remaining, key=lambda x: x[1])
        else:

            def _mmr(
                item: tuple[RetrievalResult, float],
                _prefixes: list[str] = selected_prefixes,
            ) -> float:
                r, rel = item
                sim = 1.0 if _source_prefix(r.doc_id) in _prefixes else 0.0
                return mmr_lambda * rel - (1.0 - mmr_lambda) * sim

            best = max(remaining, key=_mmr)

        result, _ = best
        selected.append(result)
        selected_prefixes.append(_source_prefix(result.doc_id))
        remaining.remove(best)

    return selected
