"""Hybrid Retrieval Query Construction — fusion config for the hybrid leg."""

from __future__ import annotations

from src.internal.retrieval.fusion_learner import adaptive_mmr_lambda

from ..route import RetrieverTarget, RouteDecision
from .base import ConstructedQuery


class HybridRetrievalQueryConstructor:
    def __init__(
        self, rrf_k: int = 60, w_sparse: float = 0.5, w_dense: float = 0.5
    ) -> None:
        self._rrf_k = rrf_k
        self._w_sparse = w_sparse
        self._w_dense = w_dense

    def construct(self, query: str, route: RouteDecision) -> ConstructedQuery:
        return ConstructedQuery(
            target=RetrieverTarget.HYBRID,
            payload={
                "rrf_k": self._rrf_k,
                "w_sparse": self._w_sparse,
                "w_dense": self._w_dense,
                "mmr_lambda": adaptive_mmr_lambda(query),
            },
            text=query,
        )
