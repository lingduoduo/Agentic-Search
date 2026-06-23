"""Vector Search Query Construction — dense-leg parameters."""

from __future__ import annotations

from ..route import RetrieverTarget, RouteDecision
from .base import ConstructedQuery


class VectorSearchQueryConstructor:
    def __init__(self, top_k: int = 10) -> None:
        self._top_k = top_k

    def construct(self, query: str, route: RouteDecision) -> ConstructedQuery:
        namespace = route.sources[0] if route.sources else None
        return ConstructedQuery(
            target=RetrieverTarget.DENSE,
            payload={"top_k": self._top_k, "namespace": namespace, "filters": {}},
            text=query,
        )
