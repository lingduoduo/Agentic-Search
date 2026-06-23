"""Per-query router: heuristic default with optional logical/semantic strategies."""

from __future__ import annotations

import logging

from .registry import RouteRegistry
from .route import RetrieverTarget, Route, RouteDecision

logger = logging.getLogger(__name__)

_SQL_CUES = (
    "how many",
    "how much",
    "count of",
    "number of",
    "total ",
    "sum of",
    "average ",
    "avg ",
    "per year",
    "per month",
    "per ",
    "group by",
    "most ",
    "least ",
    "top ",
    "ranked by",
    "aggregate",
)
_GRAPH_CUES = (
    "connected to",
    "related to",
    "relationship between",
    "related entities",
    "linked to",
    "associated with",
    "path between",
    "neighbors of",
    "depends on",
)
_API_CUES = (
    "current ",
    "latest ",
    "real-time",
    "real time",
    "right now",
    "today's",
    "live ",
    "as of now",
    "up to date",
    "up-to-date",
)


def _matches(query: str, cues: tuple[str, ...]) -> bool:
    q = query.lower()
    return any(cue in q for cue in cues)


class Router:
    def __init__(
        self,
        registry: RouteRegistry,
        llm: object | None = None,
        embedder: object | None = None,
        logical: bool = False,
        semantic: bool = False,
    ) -> None:
        self._registry = registry
        self._llm = llm
        self._embedder = embedder
        self._logical = logical
        self._semantic = semantic

    def route(self, query: str) -> RouteDecision:
        if self._logical and self._llm is not None:
            try:
                return self._logical_route(query)
            except Exception as exc:
                logger.warning("logical route failed, falling back: %s", exc)
        if self._semantic and self._embedder is not None:
            try:
                return self._semantic_route(query)
            except Exception as exc:
                logger.warning("semantic route failed, falling back: %s", exc)
        return self._heuristic(query)

    def _decision(
        self, route: Route, *, confidence: float, strategy: str
    ) -> RouteDecision:
        return RouteDecision(
            domain=route.name,
            sources=list(route.sources),
            retriever=route.retriever,
            construction_target=route.retriever,
            confidence=confidence,
            strategy=strategy,
        )

    def _route_for_target(self, target: RetrieverTarget) -> Route:
        return self._registry.by_retriever(target) or self._registry.default()

    def _heuristic(self, query: str) -> RouteDecision:
        if _matches(query, _SQL_CUES):
            target = RetrieverTarget.SQL
        elif _matches(query, _GRAPH_CUES):
            target = RetrieverTarget.GRAPH
        elif _matches(query, _API_CUES):
            target = RetrieverTarget.API
        else:
            return self._decision(
                self._registry.default(), confidence=0.5, strategy="heuristic"
            )
        route = self._registry.by_retriever(target)
        if route is None:  # target not registered → safe default
            return self._decision(
                self._registry.default(), confidence=0.5, strategy="heuristic"
            )
        return self._decision(route, confidence=0.7, strategy="heuristic")

    # Logical/semantic strategies are implemented in Task 3.
    def _logical_route(self, query: str) -> RouteDecision:  # pragma: no cover - Task 3
        raise NotImplementedError

    def _semantic_route(self, query: str) -> RouteDecision:  # pragma: no cover - Task 3
        raise NotImplementedError
