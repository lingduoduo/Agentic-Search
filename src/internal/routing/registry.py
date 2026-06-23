"""Config-driven registry of routes (domains → sources → retriever target)."""

from __future__ import annotations

import json
import logging
import os

from .route import RetrieverTarget, Route

logger = logging.getLogger(__name__)

# The first route is the default (used when no signal matches). It mirrors the
# current corpus: general docs served by hybrid retrieval.
DEFAULT_ROUTES: tuple[Route, ...] = (
    Route(
        "docs",
        "General documentation, articles, and information-retrieval topics; "
        "open-ended semantic or keyword search over unstructured text.",
        ("local",),
        RetrieverTarget.HYBRID,
    ),
    Route(
        "structured",
        "Tabular metrics, counts, aggregations, and numeric records stored in a "
        "relational database; questions asking how many, totals, averages, or per-group breakdowns.",
        ("analytics_db",),
        RetrieverTarget.SQL,
    ),
    Route(
        "graph",
        "Entity relationships and connections: how named entities relate, link, "
        "or connect, and paths between them.",
        ("knowledge_graph",),
        RetrieverTarget.GRAPH,
    ),
    Route(
        "live",
        "Live external data accessed via an API: current prices, weather, or "
        "real-time lookups that change moment to moment.",
        ("external_api",),
        RetrieverTarget.API,
    ),
)


class RouteRegistry:
    def __init__(self, routes) -> None:
        self._routes: list[Route] = list(routes)
        if not self._routes:
            raise ValueError("RouteRegistry requires at least one route")

    @property
    def routes(self) -> list[Route]:
        return list(self._routes)

    def get(self, name: str) -> Route | None:
        return next((r for r in self._routes if r.name == name), None)

    def by_retriever(self, target: RetrieverTarget) -> Route | None:
        return next((r for r in self._routes if r.retriever is target), None)

    def default(self) -> Route:
        return self._routes[0]

    @classmethod
    def from_file(cls, path: str) -> "RouteRegistry":
        with open(path) as f:
            raw = json.load(f)
        routes = [
            Route(
                name=str(item["name"]),
                description=str(item.get("description", "")),
                sources=tuple(item.get("sources", [])),
                retriever=RetrieverTarget(str(item["retriever"]).lower()),
            )
            for item in raw
        ]
        return cls(routes)

    @classmethod
    def from_env(cls) -> "RouteRegistry":
        path = os.environ.get("ROUTING_REGISTRY_PATH")
        if path and os.path.exists(path):
            try:
                return cls.from_file(path)
            except Exception as exc:
                logger.warning("Route registry load failed, using defaults: %s", exc)
        return cls(DEFAULT_ROUTES)
