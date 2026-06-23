"""Semantic routing: pick the route whose description is nearest to the query."""

from __future__ import annotations

import math

from .route import Route


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def cosine_route(query: str, routes: list[Route], embedder) -> tuple[Route, float]:
    """Return (best_route, score). Embeds [query, *descriptions] in one call."""
    texts = [query] + [r.description for r in routes]
    vectors = embedder(texts)
    q_vec, route_vecs = vectors[0], vectors[1:]
    best_idx, best_score = 0, -1.0
    for i, vec in enumerate(route_vecs):
        score = _cosine(q_vec, vec)
        if score > best_score:
            best_idx, best_score = i, score
    return routes[best_idx], best_score
