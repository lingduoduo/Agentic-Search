"""Query-construction interface: NL query → backend-specific structured query."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..route import RetrieverTarget, RouteDecision


@dataclass(frozen=True)
class ConstructedQuery:
    target: RetrieverTarget
    payload: dict = field(default_factory=dict)
    text: str | None = None


class QueryConstructor(Protocol):
    def construct(self, query: str, route: RouteDecision) -> ConstructedQuery:
        """Build the backend query. Must not raise — degrade to empty payload."""
        ...
