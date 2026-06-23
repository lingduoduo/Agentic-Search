"""Metadata Filter Construction — wraps the existing LLM filter extractor."""

from __future__ import annotations

from src.internal.retrieval.query_constructor import (
    QueryConstructor as _FilterExtractor,
)

from ..route import RetrieverTarget, RouteDecision
from .base import ConstructedQuery


class MetadataFilterConstructor:
    def __init__(self, llm: object) -> None:
        self._extractor = _FilterExtractor(llm)

    def construct(self, query: str, route: RouteDecision) -> ConstructedQuery:
        cleaned, filters = self._extractor.extract_filters(query)
        return ConstructedQuery(
            target=RetrieverTarget.METADATA,
            payload={"filters": filters},
            text=cleaned,
        )
