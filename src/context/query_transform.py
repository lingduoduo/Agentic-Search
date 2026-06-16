"""QueryTransformConfig, TransformedQueryBundle, QueryTransformPipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryTransformConfig:
    decompose: bool = False
    hyde: bool = False
    step_back: bool = False
    keywords: bool = False
    construct_filters: bool = False
    max_variants: int = 5


@dataclass(frozen=True)
class TransformedQueryBundle:
    original: str
    sub_queries: list[str] = field(default_factory=list)
    hyde_text: str | None = None
    step_back: str | None = None
    keywords: list[str] = field(default_factory=list)
    merged_filters: dict = field(default_factory=dict)

    def retrieval_variants(self, max_variants: int = 5) -> list[str]:
        """Return deduplicated query variants, always including original.

        Order: sub_queries → hyde_text → step_back → keywords → original (always present).
        Truncated to max_variants total.
        """
        seen: set[str] = set()
        candidates: list[str] = []

        def _add(text: str | None) -> None:
            if text and text.lower() not in seen:
                seen.add(text.lower())
                candidates.append(text)

        for q in self.sub_queries:
            _add(q)
        _add(self.hyde_text)
        _add(self.step_back)
        for kw in self.keywords:
            _add(kw)

        original_already_in = self.original.lower() in seen
        if original_already_in:
            result = candidates[:max_variants]
        else:
            result = candidates[: max_variants - 1]
            result.append(self.original)

        return result if result else [self.original]
