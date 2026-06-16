"""QueryTransformConfig, TransformedQueryBundle, QueryTransformPipeline."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.internal.servers.secondary_llm_flows.query_expansion import expand_keywords

if TYPE_CHECKING:
    from src.internal.retrieval.query_constructor import QueryConstructor

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


class QueryTransformPipeline:
    """Orchestrates query transformation techniques behind one interface.

    Each transformer is independently fallback-safe: on LLM failure, that
    transformer returns its empty/None default and the pipeline continues.
    """

    def __init__(self, config: QueryTransformConfig, llm: object) -> None:
        from src.context.query_enhancer import QueryEnhancer

        self._config = config
        self._llm = llm
        self._enhancer = QueryEnhancer(llm)  # type: ignore[arg-type]
        self._constructor: QueryConstructor | None = None
        if config.construct_filters:
            from src.internal.retrieval.query_constructor import QueryConstructor as QC

            self._constructor = QC(llm)  # type: ignore[arg-type]

    def transform(
        self,
        query: str,
        filters: dict | None = None,
    ) -> TransformedQueryBundle:
        """Run enabled transformations and return a bundle of all query variants."""
        sub_queries: list[str] = []
        hyde_text: str | None = None
        step_back_q: str | None = None
        keywords: list[str] = []
        extracted_filters: dict = {}

        if self._config.decompose:
            sub_queries = self._enhancer.decompose(query)
        if self._config.hyde:
            hyde_text = self._enhancer.hyde(query)
        if self._config.step_back:
            step_back_q = self._enhancer.step_back(query)
        if self._config.keywords:
            keywords = expand_keywords(query, self._llm)  # type: ignore[arg-type]
        if self._config.construct_filters and self._constructor is not None:
            _, extracted_filters = self._constructor.extract_filters(query)

        # Caller-supplied filters win on key conflict.
        merged_filters: dict = {**extracted_filters, **(filters or {})}

        return TransformedQueryBundle(
            original=query,
            sub_queries=sub_queries,
            hyde_text=hyde_text,
            step_back=step_back_q,
            keywords=keywords,
            merged_filters=merged_filters,
        )

    @classmethod
    def from_env(cls, llm: object) -> QueryTransformPipeline | None:
        """Return None if no QT_* env vars are enabled (zero overhead for callers)."""

        def _bool(name: str) -> bool:
            return os.environ.get(name, "").lower() in ("1", "true", "yes")

        config = QueryTransformConfig(
            decompose=_bool("QT_DECOMPOSE"),
            hyde=_bool("QT_HYDE"),
            step_back=_bool("QT_STEP_BACK"),
            keywords=_bool("QT_KEYWORDS"),
            construct_filters=_bool("QT_CONSTRUCT_FILTERS"),
            max_variants=int(os.environ.get("QT_MAX_VARIANTS", "5")),
        )

        if not any(
            [
                config.decompose,
                config.hyde,
                config.step_back,
                config.keywords,
                config.construct_filters,
            ]
        ):
            return None

        return cls(config, llm)
