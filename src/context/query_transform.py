"""QueryTransformConfig, TransformedQueryBundle, QueryTransformPipeline."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryTransformConfig:
    decompose: bool = False
    hyde: bool = False
    step_back: bool = False
    keywords: bool = False
    construct_filters: bool = False
    multi_query: bool = False
    max_variants: int = 5


def config_signature(config: QueryTransformConfig) -> str:
    """Stable string identifying which transforms are enabled (for cache keys)."""
    return "|".join(
        [
            f"d={int(config.decompose)}",
            f"h={int(config.hyde)}",
            f"s={int(config.step_back)}",
            f"k={int(config.keywords)}",
            f"c={int(config.construct_filters)}",
            f"m={int(getattr(config, 'multi_query', False))}",
            f"mv={config.max_variants}",
        ]
    )


@dataclass(frozen=True)
class TransformedQueryBundle:
    original: str
    sub_queries: list[str] = field(default_factory=list)
    hyde_text: str | None = None
    step_back: str | None = None
    keywords: list[str] = field(default_factory=list)
    merged_filters: dict = field(default_factory=dict)
    multi_query: list[str] = field(default_factory=list)

    def retrieval_variants(self, max_variants: int = 5) -> list[str]:
        """Return deduplicated query variants, always including original last.

        Order: sub_queries → hyde_text → step_back → keywords → original (always last).
        Truncated to max_variants total. original is always present.
        """
        seen: set[str] = set()
        seen.add(
            self.original.lower()
        )  # pre-seed so original is excluded from candidates
        candidates: list[str] = []

        def _add(text: str | None) -> None:
            if text and text.lower() not in seen:
                seen.add(text.lower())
                candidates.append(text)

        for q in self.sub_queries:
            _add(q)
        for q in self.multi_query:
            _add(q)
        _add(self.hyde_text)
        _add(self.step_back)
        for kw in self.keywords:
            _add(kw)

        # Reserve last slot for original, then always append it
        result = candidates[: max_variants - 1]
        result.append(self.original)
        return result


class QueryTransformPipeline:
    """Orchestrates query transformation techniques behind one interface.

    Each transformer is independently fallback-safe: on LLM failure, that
    transformer returns its empty/None default and the pipeline continues.
    """

    def __init__(self, config: QueryTransformConfig, llm: object) -> None:
        from src.context.query_enhancer import QueryEnhancer
        from src.internal.retrieval.query_constructor import QueryConstructor as QC

        self._config = config
        self._llm = llm
        self._enhancer = QueryEnhancer(llm)  # type: ignore[arg-type]
        self._constructor = QC(llm)  # type: ignore[arg-type]

    @property
    def base_config(self) -> QueryTransformConfig:
        return self._config

    @property
    def max_variants(self) -> int:
        return self._config.max_variants

    def _build_jobs(
        self, query: str, config: QueryTransformConfig
    ) -> dict[str, Callable[[], object]]:
        """Map each enabled transform to a zero-arg callable producing its field value."""
        jobs: dict[str, Callable[[], object]] = {}
        if config.decompose:
            jobs["sub_queries"] = lambda: self._enhancer.decompose(query)
        if config.hyde:
            jobs["hyde_text"] = lambda: self._enhancer.hyde(query)
        if config.step_back:
            jobs["step_back"] = lambda: self._enhancer.step_back(query)
        if config.keywords:

            def _keywords() -> object:
                from src.internal.servers.secondary_llm_flows.query_expansion import (
                    expand_keywords,
                )

                return expand_keywords(query, self._llm)  # type: ignore[arg-type]

            jobs["keywords"] = _keywords
        if config.construct_filters:
            jobs["_filters"] = lambda: self._constructor.extract_filters(query)[1]
        if config.multi_query:

            def _mq() -> object:
                from src.internal.retrieval.multi_query import MultiQueryGenerator

                return MultiQueryGenerator(
                    self._llm, n=int(os.environ.get("QT_MULTI_QUERY_N", "3"))
                ).generate(query)

            jobs["multi_query"] = _mq
        return jobs

    def _assemble(
        self, query: str, results: dict, caller_filters: dict | None
    ) -> TransformedQueryBundle:
        extracted = results.get("_filters") or {}
        return TransformedQueryBundle(
            original=query,
            sub_queries=results.get("sub_queries") or [],
            hyde_text=results.get("hyde_text"),
            step_back=results.get("step_back"),
            keywords=results.get("keywords") or [],
            merged_filters={**extracted, **(caller_filters or {})},
            multi_query=results.get("multi_query") or [],
        )

    def transform(
        self,
        query: str,
        filters: dict | None = None,
        *,
        config_override: QueryTransformConfig | None = None,
    ) -> TransformedQueryBundle:
        """Run enabled transformations and return a bundle of all query variants."""
        config = config_override or self._config
        jobs = self._build_jobs(query, config)
        results = {name: fn() for name, fn in jobs.items()}
        return self._assemble(query, results, filters)

    @classmethod
    def from_env(cls, llm: object) -> QueryTransformPipeline | None:
        """Return None if no QT_* env vars are enabled (zero overhead for callers)."""

        def _bool(name: str) -> bool:
            return os.environ.get(name, "").lower() in ("1", "true", "yes")

        def _parse_max_variants() -> int:
            try:
                return max(1, int(os.environ.get("QT_MAX_VARIANTS") or "5"))
            except (ValueError, TypeError):
                return 5

        config = QueryTransformConfig(
            decompose=_bool("QT_DECOMPOSE"),
            hyde=_bool("QT_HYDE"),
            step_back=_bool("QT_STEP_BACK"),
            keywords=_bool("QT_KEYWORDS"),
            construct_filters=_bool("QT_CONSTRUCT_FILTERS"),
            multi_query=_bool("QT_MULTI_QUERY"),
            max_variants=_parse_max_variants(),
        )

        if not any(
            [
                config.decompose,
                config.hyde,
                config.step_back,
                config.keywords,
                config.construct_filters,
                config.multi_query,
            ]
        ):
            return None

        return cls(config, llm)
