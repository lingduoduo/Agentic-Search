"""Query enhancement for best-in-class RAG: decomposition and HyDE."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .models import ChatMessage, LLMClient

logger = logging.getLogger(__name__)

_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+[.)]\s*|[-*•]\s*)")
_ARTIFACT_RE = re.compile(r'[\[\]"\'`]')

_DECOMPOSE_PROMPT = """Break the following question into 2-4 focused sub-questions that together cover the full intent.
Return one sub-question per line. Do not number them. Do not include the original question.
If the question is already simple, return it unchanged on a single line.

Question: {query}""".strip()

_HYDE_PROMPT = """Write a single concise paragraph that would be the ideal answer to the following question.
Write as if you are an expert. Be factual and specific. Do not say "I" or ask follow-up questions.

Question: {query}
Answer:""".strip()


def _clean_line(line: str) -> str:
    return _ARTIFACT_RE.sub("", _LIST_MARKER_RE.sub("", line)).strip()


def _llm_text(response: object) -> str:
    if hasattr(response, "text"):
        return response.text
    if hasattr(response, "content"):
        return response.content
    return str(response)


@dataclass
class QueryBundle:
    original: str
    sub_queries: list[str] = field(default_factory=list)
    hyde_text: str | None = None

    def all_queries(self) -> list[str]:
        """Return deduplicated list: sub_queries first, then hyde_text appended."""
        seen: set[str] = set()
        result: list[str] = []
        for q in self.sub_queries:
            if q not in seen:
                seen.add(q)
                result.append(q)
        if self.hyde_text and self.hyde_text not in seen:
            result.append(self.hyde_text)
        return result or [self.original]


class QueryEnhancer:
    """Enhance a query with LLM-driven decomposition and HyDE.

    Both strategies degrade gracefully: when no LLM is provided, or on any
    LLM failure, the original query is returned unchanged so the retrieval
    pipeline is never blocked.
    """

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm

    def decompose(self, query: str) -> list[str]:
        """Break query into focused sub-queries.  Falls back to [query] on failure."""
        if self.llm is None:
            return [query]
        try:
            raw = _llm_text(
                self.llm.complete(
                    [
                        ChatMessage(
                            role="user", content=_DECOMPOSE_PROMPT.format(query=query)
                        )
                    ]
                )
            ).strip()
        except Exception as exc:
            logger.warning("Query decomposition failed: %s", exc)
            return [query]

        if not raw:
            return [query]

        original_lower = query.lower()
        seen: set[str] = {original_lower}
        result: list[str] = []
        for line in raw.splitlines():
            cleaned = _clean_line(line)
            if cleaned and cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                result.append(cleaned)

        return result if result else [query]

    def hyde(self, query: str) -> str | None:
        """Generate a hypothetical answer for HyDE dense retrieval.  Returns None on failure."""
        if self.llm is None:
            return None
        try:
            raw = _llm_text(
                self.llm.complete(
                    [ChatMessage(role="user", content=_HYDE_PROMPT.format(query=query))]
                )
            ).strip()
            return raw or None
        except Exception as exc:
            logger.warning("HyDE generation failed: %s", exc)
            return None

    def enhance(self, query: str) -> QueryBundle:
        """Run decompose + HyDE and return a QueryBundle."""
        return QueryBundle(
            original=query,
            sub_queries=self.decompose(query),
            hyde_text=self.hyde(query),
        )
