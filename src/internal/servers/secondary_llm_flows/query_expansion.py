"""LLM-backed keyword expansion for BM25 search."""

from __future__ import annotations

import logging
import re
from datetime import datetime

from src.context.models import ChatMessage
from src.context.models import LLMClient
from src.context.models import LLMResponse
from src.internal.prompts.query_expansion import KEYWORD_EXPANSION_PROMPT

logger = logging.getLogger(__name__)

_ARTIFACT_RE = re.compile(r'[\[\]"\'`]')
_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+[.)]\s*|[-*]\s*)")

_TEMPORAL_SINGLE_WORDS = frozenset(
    {
        "latest",
        "recent",
        "current",
        "today",
        "now",
        "newest",
        "new",
        "updated",
        "2024",
        "2025",
        "2026",
    }
)
_TEMPORAL_PHRASES = frozenset({"this year", "this month"})
_TEMPORAL_WORD_RE = re.compile(
    r"\b("
    + "|".join(
        re.escape(w) for w in sorted(_TEMPORAL_SINGLE_WORDS, key=len, reverse=True)
    )
    + r")\b",
    re.IGNORECASE,
)


def is_time_sensitive(query: str) -> bool:
    """Return True when the query contains temporal keywords that suggest recency matters."""
    if _TEMPORAL_WORD_RE.search(query):
        return True
    lower = query.lower()
    return any(phrase in lower for phrase in _TEMPORAL_PHRASES)


def with_temporal_context(query: str) -> str:
    """Append current month and year to time-sensitive queries; return query unchanged otherwise."""
    if not is_time_sensitive(query):
        return query
    date_str = datetime.now().strftime("%B %Y")
    return f"{query} {date_str}"


def _clean_keyword_line(line: str) -> str:
    cleaned = _ARTIFACT_RE.sub("", line)
    cleaned = _LIST_MARKER_RE.sub("", cleaned)
    return cleaned.strip()


def _llm_text(response: LLMResponse | str) -> str:
    return response if isinstance(response, str) else response.content


def expand_keywords(user_query: str, llm: LLMClient) -> list[str]:
    """Expand a user query into keyword-only queries suitable for BM25 search.

    Returns expanded queries only — the original query is excluded.
    """
    prompt = KEYWORD_EXPANSION_PROMPT.format(user_query=user_query)
    try:
        text = _llm_text(
            llm.complete([ChatMessage(role="user", content=prompt)])
        ).strip()
    except Exception as exc:
        logger.warning("Keyword expansion failed: %s", exc)
        return []

    if not text:
        logger.warning("Keyword expansion returned empty response.")
        return []

    seen_lower: set[str] = {user_query.lower()}
    expanded: list[str] = []
    for line in text.splitlines():
        cleaned = _clean_keyword_line(line)
        if cleaned and cleaned.lower() not in seen_lower:
            seen_lower.add(cleaned.lower())
            expanded.append(cleaned)

    if not expanded:
        logger.warning("Keyword expansion produced no usable queries.")
    else:
        logger.debug("Keyword expansion generated %d queries.", len(expanded))

    return expanded


__all__ = ["expand_keywords", "is_time_sensitive", "with_temporal_context"]
