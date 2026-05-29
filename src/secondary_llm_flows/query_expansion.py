"""LLM-backed keyword expansion for BM25 search."""

from __future__ import annotations

import logging
import re

from src.context.models import ChatMessage
from src.context.models import LLMClient
from src.context.models import LLMResponse
from src.prompts.query_expansion import KEYWORD_EXPANSION_PROMPT

logger = logging.getLogger(__name__)

# Strips brackets, quotes, and backticks that LLMs often add unintentionally.
_ARTIFACT_RE = re.compile(r'[\[\]"\'`]')
# Strips leading list markers: "1.", "2)", "-", "*"
_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+[.)]\s*|[-*]\s*)")


def _clean_keyword_line(line: str) -> str:
    cleaned = _ARTIFACT_RE.sub("", line)
    cleaned = _LIST_MARKER_RE.sub("", cleaned)
    return cleaned.strip()


def _llm_text(response: LLMResponse | str) -> str:
    return response if isinstance(response, str) else response.content


def expand_keywords(user_query: str, llm: LLMClient) -> list[str]:
    """Expand a user query into keyword-only queries suitable for BM25 search.

    Returns expanded queries only — the original query is excluded. Returns an
    empty list when the LLM call fails, the response is empty, or all lines
    duplicate the original query.
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


__all__ = ["expand_keywords"]
