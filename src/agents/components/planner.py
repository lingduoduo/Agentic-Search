"""Planner: turn the policy LM's tagged output into a typed action decision.

Recognizes the search-loop action vocabulary, including the Phase B additions:
a ``retriever`` attribute on ``<search>`` (``web`` / ``vdb``) and a ``<rerank/>``
tag. Unparseable or unknown input degrades to a safe default (a vector-DB
search), so a malformed generation never crashes the loop.

Precedence: search > rerank > answer. The model searches before it answers, so a
turn that contains both is treated as a search step.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..state import Retriever

_SEARCH_RE = re.compile(
    r"<search(?:\s+retriever=\"(?P<retriever>\w+)\")?\s*>(?P<query>.*?)</search>",
    re.DOTALL | re.IGNORECASE,
)
_RERANK_RE = re.compile(r"<rerank\s*/?>", re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(?P<text>.*?)</answer>", re.DOTALL | re.IGNORECASE)

_RETRIEVER_BY_NAME = {
    "web": Retriever.WEB,
    "vdb": Retriever.VECTOR_DB,
    "vector_db": Retriever.VECTOR_DB,
}

_FALLBACK_QUERY_MAX_CHARS = 256


def _normalize_query(query: str) -> str:
    """Whitespace- and case-insensitive key for duplicate detection."""
    return " ".join(query.split()).casefold()


@dataclass(frozen=True)
class SearchAction:
    query: str
    retriever: Retriever = Retriever.VECTOR_DB
    is_duplicate: bool = False


@dataclass(frozen=True)
class RerankAction:
    pass


@dataclass(frozen=True)
class AnswerAction:
    text: str


PlannerDecision = SearchAction | RerankAction | AnswerAction


class Planner:
    """Parse one generation step into a single typed :class:`PlannerDecision`."""

    def decide(
        self, text: str, previous_queries: Sequence[str] = ()
    ) -> PlannerDecision:
        seen = {_normalize_query(q) for q in previous_queries}
        search = _SEARCH_RE.search(text)
        if search:
            retriever = _RETRIEVER_BY_NAME.get(
                (search.group("retriever") or "").lower(), Retriever.VECTOR_DB
            )
            query = search.group("query").strip()
            return SearchAction(
                query=query,
                retriever=retriever,
                is_duplicate=_normalize_query(query) in seen,
            )

        if _RERANK_RE.search(text):
            return RerankAction()

        answer = _ANSWER_RE.search(text)
        if answer:
            return AnswerAction(text=answer.group("text").strip())

        # Safe default: a *bounded* best-effort vector-DB search on the first
        # non-empty line, so a long reasoning trace is never dumped at the retriever.
        fallback = next(
            (line.strip() for line in text.splitlines() if line.strip()), ""
        )[:_FALLBACK_QUERY_MAX_CHARS]
        return SearchAction(
            query=fallback,
            retriever=Retriever.VECTOR_DB,
            is_duplicate=_normalize_query(fallback) in seen,
        )
