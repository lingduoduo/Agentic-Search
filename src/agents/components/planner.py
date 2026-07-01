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

from ..core.state import AgentState, Retriever

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

    @staticmethod
    def parse_actions(text: str, action_tags: Sequence[str]) -> list[tuple[str, str]]:
        tags = "|".join(re.escape(tag) for tag in action_tags)
        action_re = re.compile(
            rf"<({tags})(?:\s+[^>]*)?>(.*?)</\1>",
            re.DOTALL,
        )
        return [
            (match.group(1), match.group(2).strip())
            for match in action_re.finditer(text)
        ]

    @staticmethod
    def round_retriever(text: str, search_tags: Sequence[str]) -> Retriever:
        tags = "|".join(re.escape(tag) for tag in search_tags)
        match = re.search(
            rf'<(?:{tags})\s+[^>]*retriever="(?P<retriever>\w+)"',
            text,
            re.IGNORECASE,
        )
        if match and match.group("retriever").lower() == "web":
            return Retriever.WEB
        return Retriever.VECTOR_DB

    @staticmethod
    def round_rerank(text: str, search_tags: Sequence[str]) -> bool:
        tags = "|".join(re.escape(tag) for tag in search_tags)
        match = re.search(
            rf'<(?:{tags})\s+[^>]*rerank="(?P<rerank>\w+)"',
            text,
            re.IGNORECASE,
        )
        return bool(match and match.group("rerank").lower() == "true")

    @staticmethod
    def partition_search_requests(
        query_specs: list[tuple[str | None, str]],
        state: AgentState,
        effective_limit: int,
    ) -> tuple[list[tuple[str | None, str]], list[str], list[str]]:
        seen = {_normalize_query(query) for query in state.previous_queries}
        at_limit = effective_limit > 0 and state.search_rounds >= effective_limit
        allowed: list[tuple[str | None, str]] = []
        repeated: list[str] = []
        overflow: list[str] = []
        for task_id, query in query_specs:
            if _normalize_query(query) in seen:
                repeated.append(query)
            elif at_limit:
                overflow.append(query)
            else:
                allowed.append((task_id, query))
        return allowed, repeated, overflow

    def decide(self, text: str, state: AgentState) -> PlannerDecision:
        seen = {_normalize_query(q) for q in state.previous_queries}
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
