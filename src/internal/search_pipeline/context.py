"""Deterministic session context construction for document retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from src.context import ChatMessage


_FOLLOW_UP_PREFIX = re.compile(
    r"^(?:and\b|also\b|but\b|what about\b|how about\b|tell me more\b|"
    r"go on\b|continue\b|why(?:\s|\?|$))",
    re.IGNORECASE,
)
_REFERENCE_PRONOUN = re.compile(
    r"\b(?:it|its|they|them|their|this|that|these|those|he|him|his|she|her)\b",
    re.IGNORECASE,
)
_ASSISTANT_INTERNAL_MARKUP = re.compile(
    r"<\s*/?\s*(?:tool(?:_call|_result)?|evidence|search_results?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RetrievalContext:
    """Original request plus the bounded session context used for retrieval."""

    query: str
    retrieval_query: str
    history: list[ChatMessage]


def _is_follow_up(query: str) -> bool:
    normalized = query.strip()
    return bool(
        _FOLLOW_UP_PREFIX.search(normalized) or _REFERENCE_PRONOUN.search(normalized)
    )


def _safe_history(history: Iterable[ChatMessage]) -> list[ChatMessage]:
    return [
        message
        for message in history
        if not (
            message.role.lower() == "assistant"
            and _ASSISTANT_INTERNAL_MARKUP.search(message.content)
        )
    ]


def _most_recent_user_topic(history: list[ChatMessage]) -> str | None:
    for message in reversed(history):
        if (
            message.role.lower() == "user"
            and message.content.strip()
            and not _is_follow_up(message.content)
        ):
            return message.content
    return None


def build_retrieval_context(
    query: str,
    history: Iterable[ChatMessage],
    max_messages: int = 6,
) -> RetrievalContext:
    """Build bounded history and resolve simple follow-ups without an LLM."""

    if max_messages < 0:
        raise ValueError("max_messages must be non-negative")

    safe_history = _safe_history(history)
    bounded_history = safe_history[-max_messages:] if max_messages else []
    retrieval_query = query
    if _is_follow_up(query):
        topic = _most_recent_user_topic(bounded_history)
        if topic is not None:
            retrieval_query = f"{topic}\n{query}"

    return RetrievalContext(
        query=query,
        retrieval_query=retrieval_query,
        history=bounded_history,
    )
