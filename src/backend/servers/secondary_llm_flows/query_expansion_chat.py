"""Chat-history-aware query rephrasing and keyword expansion."""

from __future__ import annotations

import logging
from datetime import datetime
from datetime import timezone

from src.backend.chat.models import ChatMessageSimple
from src.backend.configs.constants import MessageType
from src.backend.llm.interfaces import LLM
from src.backend.llm.models import AssistantMessage
from src.backend.llm.models import ChatCompletionMessage
from src.backend.llm.models import ReasoningEffort
from src.backend.llm.models import SystemMessage
from src.backend.llm.models import UserMessage
from src.backend.prompts.query_expansion import KEYWORD_REPHRASE_SYSTEM_PROMPT
from src.backend.prompts.query_expansion import KEYWORD_REPHRASE_USER_PROMPT
from src.backend.prompts.query_expansion import REPHRASE_CONTEXT_PROMPT
from src.backend.prompts.query_expansion import SEMANTIC_QUERY_REPHRASE_SYSTEM_PROMPT
from src.backend.prompts.query_expansion import SEMANTIC_QUERY_REPHRASE_USER_PROMPT

logger = logging.getLogger(__name__)


def _get_current_date_str() -> str:
    return datetime.now(tz=timezone.utc).strftime("%A, %B %d, %Y")


def _build_additional_context(
    user_info: str | None = None,
    memories: list[str] | None = None,
) -> str:
    has_user_info = user_info and user_info.strip()
    has_memories = memories and any(m.strip() for m in memories)

    if not has_user_info and not has_memories:
        return ""

    formatted_user_info = user_info if has_user_info else "N/A"
    formatted_memories = (
        "\n".join(f"- {memory}" for memory in memories)
        if has_memories and memories
        else "N/A"
    )

    return REPHRASE_CONTEXT_PROMPT.format(
        user_info=formatted_user_info,
        memories=formatted_memories,
    )


def _build_message_history(
    history: list[ChatMessageSimple],
) -> list[ChatCompletionMessage]:
    messages: list[ChatCompletionMessage] = []

    for msg in history:
        if msg.message_type == MessageType.USER:
            messages.append(UserMessage(content=msg.message))
        elif msg.message_type == MessageType.ASSISTANT:
            messages.append(AssistantMessage(content=msg.message))

    return messages


def semantic_query_rephrase(
    history: list[ChatMessageSimple],
    llm: LLM,
    user_info: str | None = None,
    memories: list[str] | None = None,
) -> str:
    """Rephrase a query into a standalone query using chat history context.

    Raises:
        ValueError: If history is empty or contains no user messages.
        RuntimeError: If LLM fails to generate a rephrased query.
    """
    if not history:
        raise ValueError("History cannot be empty for query rephrasing")

    last_user_message_idx = None
    for i in range(len(history) - 1, -1, -1):
        if history[i].message_type == MessageType.USER:
            last_user_message_idx = i
            break

    if last_user_message_idx is None:
        raise ValueError("History must contain at least one user message")

    user_query = history[last_user_message_idx].message
    additional_context = _build_additional_context(user_info, memories)

    system_msg = SystemMessage(
        content=SEMANTIC_QUERY_REPHRASE_SYSTEM_PROMPT.format(
            current_date=_get_current_date_str()
        )
    )

    messages: list[ChatCompletionMessage] = [system_msg]
    messages.extend(_build_message_history(history[:last_user_message_idx]))
    messages.append(
        UserMessage(
            content=SEMANTIC_QUERY_REPHRASE_USER_PROMPT.format(
                additional_context=additional_context, user_query=user_query
            )
        )
    )

    response = llm.invoke(prompt=messages, reasoning_effort=ReasoningEffort.OFF)
    final_query = response.choice.message.content

    if not final_query:
        raise RuntimeError("LLM failed to generate a rephrased query")

    return final_query


def keyword_query_expansion(
    history: list[ChatMessageSimple],
    llm: LLM,
    user_info: str | None = None,
    memories: list[str] | None = None,
) -> list[str] | None:
    """Expand a query into multiple keyword-only queries using chat history context.

    Raises:
        ValueError: If history is empty or contains no user messages.
    """
    if not history:
        raise ValueError("History cannot be empty for keyword query expansion")

    last_user_message_idx = None
    for i in range(len(history) - 1, -1, -1):
        if history[i].message_type == MessageType.USER:
            last_user_message_idx = i
            break

    if last_user_message_idx is None:
        raise ValueError("History must contain at least one user message")

    user_query = history[last_user_message_idx].message
    additional_context = _build_additional_context(user_info, memories)

    system_msg = SystemMessage(
        content=KEYWORD_REPHRASE_SYSTEM_PROMPT.format(
            current_date=_get_current_date_str()
        )
    )

    messages: list[ChatCompletionMessage] = [system_msg]
    messages.extend(_build_message_history(history[:last_user_message_idx]))
    messages.append(
        UserMessage(
            content=KEYWORD_REPHRASE_USER_PROMPT.format(
                additional_context=additional_context, user_query=user_query
            )
        )
    )

    response = llm.invoke(prompt=messages, reasoning_effort=ReasoningEffort.OFF)
    content = response.choice.message.content

    if not content:
        return []

    return [line.strip() for line in content.strip().split("\n") if line.strip()]
