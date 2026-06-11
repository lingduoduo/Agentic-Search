from __future__ import annotations

import json
import logging
import re

from src.internal.chat.models import ChatMessageSimple
from src.internal.configs.constants import MessageType
from src.internal.llm.interfaces import LLM
from src.internal.llm.models import ReasoningEffort
from src.internal.llm.models import UserMessage
from src.internal.prompts.basic_memory import FULL_MEMORY_UPDATE_PROMPT

logger = logging.getLogger(__name__)

MAX_USER_MESSAGES = 3
MAX_CHARS_PER_MESSAGE = 500


def _parse_llm_json_response(content: str) -> dict | None:
    content = content.strip()
    try:
        result = json.loads(content)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError:
            pass
    return None


def _format_chat_history(chat_history: list[ChatMessageSimple]) -> str:
    user_messages = [
        msg for msg in chat_history if msg.message_type == MessageType.USER
    ]

    if not user_messages:
        return "No chat history available."

    recent_user_messages = user_messages[-MAX_USER_MESSAGES:]

    formatted_parts = []
    for msg in recent_user_messages:
        if len(msg.message) > MAX_CHARS_PER_MESSAGE:
            truncated_message = msg.message[:MAX_CHARS_PER_MESSAGE] + "[...truncated]"
        else:
            truncated_message = msg.message
        formatted_parts.append(f"\nUser message:\n{truncated_message}\n")

    return "".join(formatted_parts).strip()


def _format_existing_memories(existing_memories: list[str]) -> str:
    if not existing_memories:
        return "No existing memories."

    return "\n".join(
        f"{i}. {memory}" for i, memory in enumerate(existing_memories, start=1)
    )


def _format_user_basic_information(
    user_name: str | None,
    user_email: str | None,
    user_role: str | None,
) -> str:
    lines = []
    if user_name:
        lines.append(f"User name: {user_name}")
    if user_email:
        lines.append(f"User email: {user_email}")
    if user_role:
        lines.append(f"User role: {user_role}")

    if not lines:
        return ""

    return "\n\n# User Basic Information\n" + "\n".join(lines)


def process_memory_update(
    new_memory: str,
    existing_memories: list[str],
    chat_history: list[ChatMessageSimple],
    llm: LLM,
    user_name: str | None = None,
    user_email: str | None = None,
    user_role: str | None = None,
) -> tuple[str, int | None]:
    """Determine if a memory should be added or updated.

    Returns:
        Tuple of (memory_text, index_to_replace) where index_to_replace is None for add.
    """
    formatted_chat_history = _format_chat_history(chat_history)
    formatted_memories = _format_existing_memories(existing_memories)
    formatted_user_info = _format_user_basic_information(
        user_name, user_email, user_role
    )

    prompt = FULL_MEMORY_UPDATE_PROMPT.format(
        chat_history=formatted_chat_history,
        user_basic_information=formatted_user_info,
        existing_memories=formatted_memories,
        new_memory=new_memory,
    )

    try:
        prompt_msg = UserMessage(content=prompt)
        response = llm.invoke(prompt=prompt_msg, reasoning_effort=ReasoningEffort.OFF)
        content = response.choice.message.content
    except Exception as e:
        logger.warning("LLM invocation failed for memory update: %s", e)
        return (new_memory, None)

    if not content:
        logger.warning(
            "LLM returned empty response for memory update, defaulting to add"
        )
        return (new_memory, None)

    parsed_response = _parse_llm_json_response(content)

    if not parsed_response:
        logger.warning(
            "Failed to parse JSON from LLM response: %s..., defaulting to add",
            content[:200],
        )
        return (new_memory, None)

    operation = parsed_response.get("operation", "add").lower()
    memory_id = parsed_response.get("memory_id")
    memory_text = parsed_response.get("memory_text", new_memory)

    if not memory_text or not isinstance(memory_text, str):
        memory_text = new_memory

    if operation == "add":
        logger.debug("Memory update operation: add")
        return (memory_text, None)

    if operation == "update":
        if memory_id is None:
            logger.warning("Update operation specified but no memory_id provided")
            return (memory_text, None)

        try:
            memory_id_int = int(memory_id)
        except (ValueError, TypeError):
            logger.warning("Invalid memory_id format: %s", memory_id)
            return (memory_text, None)

        index_to_replace = memory_id_int - 1

        if index_to_replace < 0 or index_to_replace >= len(existing_memories):
            logger.warning(
                "memory_id %s out of range (1-%s), defaulting to add",
                memory_id_int,
                len(existing_memories),
            )
            return (memory_text, None)

        logger.debug("Memory update operation: update at index %s", index_to_replace)
        return (memory_text, index_to_replace)

    logger.warning("Unknown operation '%s', defaulting to add", operation)
    return (memory_text, None)
