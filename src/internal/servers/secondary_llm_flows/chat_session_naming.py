from __future__ import annotations

import logging

from src.internal.chat.llm_step import translate_history_to_llm_format
from src.internal.chat.models import ChatMessageSimple
from src.internal.configs.constants import MessageType
from src.internal.llm.interfaces import LLM
from src.internal.llm.models import ReasoningEffort
from src.internal.llm.utils import llm_response_to_string
from src.internal.prompts.chat_prompts import CHAT_NAMING_REMINDER
from src.internal.prompts.chat_prompts import CHAT_NAMING_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def generate_chat_session_name(
    chat_history: list[ChatMessageSimple],
    llm: LLM,
) -> str:
    system_prompt = ChatMessageSimple(
        message=CHAT_NAMING_SYSTEM_PROMPT,
        token_count=100,
        message_type=MessageType.SYSTEM,
    )

    reminder_prompt = ChatMessageSimple(
        message=CHAT_NAMING_REMINDER,
        token_count=100,
        message_type=MessageType.USER_REMINDER,
    )

    complete_message_history = [system_prompt] + chat_history + [reminder_prompt]

    llm_facing_history = translate_history_to_llm_format(
        complete_message_history, llm.config
    )

    response = llm.invoke(llm_facing_history, reasoning_effort=ReasoningEffort.OFF)
    new_name_raw = llm_response_to_string(response)

    return new_name_raw.strip().strip('"')
