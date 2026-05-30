"""LLM-backed search-vs-chat flow classification."""

from __future__ import annotations

import logging
import time

from src.context.models import ChatMessage
from src.context.models import LLMClient
from src.context.models import LLMResponse
from src.backend.prompts.search_flow_classification import CHAT_CLASS
from src.backend.prompts.search_flow_classification import SEARCH_CHAT_PROMPT
from src.backend.prompts.search_flow_classification import SEARCH_CLASS

logger = logging.getLogger(__name__)


def _llm_text(response: LLMResponse | str) -> str:
    return response if isinstance(response, str) else response.content


def classify_is_search_flow(query: str, llm: LLMClient) -> bool:
    """Return True if the query should be routed to document search, False for chat.

    The function defaults to False (chat) whenever the response is empty,
    unexpected, or contains both class names — chat is the safer fallback
    because a mislabelled search query is more disruptive than a mislabelled
    chat query.
    """
    t0 = time.monotonic()
    prompt = SEARCH_CHAT_PROMPT.format(user_query=query)
    content = (
        _llm_text(llm.complete([ChatMessage(role="user", content=prompt)]))
        .strip()
        .lower()
    )
    logger.debug(
        "Search flow classification took %dms.", int((time.monotonic() - t0) * 1000)
    )

    if not content:
        logger.warning(
            "Search flow classification returned empty response; defaulting to chat."
        )
        return False

    # When both labels appear prefer chat — it is the safer default.
    if CHAT_CLASS in content:
        return False
    if SEARCH_CLASS in content:
        return True

    logger.warning(
        "Search flow classification returned unexpected response %r; defaulting to chat.",
        content,
    )
    return False


__all__ = ["classify_is_search_flow"]
