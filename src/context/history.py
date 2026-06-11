"""History token estimation and compression for chat sessions."""

from __future__ import annotations

import logging

from .models import ChatMessage, LLMClient

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Rough token count: ~4 characters per token."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def compress_history(
    messages: list[ChatMessage],
    llm: LLMClient | None = None,
    *,
    max_tokens: int = 4000,
    keep_recent: int = 6,
) -> list[ChatMessage]:
    """Return history trimmed to fit within `max_tokens`.

    Keeps the last `keep_recent` messages verbatim.  If the remainder is still
    over budget, summarises the older messages via `llm` (or drops them when no
    LLM is available).
    """
    total = sum(estimate_tokens(m.content) for m in messages)
    if total <= max_tokens:
        return messages

    if len(messages) <= keep_recent:
        return messages

    old = messages[:-keep_recent]
    recent = messages[-keep_recent:]

    if llm is None:
        logger.debug(
            "compress_history: dropping %d old messages (no LLM for summarisation)",
            len(old),
        )
        return recent

    text_to_summarize = "\n".join(f"{m.role}: {m.content[:600]}" for m in old)
    summary_prompt = [
        ChatMessage(
            role="user",
            content=(
                "Summarise the following conversation history in 2-3 concise sentences, "
                "preserving the key facts and decisions:\n\n"
                f"{text_to_summarize}"
            ),
        )
    ]
    try:
        result = llm.complete(summary_prompt)
        summary_text = result.text if hasattr(result, "text") else str(result)
        summary_msg = ChatMessage(
            role="system",
            content=f"[Earlier conversation summary: {summary_text}]",
        )
        logger.debug(
            "compress_history: compressed %d old messages into a summary", len(old)
        )
        return [summary_msg] + recent
    except Exception:
        logger.warning(
            "compress_history: LLM summarisation failed, dropping old messages"
        )
        return recent
