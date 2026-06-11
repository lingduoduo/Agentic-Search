"""Streaming chat orchestrator.

Produces an NDJSON stream of typed packets:
  SearchQueriesPacket  — retrieval queries sent
  SearchDocsPacket     — retrieved documents
  SearchToolStart      — tool invocation starting (when tool_names provided)
  SearchToolQueriesDelta  — tool names being called
  SearchToolDocumentsDelta — tool results as doc cards
  AgentResponseStart   — LLM generation beginning
  AgentResponseDelta   — incremental token
  OverallStop          — stream complete
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from src.backend.db import AgenticSearchStore
from src.context.history import compress_history
from src.context.models import (
    ChatMessage,
    ContextDocument,
    SearchContextBundle,
    SearchFilters,
)
from src.context.pipeline import retrieve_context, synthesize_answer_from_context
from src.context.prompts import build_chat_prompt
from .models import SearchDocWithContent
from .streaming_models import (
    AgentResponseDelta,
    AgentResponseStart,
    OverallStop,
    PacketException,
    SearchDocsPacket,
    SearchQueriesPacket,
    SearchToolDocumentsDelta,
    SearchToolQueriesDelta,
    SearchToolStart,
)

logger = logging.getLogger(__name__)

_MAX_HISTORY_TOKENS = 4000
_KEEP_RECENT = 6


def _doc_to_view(doc: ContextDocument) -> SearchDocWithContent:
    return SearchDocWithContent(
        title=doc.title,
        url=doc.url,
        content=doc.content,
        score=float(doc.score or 0.0),
    )


def _emit(packet: Any) -> str:
    return json.dumps(packet.model_dump()) + "\n"


async def _iter_llm_stream(llm: Any, messages: list[ChatMessage]) -> AsyncIterator[str]:
    """Wrap synchronous LLM.stream() into an async generator via a thread."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | Exception | None] = asyncio.Queue(maxsize=100)

    def _producer() -> None:
        try:
            for chunk in llm.stream(messages):
                token = chunk.choice.delta.content
                if token:
                    loop.call_soon_threadsafe(queue.put_nowait, token)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    loop.run_in_executor(None, _producer)

    while True:
        item = await queue.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item


async def stream_chat_response(
    query: str,
    *,
    session_id: str,
    llm: Any | None,
    db: AgenticSearchStore,
    search_url: str,
    top_k: int = 5,
    file_content: str | None = None,
    tool_names: list[str] | None = None,
    filters: SearchFilters | None = None,
) -> AsyncIterator[str]:
    """Async generator that yields NDJSON lines for the streaming chat endpoint."""

    # ── 1. Load and compress history ────────────────────────────────────────
    raw_messages = db.list_chat_messages(session_id)
    history = [ChatMessage(role=m.role, content=m.content) for m in raw_messages]
    history = compress_history(
        history, llm, max_tokens=_MAX_HISTORY_TOKENS, keep_recent=_KEEP_RECENT
    )

    # ── 2. Persist user message now (before retrieval) ───────────────────────
    db.add_chat_message(session_id, role="user", content=query)

    # ── 3. Build retrieval query (include file hint if provided) ────────────
    retrieval_query = query
    if file_content:
        retrieval_query = f"{query}\n\n[Attached file excerpt: {file_content[:500]}]"

    yield _emit(SearchQueriesPacket(all_executed_queries=[retrieval_query]))

    # ── 4. Retrieve documents ────────────────────────────────────────────────
    try:
        context: SearchContextBundle = await retrieve_context(
            retrieval_query,
            search_url=search_url,
            top_k=top_k,
            filters=filters,
        )
    except Exception as exc:
        logger.error("stream_chat: retrieval failed: %s", exc)
        yield _emit(PacketException(message=f"Retrieval failed: {exc}"))
        yield _emit(OverallStop())
        return

    doc_views = [_doc_to_view(d) for d in context.documents]
    yield _emit(SearchDocsPacket(search_docs=doc_views))

    # ── 5. Tool invocations ──────────────────────────────────────────────────
    extra_context_parts: list[str] = []
    if tool_names:
        from src.tools import tool_registry  # lazy to avoid circular import

        yield _emit(SearchToolStart(is_internet_search=False))
        yield _emit(SearchToolQueriesDelta(queries=tool_names))

        tool_result_docs: list[SearchDocWithContent] = []
        for name in tool_names:
            try:
                response, _raw, errors = await tool_registry.invoke(
                    name, {"query": query}
                )
                if not errors and response:
                    extra_context_parts.append(f"[Tool '{name}' result]\n{response}")
                    tool_result_docs.append(
                        SearchDocWithContent(
                            title=f"Tool: {name}",
                            content=str(response)[:1000],
                            score=0.0,
                        )
                    )
            except Exception as exc:
                logger.warning("stream_chat: tool %r failed: %s", name, exc)

        if tool_result_docs:
            yield _emit(SearchToolDocumentsDelta(documents=tool_result_docs))

    # ── 6. Build prompt ──────────────────────────────────────────────────────
    # Inject file content and tool results into the question for the LLM
    augmented_question = query
    if file_content:
        augmented_question = f"{query}\n\n[File context]\n{file_content[:2000]}"
    if extra_context_parts:
        augmented_question += "\n\n" + "\n\n".join(extra_context_parts)

    prompt = build_chat_prompt(augmented_question, context, history=history)

    # ── 7. Stream LLM answer ─────────────────────────────────────────────────
    yield _emit(AgentResponseStart(final_documents=doc_views))

    answer_tokens: list[str] = []
    can_stream = llm is not None and hasattr(llm, "stream")

    try:
        if can_stream:
            async for token in _iter_llm_stream(llm, prompt.messages):
                answer_tokens.append(token)
                yield _emit(AgentResponseDelta(content=token))
        elif llm is not None:
            raw = llm.complete(prompt.messages)
            text = raw.text if hasattr(raw, "text") else str(raw)
            answer_tokens.append(text)
            yield _emit(AgentResponseDelta(content=text))
        else:
            text = synthesize_answer_from_context(query, context)
            answer_tokens.append(text)
            yield _emit(AgentResponseDelta(content=text))
    except Exception as exc:
        logger.error("stream_chat: LLM generation failed: %s", exc)
        fallback = synthesize_answer_from_context(query, context)
        answer_tokens.append(fallback)
        yield _emit(AgentResponseDelta(content=fallback))

    yield _emit(OverallStop())

    # ── 8. Persist assistant message ─────────────────────────────────────────
    answer = "".join(answer_tokens)
    from src.context.utils import extract_citations

    db.add_chat_message(
        session_id,
        role="assistant",
        content=answer,
        metadata={
            "citations": extract_citations(answer),
            "document_ids": [d.id for d in context.documents],
            "mode": "chat_stream",
            "has_file": bool(file_content),
            "tools_used": tool_names or [],
        },
    )
