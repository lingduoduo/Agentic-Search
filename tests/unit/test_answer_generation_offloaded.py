"""Answer synthesis must not run on the event loop.

`generate_answer` is synchronous and calls `llm.complete`, which is a blocking
`requests` call. Awaiting it directly from an async request handler freezes the
entire event loop for the whole completion -- seconds, not milliseconds -- so
every other in-flight session's stream stalls behind one user's answer.

Both async answer paths had this:

    src/context/pipeline.py       answer_with_retrieval   (classic RAG route)
    src/agents/search/agentic_rag.py  AgenticRAGLoop.run  (Assist CHAT route)

`agentic_rag.py` already offloaded its two *cheap* utility calls -- the
sufficiency check and the gap analysis -- with `asyncio.to_thread`. Only the
single most expensive call, the answer itself, ran unprotected.

These tests assert on THREAD IDENTITY rather than timing: "did this run on the
event-loop thread" is the actual invariant, and it cannot flake the way a
duration threshold can.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from src.context.models import ChatMessage, ContextDocument, SearchContextBundle


def _bundle(query: str = "q") -> SearchContextBundle:
    """A bundle with real evidence.

    Non-empty on purpose: `generate_answer` short-circuits without ever calling
    the LLM when there is no evidence, so an empty bundle would make these tests
    pass for the wrong reason.
    """
    return SearchContextBundle(
        query=query,
        documents=[
            ContextDocument(
                id="d1",
                title="FAISS",
                content="FAISS is a library for efficient similarity search.",
                score=0.9,
            )
        ],
    )


class _ThreadRecordingLLM:
    """Records which thread `complete` ran on."""

    def __init__(self) -> None:
        self.thread_name: str | None = None
        self.calls = 0

    def complete(self, messages: Any, **kwargs: Any) -> str:
        self.calls += 1
        self.thread_name = threading.current_thread().name
        return "a grounded answer"


@pytest.mark.asyncio
async def test_answer_with_retrieval_offloads_generation(monkeypatch):
    """The classic RAG path must synthesize off the event-loop thread."""
    from src.context import pipeline

    loop_thread = threading.current_thread().name

    async def _fake_retrieve(question: str, **kwargs: Any):
        return _bundle(question)

    monkeypatch.setattr(pipeline, "retrieve_context", _fake_retrieve)

    llm = _ThreadRecordingLLM()
    await pipeline.answer_with_retrieval("what is faiss?", llm=llm)

    assert llm.calls >= 1, "the llm was never called; the test proves nothing"
    assert llm.thread_name is not None
    assert llm.thread_name != loop_thread, (
        "answer synthesis ran on the event-loop thread; a blocking requests "
        "call there stalls every other concurrent session for its full duration"
    )


@pytest.mark.asyncio
async def test_the_event_loop_stays_responsive_during_synthesis(monkeypatch):
    """A second task must make progress while an answer is being generated.

    The thread-identity assertion above is the precise invariant; this one is
    the consequence a user would actually notice, and it fails loudly if the
    offload is ever reverted.
    """
    from src.context import pipeline

    async def _fake_retrieve(question: str, **kwargs: Any):
        return _bundle(question)

    monkeypatch.setattr(pipeline, "retrieve_context", _fake_retrieve)

    import time

    class _SlowLLM:
        def complete(self, messages: Any, **kwargs: Any) -> str:
            time.sleep(0.25)  # a blocking network call, in miniature
            return "answer"

    ticks = 0

    async def _ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    tick_task = asyncio.create_task(_ticker())
    try:
        await pipeline.answer_with_retrieval("q", llm=_SlowLLM())
    finally:
        tick_task.cancel()

    # Blocked, ticks would be ~0. Offloaded, the 0.25s sleep leaves room for many.
    assert ticks >= 5, (
        f"event loop advanced only {ticks} ticks during a 0.25s synthesis; "
        "it was blocked"
    )


@pytest.mark.asyncio
async def test_agentic_rag_offloads_generation(monkeypatch):
    """The Assist CHAT route must synthesize off the event-loop thread too."""
    from src.agents.search import agentic_rag as ar

    loop_thread = threading.current_thread().name
    llm = _ThreadRecordingLLM()

    async def _fake_retrieve(queries: Any, **kwargs: Any):
        return [_bundle() for _ in queries]

    monkeypatch.setattr(ar, "retrieve_contexts", _fake_retrieve)

    loop_obj = ar.AgenticRAGLoop(ar.AgenticRAGConfig(max_rounds=1, topk=1), llm=llm)
    await loop_obj.run(
        "what is faiss?", chat_history=[ChatMessage(role="user", content="q")]
    )

    assert llm.calls >= 1, "the llm was never called; the test proves nothing"
    assert llm.thread_name != loop_thread, (
        "AgenticRAGLoop synthesized the answer on the event-loop thread"
    )
