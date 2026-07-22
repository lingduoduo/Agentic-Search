"""Unit tests for the shared agent-loop runners in app.py.

Each runner builds + runs one loop and returns the canonical tuple
``(answer, citations, documents, intent, extra)`` consumed by both the
auto-route dispatcher and the explicit-mode chain.
"""

from __future__ import annotations

import types

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.search import AgenticRAGResult
from src.agents.core.base import AgentLoopOutput
from src.context.models import ContextDocument, SearchContextBundle, SearchFilters
from src.context.search import SearchResult
from src.internal.servers.web import app as web_app


def _search_output(
    *, turns=None, rounds=None, num_turns=1, final_answer="grounded", trace=("e1",)
):
    if rounds is None:
        rounds = [list(turns)] if turns else []
    if turns is None:
        turns = [ctx for round_ctxs in rounds for ctx in round_ctxs]
    context = types.SimpleNamespace(turns=turns, rounds=rounds)
    return AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=num_turns,
        final_answer=final_answer,
        context=context,
        control_flow_trace=list(trace),
    )


@pytest.mark.asyncio
async def test_run_search_agent_returns_canonical_tuple(monkeypatch):
    ctx = types.SimpleNamespace(
        results=[SearchResult(contents="Title\nbody", url=None)]
    )
    monkeypatch.setattr(
        "src.agents.search.SearchAgentLoop.run",
        AsyncMock(return_value=_search_output(rounds=[[ctx]])),
    )
    answer, citations, documents, intent, extra = await web_app._run_search_agent(
        "q",
        manager=MagicMock(),
        tokenizer=MagicMock(),
        search_url="http://x/retrieve",
        top_k=5,
        on_turn=None,
        on_trace=None,
    )
    assert answer == "grounded"
    assert intent == "search"
    assert len(documents) == 1
    assert documents[0].citation == "[R1Q1D1]"
    assert citations == ["[R1Q1D1]"]
    # Source cards must show a provider label, not "Unknown": the search-agent
    # path retrieves from the local retrieval server.
    assert documents[0].metadata["source"] == "Local Retrieval"
    assert extra["num_turns"] == 1
    assert extra["control_flow_trace"] == ["e1"]


@pytest.mark.asyncio
async def test_run_agentic_rag_returns_canonical_tuple(monkeypatch):
    result = AgenticRAGResult(
        answer="synth",
        citations=["[D1]"],
        rounds_used=2,
        context=SearchContextBundle(query="q", documents=[]),
    )
    monkeypatch.setattr(
        "src.agents.search.agentic_rag.AgenticRAGLoop.run",
        AsyncMock(return_value=result),
    )
    answer, citations, documents, intent, extra = await web_app._run_agentic_rag(
        "q",
        llm=MagicMock(),
        search_url="http://x/retrieve",
        top_k=5,
        history=[],
    )
    assert answer == "synth"
    assert intent == "chat"
    assert citations == ["[D1]"]
    # F3: _run_agentic_rag now returns the chat-loop control-flow trace in extra.
    # The monkeypatched run() ignores the recorder, so the trace is empty here.
    assert extra == {"rounds_used": 2, "control_flow_trace": []}


@pytest.mark.asyncio
async def test_run_agentic_rag_labels_document_source(monkeypatch):
    # RAG documents carry raw retrieval metadata (no "source" key); the runner
    # must stamp a provider label so source cards don't render "Unknown".
    doc = ContextDocument(
        id="D1", title="T", content="c", url=None, score=0.5, metadata={}
    )
    result = AgenticRAGResult(
        answer="synth",
        citations=["[D1]"],
        rounds_used=1,
        context=SearchContextBundle(query="q", documents=[doc]),
    )
    monkeypatch.setattr(
        "src.agents.search.agentic_rag.AgenticRAGLoop.run",
        AsyncMock(return_value=result),
    )
    _, _, documents, _, _ = await web_app._run_agentic_rag(
        "q",
        llm=MagicMock(),
        search_url="http://x/retrieve",
        top_k=5,
        history=[],
    )
    assert documents[0].metadata["source"] == "Local Retrieval"
    # Citation id is preserved so answer links still resolve to the card.
    assert documents[0].citation == "[D1]"


def test_document_with_metadata_prefers_real_source_over_provider_label():
    # When the retrieval layer supplies a real per-document source, it wins over
    # the generic provider label; source_provider still records the fetcher.
    doc = ContextDocument(
        id="D1",
        title="T",
        content="c",
        url=None,
        score=0.5,
        metadata={"source": "Team Wiki"},
    )
    labeled = web_app._document_with_metadata(
        doc, source_provider="retrieval", query="q", entry_point="rag"
    )
    assert labeled.metadata["source"] == "Team Wiki"
    assert labeled.metadata["source_provider"] == "retrieval"


def test_document_with_metadata_falls_back_to_provider_label():
    doc = ContextDocument(
        id="D1", title="T", content="c", url=None, score=0.5, metadata={}
    )
    labeled = web_app._document_with_metadata(
        doc, source_provider="retrieval", query="q", entry_point="rag"
    )
    assert labeled.metadata["source"] == "Local Retrieval"


@pytest.mark.asyncio
async def test_run_agentic_rag_threads_access_filters_into_loop(monkeypatch):
    filters = SearchFilters(access_acl=["user:alice"])
    observed = {}

    async def fake_run(self, question, **kwargs):
        observed["filters"] = self.config.filters
        return AgenticRAGResult(
            answer="synth",
            citations=[],
            rounds_used=1,
            context=SearchContextBundle(query=question, documents=[]),
        )

    monkeypatch.setattr("src.agents.search.agentic_rag.AgenticRAGLoop.run", fake_run)
    await web_app._run_agentic_rag(
        "q",
        llm=MagicMock(),
        search_url="http://x/retrieve",
        top_k=5,
        filters=filters,
        history=[],
    )

    assert observed["filters"] is filters


@pytest.mark.asyncio
async def test_run_tool_agent_exposes_assistant_fallback(monkeypatch):
    output = AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        final_answer="",  # empty → callers decide policy
        action_trace="",
        trajectory_messages=[{"role": "assistant", "content": "fallback text"}],
    )
    monkeypatch.setattr(
        "src.agents.tool.tool_calling.ToolAgentLoop.run",
        AsyncMock(return_value=output),
    )
    answer, citations, documents, intent, extra = await web_app._run_tool_agent(
        "q",
        manager=MagicMock(),
        tokenizer=MagicMock(),
        search_url="http://x/retrieve",
        history=[],
        resolved=types.SimpleNamespace(tool_agent_parser="hermes"),
        on_turn=None,
        with_search_tool=False,
    )
    assert answer == ""  # runner applies no fallback
    assert extra["_assistant_fallback"] == "fallback text"
    assert extra["tool_calls"] == []
    assert extra["num_turns"] == 1
