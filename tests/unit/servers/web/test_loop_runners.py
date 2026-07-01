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
from src.context.models import SearchContextBundle
from src.context.search import SearchResult
from src.internal.servers.web import app as web_app


def _search_output(*, turns, num_turns=1, final_answer="grounded", trace=("e1",)):
    context = types.SimpleNamespace(turns=turns)
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
    turn = types.SimpleNamespace(
        results=[SearchResult(contents="Title\nbody", url=None)]
    )
    monkeypatch.setattr(
        "src.agents.search.SearchAgentLoop.run",
        AsyncMock(return_value=_search_output(turns=[turn])),
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
    assert citations == [documents[0].citation]
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
