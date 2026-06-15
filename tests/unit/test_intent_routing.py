"""Tests for rule-based classifier and trajectory intent inference."""

import json

import pytest
from src.agents.base import AgentLoopOutput
from src.internal.servers.web.intent_routing import (
    _infer_intent_from_output,
    _rule_based_is_search,
)
from src.tools.routing_tools import build_rag_routing_tool, build_search_routing_tool


def _make_output(
    action_trace: str | None = None, final_answer: str | None = None
) -> AgentLoopOutput:
    return AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        action_trace=action_trace,
        final_answer=final_answer,
    )


# --- _rule_based_is_search ---


def test_rule_based_is_search_find_keyword():
    assert _rule_based_is_search("find me the onboarding doc") is True


def test_rule_based_is_search_short_keyword_query():
    assert _rule_based_is_search("procurement process") is True  # ≤5 tokens, no verb


def test_rule_based_is_search_list_keyword():
    assert _rule_based_is_search("list all pull requests since last week") is True


def test_rule_based_is_chat_explain_keyword():
    assert _rule_based_is_search("explain how FAISS works") is False


def test_rule_based_is_chat_what_is():
    assert (
        _rule_based_is_search("what is the difference between BM25 and dense retrieval")
        is False
    )


def test_rule_based_is_chat_default_no_signal():
    assert _rule_based_is_search("what led us to win the deal with company X") is False


def test_rule_based_is_search_show_me():
    assert _rule_based_is_search("show me the deployment runbook") is True


def test_rule_based_is_search_empty_returns_false():
    assert _rule_based_is_search("") is False


def test_rule_based_chat_signal_beats_search_signal():
    # 'explain' (chat) + 'find' (search) in same query → chat wins
    assert _rule_based_is_search("explain how to find docs") is False


# --- _infer_intent_from_output ---


def _trace(tool_name: str) -> str:
    return json.dumps(
        {
            "tool_name": tool_name,
            "status": "completed",
            "result": "ok",
            "performance": {},
            "error_code": None,
            "error_message": None,
            "optimization_suggestions": [],
            "retry_count": 0,
        }
    )


def test_infer_intent_search_routing_tool():
    output = _make_output(action_trace=_trace("search_routing_tool"))
    assert _infer_intent_from_output(output) == "search"


def test_infer_intent_rag_routing_tool():
    output = _make_output(action_trace=_trace("rag_routing_tool"))
    assert _infer_intent_from_output(output) == "chat"


def test_infer_intent_mcp_tool():
    output = _make_output(action_trace=_trace("custom_api"))
    assert _infer_intent_from_output(output) == "tool"


def test_infer_intent_no_trace_defaults_to_chat():
    output = _make_output(action_trace=None)
    assert _infer_intent_from_output(output) == "chat"


def test_infer_intent_malformed_trace_defaults_to_chat():
    output = _make_output(action_trace="not json")
    assert _infer_intent_from_output(output) == "chat"


def test_build_search_routing_tool_schema():
    tool = build_search_routing_tool(
        search_url="http://localhost:8000/retrieve", top_k=5
    )
    schema = tool.schema.to_dict()
    assert schema["function"]["name"] == "search_routing_tool"
    assert "query" in schema["function"]["parameters"]["properties"]


def test_build_rag_routing_tool_schema():
    tool = build_rag_routing_tool(
        llm=None, search_url="http://localhost:8000/retrieve", top_k=5
    )
    schema = tool.schema.to_dict()
    assert schema["function"]["name"] == "rag_routing_tool"
    assert "query" in schema["function"]["parameters"]["properties"]


@pytest.mark.asyncio
async def test_search_routing_tool_returns_json(monkeypatch):
    from src.tools import SearchPage

    async def fake_search_tool(query, *, provider, search_url, page_size):
        return [
            SearchPage(
                title="Doc A", summary="summary", url="http://example.com", error=None
            )
        ]

    monkeypatch.setattr("src.tools.routing_tools.search_tool", fake_search_tool)
    tool = build_search_routing_tool(
        search_url="http://localhost:8000/retrieve", top_k=5
    )
    result, _, _ = await tool.execute("default", {"query": "FAISS"})
    data = json.loads(result)
    assert data[0]["title"] == "Doc A"


@pytest.mark.asyncio
async def test_rag_routing_tool_returns_answer(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock
    from src.context.models import (
        AnswerGenerationResult,
        SearchContextBundle,
        PromptBundle,
    )

    fake_result = AnswerGenerationResult(
        answer="FAISS is a library.",
        citations=["[D1]"],
        context=SearchContextBundle(query="q", documents=[]),
        prompt=PromptBundle(system="", user="", messages=[]),
    )
    mock_llm = MagicMock()
    monkeypatch.setattr(
        "src.tools.routing_tools.answer_with_retrieval",
        AsyncMock(return_value=fake_result),
    )
    tool = build_rag_routing_tool(
        llm=mock_llm, search_url="http://localhost:8000/retrieve", top_k=5
    )
    result, _, _ = await tool.execute("default", {"query": "What is FAISS?"})
    data = json.loads(result)
    assert data["answer"] == "FAISS is a library."
