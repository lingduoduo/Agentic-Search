# tests/unit/test_intent_routing.py
import asyncio
import json
import json as _json
from unittest.mock import AsyncMock, patch

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


# --- routing tool builder tests ---


def test_build_search_routing_tool_schema():
    tool = build_search_routing_tool(
        search_url="http://localhost:8001/retrieve", top_k=3
    )
    assert tool.schema.name == "search_routing_tool"
    assert "query" in tool.schema.parameters.get("properties", {})


def test_build_rag_routing_tool_schema():
    tool = build_rag_routing_tool(
        llm=None, search_url="http://localhost:8001/retrieve", top_k=3
    )
    assert tool.schema.name == "rag_routing_tool"
    assert "query" in tool.schema.parameters.get("properties", {})


def test_search_routing_tool_returns_json():
    from src.tools.search import SearchPage

    fake_pages = [SearchPage(title="Doc A", summary="Content A", url="http://a.com")]

    async def run():
        with patch(
            "src.tools.routing_tools.search_tool",
            new=AsyncMock(return_value=fake_pages),
        ):
            tool = build_search_routing_tool(
                search_url="http://localhost:8001/retrieve", top_k=3
            )
            response_text, raw, _meta = await tool.execute(
                "default", {"query": "FAISS"}
            )
        return response_text

    result = asyncio.run(run())
    data = _json.loads(result)
    assert isinstance(data, list)
    assert data[0]["title"] == "Doc A"


def test_rag_routing_tool_returns_answer():
    from src.context.models import (
        AnswerGenerationResult,
        PromptBundle,
        SearchContextBundle,
    )

    fake_context = SearchContextBundle(query="q", documents=[])
    fake_prompt = PromptBundle(system="", user="", messages=[])
    fake_result = AnswerGenerationResult(
        answer="42",
        citations=["[D1]"],
        context=fake_context,
        prompt=fake_prompt,
    )

    async def run():
        with patch(
            "src.context.answer_with_retrieval", new=AsyncMock(return_value=fake_result)
        ):
            tool = build_rag_routing_tool(
                llm=None, search_url="http://localhost:8001/retrieve", top_k=3
            )
            response_text, raw, _meta = await tool.execute(
                "default", {"query": "What is the answer?"}
            )
        return response_text

    result = asyncio.run(run())
    data = _json.loads(result)
    assert data["answer"] == "42"
