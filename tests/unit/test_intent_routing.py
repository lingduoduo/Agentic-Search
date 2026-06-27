# tests/unit/test_intent_routing.py
import asyncio
import json
import json as _json
from unittest.mock import AsyncMock, patch

from src.agents.core.base import AgentLoopOutput
from src.internal.servers.web.intent_routing import _infer_intent_from_output
from src.tools.routing_tools import build_rag_routing_tool, build_search_routing_tool
from src.tools import ToolEffect


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
    assert tool.effect is ToolEffect.READ_ONLY
    assert "query" in tool.schema.parameters.get("properties", {})


def test_build_rag_routing_tool_schema():
    tool = build_rag_routing_tool(
        llm=None, search_url="http://localhost:8001/retrieve", top_k=3
    )
    assert tool.schema.name == "rag_routing_tool"
    assert tool.effect is ToolEffect.READ_ONLY
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
