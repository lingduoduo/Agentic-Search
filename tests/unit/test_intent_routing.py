"""Tests for rule-based classifier and trajectory intent inference."""

import json
from src.internal.servers.web.intent_routing import (
    _rule_based_is_search,
    _infer_intent_from_output,
)
from src.agents.base import AgentLoopOutput


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
