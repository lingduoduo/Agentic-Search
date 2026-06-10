"""Tests for src/secondary_llm_flows."""

from __future__ import annotations

from src.context.models import ChatMessage
from src.backend.servers.secondary_llm_flows import classify_is_search_flow
from src.backend.servers.secondary_llm_flows import expand_keywords
from src.backend.servers.secondary_llm_flows.query_expansion import _clean_keyword_line
from src.backend.servers.secondary_llm_flows.query_expansion import is_time_sensitive
from src.backend.servers.secondary_llm_flows.query_expansion import (
    with_temporal_context,
)


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls: list[list[ChatMessage]] = []

    def complete(self, messages: list[ChatMessage], **_) -> str:
        self.calls.append(messages)
        return self._reply


class _RaisingLLM:
    def complete(self, messages: list[ChatMessage], **_) -> str:
        raise RuntimeError("API unreachable")


# --- _clean_keyword_line ---


def test_clean_keyword_line_strips_brackets_and_quotes():
    assert _clean_keyword_line("[machine learning]") == "machine learning"
    assert _clean_keyword_line('"neural networks"') == "neural networks"
    assert _clean_keyword_line("`deep learning`") == "deep learning"


def test_clean_keyword_line_strips_list_markers():
    assert _clean_keyword_line("1. procurement process") == "procurement process"
    assert _clean_keyword_line("2) onboarding doc") == "onboarding doc"
    assert _clean_keyword_line("- keyword search") == "keyword search"
    assert _clean_keyword_line("* BM25 retrieval") == "BM25 retrieval"


def test_clean_keyword_line_leaves_plain_text_unchanged():
    assert (
        _clean_keyword_line("machine learning overview") == "machine learning overview"
    )


# --- expand_keywords ---


def test_expand_keywords_cleans_artifacts_and_returns_lines():
    llm = _FakeLLM("[machine learning]\n1. neural networks\n- deep learning")
    result = expand_keywords("ML overview", llm)
    assert result == ["machine learning", "neural networks", "deep learning"]


def test_expand_keywords_excludes_original_query_case_insensitively():
    llm = _FakeLLM("ML Overview\nneural networks")
    result = expand_keywords("ML overview", llm)
    assert "ML Overview" not in result
    assert "neural networks" in result


def test_expand_keywords_deduplicates_across_lines():
    llm = _FakeLLM("keyword search\nKeyword Search\nBM25")
    result = expand_keywords("query", llm)
    assert result == ["keyword search", "BM25"]


def test_expand_keywords_returns_empty_on_blank_response():
    assert expand_keywords("q", _FakeLLM("")) == []
    assert expand_keywords("q", _FakeLLM("   \n\n  ")) == []


def test_expand_keywords_returns_empty_on_llm_exception():
    assert expand_keywords("q", _RaisingLLM()) == []


def test_expand_keywords_includes_prompt_in_llm_call():
    from src.backend.prompts.query_expansion import KEYWORD_EXPANSION_PROMPT

    llm = _FakeLLM("keyword")
    expand_keywords("my query", llm)
    sent = llm.calls[0][0].content
    assert "my query" in sent
    assert KEYWORD_EXPANSION_PROMPT.split("{user_query}")[0] in sent


# --- classify_is_search_flow ---


def test_classify_is_search_flow_returns_true_for_search():
    assert classify_is_search_flow("Sales Runbook AMEA", _FakeLLM("search")) is True


def test_classify_is_search_flow_returns_false_for_chat():
    assert classify_is_search_flow("Write me a script", _FakeLLM("chat")) is False


def test_classify_is_search_flow_prefers_chat_when_both_labels_appear():
    assert classify_is_search_flow("q", _FakeLLM("search or chat")) is False


def test_classify_is_search_flow_defaults_to_false_on_empty_response():
    assert classify_is_search_flow("q", _FakeLLM("")) is False


def test_classify_is_search_flow_defaults_to_false_on_unexpected_response():
    assert classify_is_search_flow("q", _FakeLLM("unknown label")) is False


def test_classify_is_search_flow_includes_query_in_prompt():
    from src.backend.prompts.search_flow_classification import SEARCH_CHAT_PROMPT

    llm = _FakeLLM("search")
    classify_is_search_flow("my question", llm)
    sent = llm.calls[0][0].content
    assert "my question" in sent
    assert SEARCH_CHAT_PROMPT.split("{user_query}")[0] in sent


def test_is_time_sensitive_detects_temporal_keywords():
    assert is_time_sensitive("latest AI news") is True
    assert is_time_sensitive("current president of France") is True
    assert is_time_sensitive("recent breakthroughs in quantum computing") is True
    assert is_time_sensitive("today's stock price") is True
    assert is_time_sensitive("what is FAISS") is False
    assert is_time_sensitive("how does BM25 work") is False


def test_is_time_sensitive_is_case_insensitive():
    assert is_time_sensitive("LATEST updates") is True
    assert is_time_sensitive("Current events") is True


def test_with_temporal_context_appends_date_to_time_sensitive_queries():
    from datetime import datetime

    result = with_temporal_context("latest AI models")
    year = str(datetime.now().year)
    assert year in result
    assert "latest AI models" in result


def test_with_temporal_context_leaves_non_temporal_queries_unchanged():
    result = with_temporal_context("what is FAISS")
    assert result == "what is FAISS"


def test_is_time_sensitive_no_false_positive_substring_match():
    # "now" as a substring of common words should NOT trigger time sensitivity
    assert is_time_sensitive("knowledge management") is False
    assert is_time_sensitive("snowflake architecture") is False
    assert is_time_sensitive("renewal policy") is False
    assert is_time_sensitive("renowned scientist") is False


def test_is_time_sensitive_detects_whole_word_now():
    # "now" as a standalone word SHOULD trigger time sensitivity
    assert is_time_sensitive("what is happening now") is True
    assert is_time_sensitive("NOW is the time") is True
