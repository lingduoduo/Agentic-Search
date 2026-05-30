"""Tests for src/search and src/prompts."""

from __future__ import annotations

import asyncio


from src.context.models import ChatMessage
from src.backend.prompts import CHAT_CLASS
from src.backend.prompts import KEYWORD_EXPANSION_PROMPT
from src.backend.prompts import QUERY_TYPE_PROMPT
from src.backend.prompts import SEARCH_CHAT_PROMPT
from src.backend.prompts import SEARCH_CLASS
from src.retrieval.context import SearchResult
from src.backend.search import SearchQueryResult
from src.backend.search import classify_query_type
from src.backend.search import classify_search_flow
from src.backend.search import expand_keywords
from src.backend.search import run_expanded_search
from src.backend.search import weighted_reciprocal_rank_fusion


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls: list[list[ChatMessage]] = []

    def complete(self, messages: list[ChatMessage], **_) -> str:
        self.calls.append(messages)
        return self._reply


def test_prompts_are_exported_and_contain_placeholders():
    assert "{user_query}" in KEYWORD_EXPANSION_PROMPT
    assert "{user_query}" in QUERY_TYPE_PROMPT
    assert "{user_query}" in SEARCH_CHAT_PROMPT
    assert SEARCH_CLASS == "search"
    assert CHAT_CLASS == "chat"


def test_expand_keywords_parses_llm_lines():
    llm = _FakeLLM("machine learning\nneural networks\ndeep learning")
    expansions = expand_keywords("ML overview", llm)
    assert expansions == ["machine learning", "neural networks", "deep learning"]
    assert len(llm.calls) == 1
    assert KEYWORD_EXPANSION_PROMPT.split("{user_query}")[0] in llm.calls[0][0].content


def test_expand_keywords_strips_duplicate_of_original():
    llm = _FakeLLM("ML overview\nneural networks")
    expansions = expand_keywords("ML overview", llm)
    assert "ML overview" not in expansions
    assert "neural networks" in expansions


def test_expand_keywords_returns_empty_on_blank_response():
    llm = _FakeLLM("   \n\n")
    assert expand_keywords("hello", llm) == []


def test_classify_query_type_returns_keyword_or_semantic():
    assert classify_query_type("procurement process", _FakeLLM("keyword")) == "keyword"
    assert classify_query_type("What is ML?", _FakeLLM("semantic")) == "semantic"
    assert classify_query_type("?", _FakeLLM("something else")) == "semantic"


def test_classify_search_flow_returns_search_or_chat():
    assert (
        classify_search_flow("Sales Runbook AMEA", _FakeLLM("search")) == SEARCH_CLASS
    )
    assert classify_search_flow("Write me a script", _FakeLLM("chat")) == CHAT_CLASS
    assert classify_search_flow("?", _FakeLLM("unknown")) == CHAT_CLASS


def test_weighted_rrf_deduplicates_and_boosts_overlap():
    r_a = SearchResult(contents="doc a", score=0.9, url="u1")
    r_b_orig = SearchResult(contents="doc b", score=0.5, url="u2")
    r_b_exp = SearchResult(contents="doc b", score=0.8, url="u2")
    r_c = SearchResult(contents="doc c", score=0.4, url="u3")

    merged = weighted_reciprocal_rank_fusion(
        [[r_a, r_b_orig], [r_b_exp, r_c]], [2.0, 1.0]
    )

    # u2 appears in both lists, so it rises above u1 (which only appears once)
    urls = [r.url for r in merged]
    assert urls[0] == "u2"
    assert set(urls) == {"u1", "u2", "u3"}


def test_weighted_rrf_single_list_is_identity():
    results = [
        SearchResult(contents="a", score=0.9, url="u1"),
        SearchResult(contents="b", score=0.5, url="u2"),
    ]
    merged = weighted_reciprocal_rank_fusion([results], [1.0])
    assert [r.url for r in merged] == ["u1", "u2"]


def test_run_expanded_search_no_llm_runs_single_search(monkeypatch):
    async def fake_run_search(
        request, *, search_url="http://localhost:8000/retrieve", **_
    ):
        return [SearchResult(contents=f"result for {request.query}", score=0.9)]

    monkeypatch.setattr("src.backend.search.process_search_query.run_search", fake_run_search)

    result = asyncio.run(
        run_expanded_search("what is ML?", search_url="http://test/retrieve")
    )

    assert isinstance(result, SearchQueryResult)
    assert result.original_query == "what is ML?"
    assert result.executed_queries == ["what is ML?"]
    assert len(result.results) == 1


def test_run_expanded_search_merges_expansions_with_rrf(monkeypatch):
    call_log: list[str] = []

    async def fake_run_search(
        request, *, search_url="http://localhost:8000/retrieve", **_
    ):
        call_log.append(request.query)
        return [
            SearchResult(
                contents=f"result for {request.query}", score=0.8, url=request.query
            )
        ]

    monkeypatch.setattr("src.backend.search.process_search_query.run_search", fake_run_search)

    llm = _FakeLLM("keyword one\nkeyword two")
    result = asyncio.run(run_expanded_search("original query", llm=llm, top_k=5))

    assert result.executed_queries == ["original query", "keyword one", "keyword two"]
    assert len(call_log) == 3
    assert len(result.results) == 3


def test_run_expanded_search_skips_failed_parallel_search(monkeypatch):
    call_count = 0

    async def fake_run_search(
        request, *, search_url="http://localhost:8000/retrieve", **_
    ):
        nonlocal call_count
        call_count += 1
        if request.query != "original":
            raise RuntimeError("search backend down")
        return [SearchResult(contents="good result", score=0.9, url="u1")]

    monkeypatch.setattr("src.backend.search.process_search_query.run_search", fake_run_search)

    llm = _FakeLLM("expansion one")
    result = asyncio.run(run_expanded_search("original", llm=llm))

    assert call_count == 2
    assert len(result.results) == 1
    assert result.results[0].url == "u1"


def test_run_expanded_search_respects_expand_false(monkeypatch):
    async def fake_run_search(request, **_):
        return [SearchResult(contents="r", score=0.5)]

    monkeypatch.setattr("src.backend.search.process_search_query.run_search", fake_run_search)

    llm = _FakeLLM("should not be called")
    result = asyncio.run(run_expanded_search("q", llm=llm, expand=False))

    assert llm.calls == []
    assert result.executed_queries == ["q"]
