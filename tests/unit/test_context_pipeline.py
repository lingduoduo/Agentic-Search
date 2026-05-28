"""Tests for search-context prompt and answer helpers."""

from __future__ import annotations

import asyncio

from src.context import AgentBehavior
from src.context import AgentBehaviorConfig
from src.context import AnswerGenerationRequest
from src.context import AnswerStyle
from src.context import ChatMessage
from src.context import LLMResponse
from src.context import SearchRequest
from src.context import build_answer_prompt
from src.context import build_context_bundle
from src.context import extract_citations
from src.context import generate_answer
from src.context import synthesize_answer_from_context
from src.context.retrieval.search_runner import combine_search_results
from src.context.retrieval.search_runner import run_search
from src.retrieval.context import SearchResult


def _results() -> list[SearchResult]:
    return [
        SearchResult(
            title="FAISS",
            contents='"FAISS"\nFAISS is a vector similarity search library.',
            url="https://faiss.ai",
            score=0.9,
        ),
        SearchResult(
            title="BM25",
            contents='"BM25"\nBM25 is a sparse lexical ranking function.',
            score=0.5,
        ),
    ]


def test_build_context_bundle_formats_cited_context():
    bundle = build_context_bundle("compare retrieval", _results())

    assert [document.id for document in bundle.documents] == ["D1", "D2"]
    text = bundle.to_context_text()
    assert "[D1] FAISS" in text
    assert "https://faiss.ai" in text


def test_build_answer_prompt_includes_behavior_and_context():
    bundle = build_context_bundle("what is faiss", _results())
    prompt = build_answer_prompt(
        "what is faiss",
        bundle,
        AgentBehaviorConfig(
            behavior=AgentBehavior.RESEARCH,
            answer_style=AnswerStyle.BULLETS,
        ),
    )

    assert prompt.messages[0].role == "system"
    assert "Prefer compact bullet points" in prompt.system
    assert "[D1] FAISS" in prompt.user


def test_generate_answer_uses_llm_and_extracts_citations():
    class FakeLLM:
        def complete(self, messages, **kwargs):
            assert any(message.role == "user" for message in messages)
            return LLMResponse("FAISS supports vector search [D1].")

    bundle = build_context_bundle("what is faiss", _results())
    result = generate_answer(
        AnswerGenerationRequest(
            question="what is faiss",
            context=bundle,
            chat_history=[ChatMessage(role="user", content="hi")],
        ),
        llm=FakeLLM(),
    )

    assert result.answer == "FAISS supports vector search [D1]."
    assert result.citations == ["D1"]


def test_synthesize_answer_from_context_falls_back_without_llm():
    bundle = build_context_bundle("what is faiss", _results())

    answer = synthesize_answer_from_context("what is faiss", bundle)

    assert "[D1]" in answer
    assert "FAISS is a vector similarity search library" in answer


def test_extract_citations_deduplicates_in_order():
    assert extract_citations("Use [D2] and [D1], then [D2].") == ["D2", "D1"]


def test_combine_search_results_deduplicates_by_url_and_content():
    combined = combine_search_results(
        [
            [SearchResult(contents="same", url="u", score=0.1)],
            [SearchResult(contents="same", url="u", score=0.9)],
            [SearchResult(contents="other", score=0.2)],
        ]
    )

    assert [result.score for result in combined] == [0.9, 0.2]


def test_run_search_uses_retrieval_client(monkeypatch):
    class FakeClient:
        def __init__(self, config):
            self.config = config

        async def retrieve_one(self, query, topk=None):
            assert query == "faiss"
            assert topk == 2
            return _results()

        async def aclose(self):
            return None

    monkeypatch.setattr("src.context.retrieval.search_runner.SearchClient", FakeClient)

    rows = asyncio.run(run_search(SearchRequest(query="faiss", top_k=2)))

    assert rows[0].title == "FAISS"


def test_context_helpers_are_exported_from_top_level_src():
    from src import SearchRequest as ExportedSearchRequest
    from src import build_answer_prompt as exported_build_answer_prompt

    assert ExportedSearchRequest(query="hello").query == "hello"
    assert exported_build_answer_prompt is build_answer_prompt
