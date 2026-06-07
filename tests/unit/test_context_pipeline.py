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
from src.context import SearchFilters
from src.context import build_answer_prompt
from src.context import build_context_bundle
from src.context import extract_citations
from src.context import generate_answer
from src.context import rank_evidence_snippets
from src.context import synthesize_answer_from_context
from src.context.retrieval.search_runner import combine_search_results
from src.context.retrieval.search_runner import run_search
from src.context.search import SearchResult


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


def test_search_filters_match_access_acl_metadata():
    filters = SearchFilters(access_acl=["public", "group:eng"])

    assert filters.matches({"acl": ["group:eng"]})
    assert filters.matches({"tags": {"acl": "public,user:alice"}})
    assert not filters.matches({"acl": ["group:finance"]})


def test_search_filters_match_external_group_acl_metadata():
    filters = SearchFilters(access_acl=["external_group:okta-eng"])

    assert filters.matches({"acl": ["external_group:okta-eng"]})
    assert filters.matches({"tags": {"acl": "external_group:okta-eng,user:alice"}})
    assert not filters.matches({"acl": ["external_group:okta-finance"]})


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


def test_run_search_sends_filters_to_retrieval_client(monkeypatch):
    class FakeClient:
        def __init__(self, config):
            self.config = config

        async def retrieve_one(self, query, topk=None, filters=None):
            assert query == "faiss"
            assert topk == 2
            assert filters == {"access_acl": ["public"]}
            return [
                SearchResult(
                    title="Visible",
                    contents='"Visible"\nAllowed',
                    metadata={"acl": ["public"]},
                ),
                SearchResult(
                    title="Blocked",
                    contents='"Blocked"\nHidden',
                    metadata={"acl": ["user:alice"]},
                ),
            ]

        async def aclose(self):
            return None

    monkeypatch.setattr("src.context.retrieval.search_runner.SearchClient", FakeClient)

    rows = asyncio.run(
        run_search(
            SearchRequest(
                query="faiss",
                top_k=2,
                filters=SearchFilters(access_acl=["public"]),
            )
        )
    )

    assert [row.title for row in rows] == ["Visible"]


def test_context_helpers_are_exported_from_top_level_src():
    from src import SearchRequest as ExportedSearchRequest
    from src import build_answer_prompt as exported_build_answer_prompt
    from src import rank_evidence_snippets as exported_rank_evidence_snippets

    assert ExportedSearchRequest(query="hello").query == "hello"
    assert exported_build_answer_prompt is build_answer_prompt
    assert exported_rank_evidence_snippets is rank_evidence_snippets


# ---------------------------------------------------------------------------
# synthesize_answer_from_context — extractive behaviour
# ---------------------------------------------------------------------------

from src.context.models import SearchContextBundle  # noqa: E402
from src.context.pipeline import _overlap_score, _split_sentences, _tokenize  # noqa: E402


def test_synthesize_selects_most_relevant_sentence():
    """The sentence with the highest keyword overlap must appear in the answer."""
    doc = SearchResult(
        title="Olympics history",
        contents=(
            "The Olympics were founded in ancient Greece."
            " Paris hosted the Summer Olympics in 1900 and again in 1924."
            " The modern Games attract thousands of athletes worldwide."
        ),
        score=1.0,
    )
    bundle = build_context_bundle("When did Paris host the Olympics?", [doc])
    answer = synthesize_answer_from_context("When did Paris host the Olympics?", bundle)

    assert "[D1]" in answer
    # Must mention the specific fact, not just dump the whole document
    assert "1900" in answer or "1924" in answer or "Paris" in answer


def test_synthesize_no_documents_mentions_question():
    bundle = SearchContextBundle(query="q", documents=[])
    answer = synthesize_answer_from_context("What is quantum computing?", bundle)
    assert "quantum computing" in answer.lower() or "context" in answer.lower()


def test_synthesize_deduplicates_citations():
    """Each source document should appear at most once."""
    results = [
        SearchResult(
            title="A",
            contents='"A"\nFAISS is fast and efficient for vector search queries.',
            score=0.9,
        ),
        SearchResult(
            title="B",
            contents='"B"\nBM25 is a traditional lexical ranking method used in search.',
            score=0.7,
        ),
    ]
    bundle = build_context_bundle("fast search methods", results)
    answer = synthesize_answer_from_context("fast search methods", bundle)

    assert answer.count("[D1]") <= 1
    assert answer.count("[D2]") <= 1


def test_rank_evidence_snippets_uses_retrieval_score_as_tiebreaker():
    results = [
        SearchResult(
            title="Lower ranked",
            contents='"Lower ranked"\nVector search retrieves nearest neighbors.',
            score=0.1,
        ),
        SearchResult(
            title="Higher ranked",
            contents='"Higher ranked"\nVector search retrieves nearest neighbors.',
            score=0.9,
        ),
    ]
    bundle = build_context_bundle("vector search nearest neighbors", results)

    snippets = rank_evidence_snippets(
        "vector search nearest neighbors", bundle, max_snippets=2
    )

    assert [snippet.citation for snippet in snippets] == ["[D2]", "[D1]"]
    assert snippets[0].score > snippets[1].score


def test_rank_evidence_snippets_contextualizes_adjacent_chunks():
    results = [
        SearchResult(
            title="Policy chunk 1",
            contents='"Policy"\nThe approval workflow starts with a manager review.',
            score=0.7,
            metadata={"document_id": "policy", "chunk_id": 1},
        ),
        SearchResult(
            title="Policy chunk 2",
            contents="Finance signs off after manager approval for purchases.",
            score=0.6,
            metadata={"document_id": "policy", "chunk_id": 2},
        ),
    ]
    bundle = build_context_bundle(
        "Who signs off after manager approval?",
        results,
        merge_adjacent=True,
    )

    snippets = rank_evidence_snippets(
        "Who signs off after manager approval?", bundle, max_snippets=1
    )

    assert snippets[0].section is not None
    assert snippets[0].text == "Finance signs off after manager approval for purchases."


def test_tokenize_handles_numbers():
    tokens = _tokenize("Paris hosted the 2024 Olympics")
    assert "2024" in tokens
    assert "paris" in tokens
    assert "the" not in tokens


def test_overlap_score_empty_inputs():
    assert _overlap_score(set(), {"paris"}) == 0.0
    assert _overlap_score({"paris"}, set()) == 0.0


def test_split_sentences_preserves_long_sentences():
    text = "This is a sufficiently long first sentence about vector search. This is another sentence about BM25 ranking."
    sentences = _split_sentences(text)
    assert len(sentences) == 2


# ---------------------------------------------------------------------------
# build_answer_prompt — synthesis quality
# ---------------------------------------------------------------------------


from src.context.models import ContextDocument  # noqa: E402


def _make_bundle_for_prompt() -> SearchContextBundle:
    return SearchContextBundle(
        query="What is FAISS?",
        documents=[
            ContextDocument(
                id="D1",
                title="FAISS Overview",
                content="FAISS is a library for efficient similarity search.",
                score=0.9,
            )
        ],
    )


def test_answer_prompt_instructs_on_uncertainty():
    prompt = build_answer_prompt("What is FAISS?", _make_bundle_for_prompt())
    full_text = prompt.system + prompt.user
    assert "uncertain" in full_text.lower() or "missing" in full_text.lower()


def test_answer_prompt_instructs_on_citation_format():
    prompt = build_answer_prompt("What is FAISS?", _make_bundle_for_prompt())
    full_text = prompt.system + prompt.user
    assert "[D" in full_text


def test_answer_prompt_instructs_on_conflicting_evidence():
    prompt = build_answer_prompt("What is FAISS?", _make_bundle_for_prompt())
    assert (
        "conflict" in prompt.system.lower()
        or "contradict" in prompt.system.lower()
        or "disagree" in prompt.system.lower()
    )


def test_answer_prompt_forbids_fabrication():
    prompt = build_answer_prompt("What is FAISS?", _make_bundle_for_prompt())
    full_text = prompt.system + prompt.user
    assert (
        "fabricat" in full_text.lower()
        or "not in the context" in full_text.lower()
        or "only" in full_text.lower()
    )


# ---------------------------------------------------------------------------
# mmr_rerank
# ---------------------------------------------------------------------------

from src.context.utils import mmr_rerank  # noqa: E402


def _doc(id: str, title: str, score: float, url: str | None = None) -> ContextDocument:
    return ContextDocument(
        id=id, title=title, content=f"Content of {title}.", score=score, url=url
    )


def test_mmr_rerank_lambda_1_returns_score_order():
    docs = [_doc("D1", "A", 0.9), _doc("D2", "B", 0.7), _doc("D3", "C", 0.5)]
    result = mmr_rerank(docs, topk=2, lambda_=1.0)
    assert [d.id for d in result] == ["D1", "D2"]


def test_mmr_rerank_penalises_same_source():
    # D1 and D2 share source "example.com"; D3 is different.
    # With low lambda, D3 should be preferred over D2 as second pick.
    docs = [
        _doc("D1", "A", 0.9, url="http://example.com/1"),
        _doc("D2", "B", 0.8, url="http://example.com/2"),
        _doc("D3", "C", 0.6, url="http://other.com/1"),
    ]
    result = mmr_rerank(docs, topk=2, lambda_=0.5)
    assert result[0].id == "D1"  # highest score always first
    assert result[1].id == "D3"  # D3 wins over D2 despite lower score


def test_mmr_rerank_topk_limits_output():
    docs = [_doc(f"D{i}", f"T{i}", 1.0 - i * 0.1) for i in range(10)]
    result = mmr_rerank(docs, topk=3, lambda_=0.7)
    assert len(result) == 3


def test_mmr_rerank_fewer_docs_than_topk():
    docs = [_doc("D1", "A", 0.9), _doc("D2", "B", 0.5)]
    result = mmr_rerank(docs, topk=5, lambda_=0.7)
    assert len(result) == 2


def test_mmr_rerank_empty_input():
    assert mmr_rerank([], topk=5) == []


def test_mmr_rerank_title_used_as_source_when_no_url():
    # Two docs with same title should share source identity.
    docs = [
        _doc("D1", "Same Title", 0.9),
        _doc("D2", "Same Title", 0.8),
        _doc("D3", "Different Title", 0.6),
    ]
    result = mmr_rerank(docs, topk=2, lambda_=0.5)
    assert result[0].id == "D1"
    assert result[1].id == "D3"  # different source wins over same-title D2
