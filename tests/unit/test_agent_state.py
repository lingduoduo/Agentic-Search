"""Unit tests for SearchAgentState — the six-field search-loop state (T-A.1).

SearchAgentState is the single source of truth threaded through the search
components (Planner, SearchTool, RerankerTool, EvidenceJudge, AnswerGenerator).
It is distinct from the orchestration-level ``AgentState`` in the same module.
"""

from __future__ import annotations

from src.agents.state import (
    Citation,
    Retriever,
    SearchAgentState,
)
from src.context.search import SearchResult


def _doc(doc_id: str, score: float = 1.0) -> SearchResult:
    return SearchResult(contents=f"text-{doc_id}", score=score, title=doc_id)


def test_initializes_with_question_and_empty_defaults() -> None:
    state = SearchAgentState(question="what is faiss?")

    assert state.question == "what is faiss?"
    assert state.previous_queries == []
    assert state.retrieved_docs == []
    assert state.evidence_score == 0.0
    assert state.search_rounds == 0
    assert state.citations == []


def test_record_search_tracks_query_appends_docs_and_increments_round() -> None:
    state = SearchAgentState(question="q")

    state.record_search("faiss index", [_doc("a"), _doc("b")])

    assert state.previous_queries == ["faiss index"]
    assert [d.title for d in state.retrieved_docs] == ["a", "b"]
    assert state.search_rounds == 1


def test_record_search_dedupes_query_but_still_counts_the_round() -> None:
    """A repeated query is not re-listed, but it still consumed a retriever call."""
    state = SearchAgentState(question="q")

    state.record_search("dup", [_doc("a")])
    state.record_search("dup", [_doc("b")])

    assert state.previous_queries == ["dup"]  # deduped, order preserved
    assert [d.title for d in state.retrieved_docs] == [
        "a",
        "b",
    ]  # both rounds' docs kept
    assert state.search_rounds == 2  # each retriever call counts


def test_record_search_preserves_insertion_order_across_distinct_queries() -> None:
    state = SearchAgentState(question="q")

    state.record_search("first", [_doc("a")])
    state.record_search("second", [_doc("b")])

    assert state.previous_queries == ["first", "second"]
    assert state.search_rounds == 2


def test_record_rerank_reorders_docs_without_incrementing_search_rounds() -> None:
    state = SearchAgentState(question="q")
    state.record_search("query", [_doc("a", 0.1), _doc("b", 0.9)])

    reordered = list(reversed(state.retrieved_docs))
    state.record_rerank(reordered)

    assert [d.title for d in state.retrieved_docs] == ["b", "a"]
    assert state.search_rounds == 1  # rerank is not a retriever call


def test_set_evidence_clamps_to_unit_interval() -> None:
    state = SearchAgentState(question="q")

    state.set_evidence(0.42)
    assert state.evidence_score == 0.42

    state.set_evidence(1.5)
    assert state.evidence_score == 1.0

    state.set_evidence(-0.3)
    assert state.evidence_score == 0.0


def test_set_citations_replaces_citations() -> None:
    state = SearchAgentState(question="q")
    cites = [Citation(doc_id="a", marker="[1]", text="snippet")]

    state.set_citations(cites)

    assert state.citations == cites


def test_question_is_preserved_across_all_operations() -> None:
    state = SearchAgentState(question="immutable?")

    state.record_search("q", [_doc("a")])
    state.record_rerank([_doc("a")])
    state.set_evidence(0.5)
    state.set_citations([Citation(doc_id="a", marker="[1]")])

    assert state.question == "immutable?"


def test_retriever_enum_has_web_and_vector_db() -> None:
    assert Retriever.WEB.value == "web"
    assert Retriever.VECTOR_DB.value == "vector_db"
