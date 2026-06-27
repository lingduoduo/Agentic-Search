"""Unit tests for the canonical search fields on AgentState."""

from __future__ import annotations

from src.agents.state import (
    AgentState,
    Citation,
    Retriever,
    UserRequest,
)
from src.context.search import SearchResult


def _doc(doc_id: str, score: float = 1.0) -> SearchResult:
    return SearchResult(contents=f"text-{doc_id}", score=score, title=doc_id)


def _state(question: str = "q") -> AgentState:
    return AgentState(
        request_id="req-1",
        user_request=UserRequest(user_id="u1", channel="test", message=question),
        question=question,
    )


def test_initializes_with_question_and_empty_defaults() -> None:
    state = _state("what is faiss?")

    assert state.question == "what is faiss?"
    assert state.previous_queries == []
    assert state.retrieved_docs == []
    assert state.evidence_score == 0.0
    assert state.search_rounds == 0
    assert state.citations == []


def test_record_search_round_tracks_queries_docs_and_increments_once() -> None:
    state = _state()

    state.record_search_round(["faiss index", "vector index"], [_doc("a"), _doc("b")])

    assert state.previous_queries == ["faiss index", "vector index"]
    assert [d.title for d in state.retrieved_docs] == ["a", "b"]
    assert state.search_rounds == 1


def test_record_search_round_dedupes_queries_but_counts_each_round() -> None:
    state = _state()

    state.record_search_round(["dup", "dup"], [_doc("a")])
    state.record_search_round(["dup"], [_doc("b")])

    assert state.previous_queries == ["dup"]  # deduped, order preserved
    assert [d.title for d in state.retrieved_docs] == [
        "a",
        "b",
    ]  # both rounds' docs kept
    assert state.search_rounds == 2  # each retriever call counts


def test_record_search_round_preserves_distinct_query_order() -> None:
    state = _state()

    state.record_search_round(["first"], [_doc("a")])
    state.record_search_round(["second"], [_doc("b")])

    assert state.previous_queries == ["first", "second"]
    assert state.search_rounds == 2


def test_record_rerank_reorders_docs_without_incrementing_search_rounds() -> None:
    state = _state()
    state.record_search_round(["query"], [_doc("a", 0.1), _doc("b", 0.9)])

    reordered = list(reversed(state.retrieved_docs))
    state.record_rerank(reordered)

    assert [d.title for d in state.retrieved_docs] == ["b", "a"]
    assert state.search_rounds == 1  # rerank is not a retriever call


def test_set_evidence_clamps_to_unit_interval() -> None:
    state = _state()

    state.set_evidence(0.42)
    assert state.evidence_score == 0.42

    state.set_evidence(1.5)
    assert state.evidence_score == 1.0

    state.set_evidence(-0.3)
    assert state.evidence_score == 0.0


def test_set_citations_replaces_citations() -> None:
    state = _state()
    cites = [Citation(doc_id="a", marker="[1]", text="snippet")]

    state.set_citations(cites)

    assert state.citations == cites


def test_question_is_preserved_across_all_operations() -> None:
    state = _state("immutable?")

    state.record_search_round(["q"], [_doc("a")])
    state.record_rerank([_doc("a")])
    state.set_evidence(0.5)
    state.set_citations([Citation(doc_id="a", marker="[1]")])

    assert state.question == "immutable?"


def test_retriever_enum_has_web_and_vector_db() -> None:
    assert Retriever.WEB.value == "web"
    assert Retriever.VECTOR_DB.value == "vector_db"


def test_search_agent_state_is_not_exported() -> None:
    import src.agents.state as state_module

    assert not hasattr(state_module, "SearchAgentState")
    assert "SearchAgentState" not in state_module.__all__
