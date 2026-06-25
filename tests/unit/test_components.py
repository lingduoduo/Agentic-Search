"""Unit tests for the explicit search-loop components (T-A.2, T-A.3).

Each component is exercised in isolation with injected/fake dependencies.
"""

from __future__ import annotations

from src.agents.state import Retriever, SearchAgentState
from src.context.search import AgentContext, SearchContext, SearchResult


def _result(
    text: str = "x" * 20, score: float = 0.5, title: str | None = None
) -> SearchResult:
    return SearchResult(contents=text, score=score, title=title)


# --------------------------------------------------------------------------- #
# EvidenceJudge (T-A.2)
# --------------------------------------------------------------------------- #


def test_evidence_judge_empty_round_scores_zero_and_insufficient() -> None:
    from src.agents.components.evidence_judge import EvidenceJudge

    verdict = EvidenceJudge().judge([])

    assert verdict.score == 0.0
    assert verdict.is_sufficient is False


def test_evidence_judge_score_within_unit_interval() -> None:
    from src.agents.components.evidence_judge import EvidenceJudge

    ctx = SearchContext(query="q", results=[_result(score=10.0), _result(score=3.0)])
    verdict = EvidenceJudge().judge([ctx])

    assert 0.0 <= verdict.score <= 1.0


def test_evidence_judge_score_monotonic_with_result_scores() -> None:
    from src.agents.components.evidence_judge import EvidenceJudge

    judge = EvidenceJudge()
    weak = SearchContext(query="q", results=[_result(score=0.1)])
    strong = SearchContext(query="q", results=[_result(score=50.0)])

    assert judge.judge([strong]).score > judge.judge([weak]).score


def test_evidence_judge_mirrors_evaluator_sufficiency() -> None:
    from src.agents.components.evidence_judge import EvidenceJudge

    # Two results with enough content clears the default min_total_results=2 gate.
    sufficient = SearchContext(query="q", results=[_result(), _result()])
    verdict = EvidenceJudge().judge([sufficient])

    assert verdict.is_sufficient is True


def test_evidence_judge_update_state_writes_evidence_score() -> None:
    from src.agents.components.evidence_judge import EvidenceJudge

    state = SearchAgentState(question="q")
    ctx = SearchContext(query="q", results=[_result(score=5.0), _result(score=5.0)])

    verdict = EvidenceJudge().update_state(state, [ctx])

    assert state.evidence_score == verdict.score
    assert state.evidence_score > 0.0


# --------------------------------------------------------------------------- #
# AnswerGenerator (T-A.2)
# --------------------------------------------------------------------------- #


def _ctx_one_round() -> AgentContext:
    ctx = AgentContext()
    ctx.add_round(
        ["q1"],
        [
            [
                _result(text="alpha doc body", title="A"),
                _result(text="beta doc body", title="B"),
            ]
        ],
    )
    return ctx


def test_answer_generator_builds_citation_for_referenced_key() -> None:
    from src.agents.components.answer_generator import AnswerGenerator

    ctx = _ctx_one_round()
    result = AnswerGenerator().generate("The answer is grounded [R1Q1D2].", ctx)

    assert [c.doc_id for c in result.citations] == ["R1Q1D2"]
    assert result.citations[0].marker == "[R1Q1D2]"
    assert result.citations[0].text == "beta doc body"


def test_answer_generator_ignores_unknown_citation_keys() -> None:
    from src.agents.components.answer_generator import AnswerGenerator

    ctx = _ctx_one_round()
    result = AnswerGenerator().generate("Bogus cite [R9Q9D9].", ctx)

    assert result.citations == []


def test_answer_generator_no_citations_when_none_referenced() -> None:
    from src.agents.components.answer_generator import AnswerGenerator

    ctx = _ctx_one_round()
    result = AnswerGenerator().generate("Plain answer, no citations.", ctx)

    assert result.citations == []
    assert result.answer == "Plain answer, no citations."


def test_answer_generator_update_state_sets_citations() -> None:
    from src.agents.components.answer_generator import AnswerGenerator

    state = SearchAgentState(question="q")
    ctx = _ctx_one_round()

    AnswerGenerator().update_state(state, "Grounded [R1Q1D1].", ctx)

    assert [c.doc_id for c in state.citations] == ["R1Q1D1"]


# --------------------------------------------------------------------------- #
# SearchTool (T-A.3)
# --------------------------------------------------------------------------- #


async def test_search_tool_records_query_and_docs_into_state() -> None:
    from src.agents.components.search_tool import SearchTool

    async def fake_retrieve(query: str) -> list[SearchResult]:
        return [_result(title="a"), _result(title="b")]

    state = SearchAgentState(question="q")
    docs = await SearchTool(fake_retrieve).run(state, "faiss index")

    assert [d.title for d in docs] == ["a", "b"]
    assert state.previous_queries == ["faiss index"]
    assert [d.title for d in state.retrieved_docs] == ["a", "b"]
    assert state.search_rounds == 1


async def test_search_tool_accumulates_across_rounds() -> None:
    from src.agents.components.search_tool import SearchTool

    async def fake_retrieve(query: str) -> list[SearchResult]:
        return [_result(title=query)]

    tool = SearchTool(fake_retrieve)
    state = SearchAgentState(question="q")
    await tool.run(state, "first")
    await tool.run(state, "second")

    assert state.previous_queries == ["first", "second"]
    assert state.search_rounds == 2


async def test_search_tool_routes_to_web_when_requested() -> None:
    from src.agents.components.search_tool import SearchTool

    async def vdb(query: str) -> list[SearchResult]:
        return [_result(title="vdb")]

    async def web(query: str) -> list[SearchResult]:
        return [_result(title="web")]

    state = SearchAgentState(question="q")
    docs = await SearchTool(vdb, web_fn=web).run(state, "q", retriever=Retriever.WEB)

    assert [d.title for d in docs] == ["web"]


async def test_search_tool_defaults_to_vector_db() -> None:
    from src.agents.components.search_tool import SearchTool

    async def vdb(query: str) -> list[SearchResult]:
        return [_result(title="vdb")]

    async def web(query: str) -> list[SearchResult]:
        return [_result(title="web")]

    state = SearchAgentState(question="q")
    docs = await SearchTool(vdb, web_fn=web).run(state, "q")

    assert [d.title for d in docs] == ["vdb"]


async def test_search_tool_degrades_to_vdb_when_web_unavailable() -> None:
    from src.agents.components.search_tool import SearchTool

    async def vdb(query: str) -> list[SearchResult]:
        return [_result(title="vdb")]

    state = SearchAgentState(question="q")
    # No web_fn configured -> WEB request degrades to the vector-DB backend.
    docs = await SearchTool(vdb).run(state, "q", retriever=Retriever.WEB)

    assert [d.title for d in docs] == ["vdb"]
    assert state.search_rounds == 1


async def test_search_tool_caches_repeated_query() -> None:
    from src.agents.components.search_tool import SearchTool

    calls = {"n": 0}

    async def vdb(query: str) -> list[SearchResult]:
        calls["n"] += 1
        return [_result(title="vdb")]

    tool = SearchTool(vdb)
    state = SearchAgentState(question="q")
    await tool.run(state, "faiss index")
    await tool.run(state, "  FAISS   index ")  # same query, different spacing/case

    assert calls["n"] == 1  # second call served from cache, no extra backend hit


async def test_search_tool_caches_per_backend() -> None:
    from src.agents.components.search_tool import SearchTool

    vdb_calls = {"n": 0}
    web_calls = {"n": 0}

    async def vdb(query: str) -> list[SearchResult]:
        vdb_calls["n"] += 1
        return [_result(title="vdb")]

    async def web(query: str) -> list[SearchResult]:
        web_calls["n"] += 1
        return [_result(title="web")]

    tool = SearchTool(vdb, web_fn=web)
    state = SearchAgentState(question="q")
    await tool.run(state, "q", retriever=Retriever.VECTOR_DB)
    await tool.run(state, "q", retriever=Retriever.WEB)  # same text, other backend

    assert vdb_calls["n"] == 1
    assert web_calls["n"] == 1  # different backend → not a cache hit


async def test_search_tool_degrades_to_vdb_when_web_raises() -> None:
    from src.agents.components.search_tool import SearchTool

    async def vdb(query: str) -> list[SearchResult]:
        return [_result(title="vdb")]

    async def web(query: str) -> list[SearchResult]:
        raise RuntimeError("web backend exploded")

    state = SearchAgentState(question="q")
    docs = await SearchTool(vdb, web_fn=web).run(state, "q", retriever=Retriever.WEB)

    assert [d.title for d in docs] == ["vdb"]
    assert state.search_rounds == 1


# --------------------------------------------------------------------------- #
# Planner (T-B.1)
# --------------------------------------------------------------------------- #


def test_planner_parses_web_search() -> None:
    from src.agents.components.planner import Planner, SearchAction

    decision = Planner().decide('<search retriever="web">latest news</search>')

    assert isinstance(decision, SearchAction)
    assert decision.query == "latest news"
    assert decision.retriever is Retriever.WEB


def test_planner_search_defaults_to_vector_db_without_attribute() -> None:
    from src.agents.components.planner import Planner, SearchAction

    decision = Planner().decide("<search>what is faiss</search>")

    assert isinstance(decision, SearchAction)
    assert decision.retriever is Retriever.VECTOR_DB


def test_planner_unknown_retriever_value_falls_back_to_vector_db() -> None:
    from src.agents.components.planner import Planner, SearchAction

    decision = Planner().decide('<search retriever="quantum">q</search>')

    assert isinstance(decision, SearchAction)
    assert decision.retriever is Retriever.VECTOR_DB


def test_planner_parses_rerank_action() -> None:
    from src.agents.components.planner import Planner, RerankAction

    assert isinstance(Planner().decide("<rerank/>"), RerankAction)


def test_planner_parses_answer_action() -> None:
    from src.agents.components.planner import AnswerAction, Planner

    decision = Planner().decide("<answer>The answer is 42 [R1Q1D1].</answer>")

    assert isinstance(decision, AnswerAction)
    assert decision.text == "The answer is 42 [R1Q1D1]."


def test_planner_search_takes_precedence_over_answer() -> None:
    from src.agents.components.planner import Planner, SearchAction

    decision = Planner().decide("<search>more</search> then <answer>x</answer>")

    assert isinstance(decision, SearchAction)


def test_planner_malformed_text_defaults_to_vector_db_search() -> None:
    from src.agents.components.planner import Planner, SearchAction

    decision = Planner().decide("no tags here at all")

    assert isinstance(decision, SearchAction)
    assert decision.retriever is Retriever.VECTOR_DB


def test_planner_flags_duplicate_query() -> None:
    from src.agents.components.planner import Planner, SearchAction

    decision = Planner().decide(
        "<search>what is faiss</search>", previous_queries=["what is faiss"]
    )

    assert isinstance(decision, SearchAction)
    assert decision.query == "what is faiss"
    assert decision.is_duplicate is True


def test_planner_new_query_not_flagged_duplicate() -> None:
    from src.agents.components.planner import Planner, SearchAction

    decision = Planner().decide(
        "<search>brand new</search>", previous_queries=["what is faiss"]
    )

    assert isinstance(decision, SearchAction)
    assert decision.is_duplicate is False


def test_planner_duplicate_match_ignores_whitespace_and_case() -> None:
    from src.agents.components.planner import Planner

    decision = Planner().decide(
        "<search>  What  Is   FAISS </search>", previous_queries=["what is faiss"]
    )

    assert decision.is_duplicate is True


def test_planner_fallback_query_is_bounded() -> None:
    from src.agents.components.planner import Planner, SearchAction

    raw = "first line of reasoning\n" + ("x" * 1000)
    decision = Planner().decide(raw)

    assert isinstance(decision, SearchAction)
    assert decision.query == "first line of reasoning"
    assert len(decision.query) <= 256


# --------------------------------------------------------------------------- #
# RerankerTool (T-A.3)
# --------------------------------------------------------------------------- #


def test_reranker_tool_reorders_state_docs_without_counting_a_round() -> None:
    from src.agents.components.reranker_tool import RerankerTool

    def fake_rerank(query: str, docs: list[SearchResult]) -> list[SearchResult]:
        return sorted(docs, key=lambda d: d.score, reverse=True)

    state = SearchAgentState(question="q")
    state.record_search(
        "q", [_result(title="lo", score=0.1), _result(title="hi", score=0.9)]
    )

    reordered = RerankerTool(fake_rerank).run(state)

    assert [d.title for d in reordered] == ["hi", "lo"]
    assert [d.title for d in state.retrieved_docs] == ["hi", "lo"]
    assert state.search_rounds == 1  # rerank is not a retriever call


def test_reranker_tool_handles_empty_docs() -> None:
    from src.agents.components.reranker_tool import RerankerTool

    def fake_rerank(query: str, docs: list[SearchResult]) -> list[SearchResult]:
        return docs

    state = SearchAgentState(question="q")
    assert RerankerTool(fake_rerank).run(state) == []


def test_reranker_tool_skips_when_single_doc() -> None:
    from src.agents.components.reranker_tool import RerankerTool

    called = {"n": 0}

    def fake_rerank(query: str, docs: list[SearchResult]) -> list[SearchResult]:
        called["n"] += 1
        return docs

    state = SearchAgentState(question="q")
    state.record_search("q", [_result(title="only")])

    reordered = RerankerTool(fake_rerank).run(state)

    assert called["n"] == 0  # nothing to reorder with one doc
    assert [d.title for d in reordered] == ["only"]


def test_reranker_tool_limits_to_max_candidates() -> None:
    from src.agents.components.reranker_tool import RerankerTool

    seen_lengths: list[int] = []

    def fake_rerank(query: str, docs: list[SearchResult]) -> list[SearchResult]:
        seen_lengths.append(len(docs))
        return list(reversed(docs))

    state = SearchAgentState(question="q")
    state.record_search(
        "q",
        [
            _result(title="a"),
            _result(title="b"),
            _result(title="c"),
            _result(title="d"),
        ],
    )

    reordered = RerankerTool(fake_rerank, max_candidates=2).run(state)

    assert seen_lengths == [2]  # only the top-2 were handed to the reranker
    # top-2 [a, b] reversed -> [b, a], tail [c, d] preserved
    assert [d.title for d in reordered] == ["b", "a", "c", "d"]
    assert [d.title for d in state.retrieved_docs] == ["b", "a", "c", "d"]
