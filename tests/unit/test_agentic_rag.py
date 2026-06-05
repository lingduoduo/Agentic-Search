from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.context.models import ContextDocument, SearchContextBundle
from src.agents.agentic_rag import AgenticRAGConfig, AgenticRAGLoop, AgenticRAGResult


def _make_bundle(doc_ids: list[str], query: str = "q") -> SearchContextBundle:
    docs = [
        ContextDocument(
            id=did, title=f"Title {did}", content=f"Content about {did}.", score=0.9
        )
        for did in doc_ids
    ]
    return SearchContextBundle(query=query, documents=docs)


def _llm_responses(*responses: str) -> MagicMock:
    llm = MagicMock()
    llm.complete.side_effect = list(responses)
    return llm


# ---------------------------------------------------------------------------
# Happy path — sufficient on first round
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_result_on_single_round():
    bundle = _make_bundle(["d1", "d2"])
    # LLM: decompose → ["what is FAISS"], hyde → "FAISS is a lib", step_back, sufficiency → "yes"
    llm = _llm_responses(
        "what is FAISS",
        "FAISS is a vector search library.",
        "broader query",
        "yes",
        "Answer text [D1].",
    )
    config = AgenticRAGConfig(max_rounds=3, topk=5)

    with patch(
        "src.agents.agentic_rag.retrieve_context", AsyncMock(return_value=bundle)
    ):
        loop = AgenticRAGLoop(config, llm=llm)
        result = await loop.run("what is FAISS?")

    assert isinstance(result, AgenticRAGResult)
    assert result.rounds_used == 1
    assert result.answer
    assert result.context.documents


# ---------------------------------------------------------------------------
# Multi-round — insufficient then sufficient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_iterates_when_insufficient():
    bundle = _make_bundle(["d1"])
    # decompose, hyde, step_back, sufficiency round 1 → "no", follow-up, sufficiency round 2 → "yes", answer
    llm = _llm_responses(
        "sub-query 1",  # decompose
        "HyDE text",  # hyde
        "broader question",  # step_back
        "no",  # sufficiency round 1
        "follow-up query",  # follow-up generation
        "yes",  # sufficiency round 2
        "Final answer [D1].",  # generate_answer (llm fallback path)
    )
    config = AgenticRAGConfig(max_rounds=3, topk=5)

    with patch(
        "src.agents.agentic_rag.retrieve_context", AsyncMock(return_value=bundle)
    ):
        loop = AgenticRAGLoop(config, llm=llm)
        result = await loop.run("what is FAISS?")

    assert result.rounds_used == 2


# ---------------------------------------------------------------------------
# Max rounds cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_caps_at_max_rounds():
    bundle = _make_bundle(["d1"])
    # LLM always says "no" → should stop at max_rounds=2
    llm = _llm_responses(
        "sub",  # decompose
        "hyde",  # hyde
        "broader",  # step_back
        "no",  # sufficiency round 1
        "follow-up",  # follow-up
        "no",  # sufficiency round 2 (max_rounds=2 → no check, just synth)
        "answer",  # generate_answer
    )
    config = AgenticRAGConfig(max_rounds=2, topk=5)

    with patch(
        "src.agents.agentic_rag.retrieve_context", AsyncMock(return_value=bundle)
    ):
        loop = AgenticRAGLoop(config, llm=llm)
        result = await loop.run("q?")

    assert result.rounds_used <= 2


# ---------------------------------------------------------------------------
# No LLM — extractive fallback, single round
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_no_llm_returns_extractive_answer():
    bundle = _make_bundle(["d1", "d2"])
    config = AgenticRAGConfig(max_rounds=3, topk=5)

    with patch(
        "src.agents.agentic_rag.retrieve_context", AsyncMock(return_value=bundle)
    ):
        loop = AgenticRAGLoop(config, llm=None)
        result = await loop.run("what is content about d1?")

    assert result.rounds_used == 1
    assert result.answer
    assert result.context.documents


# ---------------------------------------------------------------------------
# Document accumulation across rounds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accumulates_unique_docs_across_rounds():
    bundle_r1 = _make_bundle(["d1"])
    bundle_r2 = _make_bundle(["d2"])
    bundles = [bundle_r1, bundle_r2]

    llm = _llm_responses("sub", "hyde", "broader", "no", "follow-up", "yes", "answer")
    config = AgenticRAGConfig(max_rounds=3, topk=5)

    call_count = 0

    async def _retrieve(*args, **kwargs):
        nonlocal call_count
        b = bundles[min(call_count, len(bundles) - 1)]
        call_count += 1
        return b

    with patch("src.agents.agentic_rag.retrieve_context", side_effect=_retrieve):
        loop = AgenticRAGLoop(config, llm=llm)
        result = await loop.run("multi-hop question?")

    doc_ids = {doc.id for doc in result.context.documents}
    assert "d1" in doc_ids
    assert "d2" in doc_ids


# ---------------------------------------------------------------------------
# Retrieval error — graceful degradation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_handles_retrieval_error_gracefully():
    config = AgenticRAGConfig(max_rounds=2, topk=5)

    with patch(
        "src.agents.agentic_rag.retrieve_context",
        AsyncMock(side_effect=RuntimeError("server down")),
    ):
        loop = AgenticRAGLoop(config, llm=None)
        result = await loop.run("what is FAISS?")

    assert isinstance(result, AgenticRAGResult)
    assert result.rounds_used >= 1  # attempted at least once


# ---------------------------------------------------------------------------
# Seen-query deduplication
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_duplicate_retrieval_queries_across_rounds():
    """Follow-up queries that duplicate earlier queries should not trigger new retrievals."""
    bundle = _make_bundle(["d1"])
    llm = _llm_responses(
        "unique sub-query",  # decompose
        "HyDE text",  # hyde
        "broader question",  # step_back
        "no",  # sufficiency round 1
        "unique sub-query",  # follow-up returns a duplicate → filtered → loop breaks
        "answer",  # generate_answer
    )
    config = AgenticRAGConfig(max_rounds=3, topk=5)
    retrieval_calls: list[str] = []

    async def _track_retrieve(query, **kwargs):
        retrieval_calls.append(query)
        return bundle

    with patch("src.agents.agentic_rag.retrieve_context", side_effect=_track_retrieve):
        loop = AgenticRAGLoop(config, llm=llm)
        result = await loop.run("what is FAISS?")

    assert retrieval_calls.count("unique sub-query") == 1
    assert result.rounds_used >= 1


# ---------------------------------------------------------------------------
# Structured gap analysis
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_follow_up_queries_do_not_duplicate_seen_queries():
    """Structured gap analysis: QUERIES matching already-seen queries terminate the loop."""
    bundle = _make_bundle(["d1"])
    original_question = "what is FAISS?"
    # decompose returns original verbatim → excluded → fallback: sub_queries = [original]
    # so "what is FAISS?" enters seen_queries on round 1
    llm = _llm_responses(
        original_question,  # decompose → original excluded → fallback sub_queries=[original]
        "HyDE text",  # hyde
        "broader",  # step_back
        "no",  # sufficiency round 1
        f"GAPS:\nmissing info\nQUERIES:\n{original_question}",  # follow-up = already seen
        "answer",  # generate_answer
    )
    config = AgenticRAGConfig(max_rounds=3, topk=5)

    with patch(
        "src.agents.agentic_rag.retrieve_context", AsyncMock(return_value=bundle)
    ):
        loop = AgenticRAGLoop(config, llm=llm)
        result = await loop.run(original_question)

    assert isinstance(result, AgenticRAGResult)
