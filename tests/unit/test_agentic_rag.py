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
    # LLM: decompose → ["what is FAISS"], hyde → "FAISS is a lib", sufficiency → "yes"
    llm = _llm_responses(
        "what is FAISS", "FAISS is a vector search library.", "yes", "Answer text [D1]."
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
    # decompose, hyde, sufficiency round 1 → "no", follow-up, sufficiency round 2 → "yes", answer
    llm = _llm_responses(
        "sub-query 1",  # decompose
        "HyDE text",  # hyde
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

    llm = _llm_responses("sub", "hyde", "no", "follow-up", "yes", "answer")
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
