from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.context.models import ContextDocument, SearchContextBundle
from src.agents.search import AgenticRAGConfig, AgenticRAGLoop, AgenticRAGResult


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


_GROUNDED_ANSWER = (
    '{"claims":[{"text":"Content about d1","evidence_ids":["D1"]}],'
    '"missing_information":[],"abstain":false}'
)


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
        _GROUNDED_ANSWER,
    )
    config = AgenticRAGConfig(max_rounds=3, topk=5)

    with patch(
        "src.agents.search.agentic_rag.retrieve_context", AsyncMock(return_value=bundle)
    ):
        loop = AgenticRAGLoop(config, llm=llm)
        result = await loop.run("what is FAISS?")

    assert isinstance(result, AgenticRAGResult)
    assert result.rounds_used == 1
    assert result.answer == "Content about d1 [D1]"
    assert result.citations == ["D1"]
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
        _GROUNDED_ANSWER,  # guarded generate_answer
    )
    config = AgenticRAGConfig(max_rounds=3, topk=5)

    with patch(
        "src.agents.search.agentic_rag.retrieve_context", AsyncMock(return_value=bundle)
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
        _GROUNDED_ANSWER,  # generate_answer
    )
    config = AgenticRAGConfig(max_rounds=2, topk=5)

    with patch(
        "src.agents.search.agentic_rag.retrieve_context", AsyncMock(return_value=bundle)
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
        "src.agents.search.agentic_rag.retrieve_context", AsyncMock(return_value=bundle)
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

    llm = _llm_responses(
        "sub", "hyde", "broader", "no", "follow-up", "yes", _GROUNDED_ANSWER
    )
    config = AgenticRAGConfig(max_rounds=3, topk=5)

    call_count = 0

    async def _retrieve(*args, **kwargs):
        nonlocal call_count
        b = bundles[min(call_count, len(bundles) - 1)]
        call_count += 1
        return b

    with patch("src.agents.search.agentic_rag.retrieve_context", side_effect=_retrieve):
        loop = AgenticRAGLoop(config, llm=llm)
        result = await loop.run("multi-hop question?")

    # After accumulation docs are re-indexed to stable D1..DN IDs.
    assert len(result.context.documents) == 2


# ---------------------------------------------------------------------------
# Retrieval error — graceful degradation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_handles_retrieval_error_gracefully():
    config = AgenticRAGConfig(max_rounds=2, topk=5)

    with patch(
        "src.agents.search.agentic_rag.retrieve_context",
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
        _GROUNDED_ANSWER,  # generate_answer
    )
    config = AgenticRAGConfig(max_rounds=3, topk=5)
    retrieval_calls: list[str] = []

    async def _track_retrieve(query, **kwargs):
        retrieval_calls.append(query)
        return bundle

    with patch(
        "src.agents.search.agentic_rag.retrieve_context", side_effect=_track_retrieve
    ):
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
        _GROUNDED_ANSWER,  # generate_answer
    )
    config = AgenticRAGConfig(max_rounds=3, topk=5)

    with patch(
        "src.agents.search.agentic_rag.retrieve_context", AsyncMock(return_value=bundle)
    ):
        loop = AgenticRAGLoop(config, llm=llm)
        result = await loop.run(original_question)

    assert isinstance(result, AgenticRAGResult)


# ---------------------------------------------------------------------------
# Control-flow trace instrumentation (F3) — chat_loop stages become observable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_emits_control_flow_events_to_recorder():
    from src.agents.core.control_flow_trace import ControlFlowRecorder

    bundle = _make_bundle(["d1", "d2"])
    llm = _llm_responses(
        "what is FAISS",  # decompose
        "FAISS is a vector search library.",  # hyde
        "broader query",  # step_back
        "yes",  # sufficiency → stop
        _GROUNDED_ANSWER,  # synthesis
    )
    recorder = ControlFlowRecorder("req-1", session_id="sess-1")

    with patch(
        "src.agents.search.agentic_rag.retrieve_context", AsyncMock(return_value=bundle)
    ):
        loop = AgenticRAGLoop(AgenticRAGConfig(max_rounds=3, topk=5), llm=llm)
        await loop.run("what is FAISS?", recorder=recorder)

    events = recorder.snapshot()
    components = [e.component for e in events]
    # the loop's stages are now observable in the trace
    assert "query_enhancer" in components
    assert "search_tool" in components
    assert "evidence_judge" in components
    assert "answer_generator" in components
    # answer leg is the terminal stage
    assert components[-1] == "answer_generator"


@pytest.mark.asyncio
async def test_run_with_zero_max_rounds_returns_empty_context():
    llm = _llm_responses("sub", "hyde", "broader", "answer")
    config = AgenticRAGConfig(max_rounds=0, topk=5)
    with patch(
        "src.agents.search.agentic_rag.retrieve_context",
        AsyncMock(return_value=_make_bundle(["d1"])),
    ):
        loop = AgenticRAGLoop(config, llm=llm)
        result = await loop.run("q?")
    assert isinstance(result, AgenticRAGResult)
    assert result.rounds_used == 0
    assert result.context.documents == []


@pytest.mark.asyncio
async def test_case_and_whitespace_variants_retrieve_once():
    bundle = _make_bundle(["d1"])
    # decompose/hyde/step_back produce case/whitespace variants of one query
    llm = _llm_responses(
        "GPT-4 cost",  # decompose
        "gpt-4   cost ",  # hyde (normalizes to same as decompose)
        " GPT-4 Cost",  # step_back (same normalized form)
        "yes",  # sufficiency
        _GROUNDED_ANSWER,  # generate_answer
    )
    config = AgenticRAGConfig(max_rounds=3, topk=5)
    calls: list[str] = []

    async def _track(query, **kwargs):
        calls.append(query)
        return bundle

    with patch("src.agents.search.agentic_rag.retrieve_context", side_effect=_track):
        loop = AgenticRAGLoop(config, llm=llm)
        await loop.run("gpt-4 cost?")

    # all three enhanced queries normalize identically → retrieved once
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_url_less_docs_dedup_by_full_content():
    # first 120 chars identical but full content differs → hash keeps both
    dup_a = ContextDocument(id="a", title="A", content="X" * 200, score=0.9)
    near = ContextDocument(id="b", title="B", content="X" * 120 + "Z" * 80, score=0.8)
    bundle = SearchContextBundle(query="q", documents=[dup_a, near])
    config = AgenticRAGConfig(max_rounds=1, topk=5)
    with patch(
        "src.agents.search.agentic_rag.retrieve_context", AsyncMock(return_value=bundle)
    ):
        loop = AgenticRAGLoop(config, llm=None)
        result = await loop.run("q?")
    assert len(result.context.documents) == 2


@pytest.mark.asyncio
async def test_follow_ups_capped_per_round():
    bundle = _make_bundle(["d1"])
    eight = "\n".join(f"followup query {i}" for i in range(8))
    llm = _llm_responses(
        "sub",
        "hyde",
        "broader",  # enhance
        "no",  # sufficiency round 1
        f"GAPS:\ng\nQUERIES:\n{eight}",  # 8 follow-ups
        "yes",  # sufficiency round 2
        _GROUNDED_ANSWER,  # generate_answer
    )
    config = AgenticRAGConfig(max_rounds=3, topk=5, max_followups_per_round=5)
    calls: list[str] = []

    async def _track(query, **kwargs):
        calls.append(query)
        return bundle

    with patch("src.agents.search.agentic_rag.retrieve_context", side_effect=_track):
        loop = AgenticRAGLoop(config, llm=llm)
        await loop.run("q?")

    followup_calls = [c for c in calls if c.startswith("followup query")]
    assert len(followup_calls) == 5


@pytest.mark.asyncio
async def test_sufficiency_check_times_out_fail_open():
    import time as _time

    def _slow_complete(messages, **kwargs):
        _time.sleep(1.0)  # exceeds the tiny timeout
        return "no"

    llm = MagicMock()
    llm.complete.side_effect = _slow_complete
    config = AgenticRAGConfig(sufficiency_timeout_s=0.05)
    loop = AgenticRAGLoop(config, llm=llm)
    bundle = _make_bundle(["d1"])

    result = await loop._is_sufficient("q?", bundle)
    assert result is True  # fail-open on timeout, and returns promptly


@pytest.mark.asyncio
async def test_generate_followup_times_out_fail_open():
    import time as _time

    def _slow_complete(messages, **kwargs):
        _time.sleep(1.0)  # exceeds the tiny timeout
        return "GAPS:\ng\nQUERIES:\nsome query"

    llm = MagicMock()
    llm.complete.side_effect = _slow_complete
    config = AgenticRAGConfig(sufficiency_timeout_s=0.05)
    loop = AgenticRAGLoop(config, llm=llm)
    bundle = _make_bundle(["d1"])

    result = await loop._generate_followup("q?", bundle)
    assert result == []  # fail-open on timeout, and returns promptly


@pytest.mark.asyncio
async def test_generate_followup_parses_queries_on_success():
    llm = _llm_responses("GAPS:\nmissing bit\nQUERIES:\nfollow-up one\nfollow-up two")
    config = AgenticRAGConfig(sufficiency_timeout_s=5.0)
    loop = AgenticRAGLoop(config, llm=llm)
    bundle = _make_bundle(["d1"])

    result = await loop._generate_followup("q?", bundle)
    assert result == ["follow-up one", "follow-up two"]


@pytest.mark.asyncio
async def test_run_without_recorder_is_unchanged():
    bundle = _make_bundle(["d1"])
    llm = _llm_responses("sub", "hyde", "broader", "yes", _GROUNDED_ANSWER)
    with patch(
        "src.agents.search.agentic_rag.retrieve_context", AsyncMock(return_value=bundle)
    ):
        loop = AgenticRAGLoop(AgenticRAGConfig(max_rounds=3, topk=5), llm=llm)
        result = await loop.run("q")  # no recorder → no crash, same result
    assert result.answer
    assert result.rounds_used == 1


@pytest.mark.asyncio
async def test_run_threads_user_memory_into_answer_request():
    """AgenticRAGLoop.run(user_memory=...) sets it on the AnswerGenerationRequest
    handed to generate_answer (the shared synthesis prompt)."""
    from src.context.models import AnswerGenerationResult, PromptBundle

    bundle = _make_bundle(["d1"])
    llm = MagicMock()
    llm.complete.return_value = "x"  # generic enhancer responses; answer is stubbed
    config = AgenticRAGConfig(max_rounds=1, topk=5)
    captured: dict = {}

    def fake_generate_answer(request, *, llm=None):
        captured["user_memory"] = request.user_memory
        return AnswerGenerationResult(
            answer="ok",
            citations=[],
            context=request.context,
            prompt=PromptBundle(system="", user="", messages=[]),
        )

    with (
        patch(
            "src.agents.search.agentic_rag.retrieve_context",
            AsyncMock(return_value=bundle),
        ),
        patch("src.agents.search.agentic_rag.generate_answer", fake_generate_answer),
    ):
        loop = AgenticRAGLoop(config, llm=llm)
        await loop.run(
            "what is FAISS?", user_memory="\n\nUser memory:\n- allergic to peanuts"
        )

    assert captured["user_memory"] == "\n\nUser memory:\n- allergic to peanuts"
