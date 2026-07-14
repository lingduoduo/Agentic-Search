from __future__ import annotations

import asyncio

from src.context import ChatMessage, ContextDocument
from src.context.search import SearchResult
from src.internal.search_pipeline.models import (
    CandidateSet,
    GeneratedAnswer,
    RankedEvidence,
)
from src.internal.search_pipeline.pipeline import SearchPipeline


class Retrieval:
    def __init__(self, candidates=(), *, error: Exception | None = None, metadata=None):
        self.candidates = list(candidates)
        self.error = error
        self.metadata = metadata or {}
        self.calls = []

    async def retrieve(self, query, history, filters, top_k):
        self.calls.append((query, history, filters, top_k))
        if self.error:
            raise self.error
        return CandidateSet(query, self.candidates, "retrieval", filters, self.metadata)


class Ranking:
    def __init__(self, *, metadata=None):
        self.metadata = metadata or {}

    async def rank(self, query, candidates, top_k):
        documents = [
            ContextDocument.from_search_result(item, index=index)
            for index, item in enumerate(candidates.candidates[:top_k], 1)
        ]
        return RankedEvidence(query, documents, self.metadata)


class Inference:
    def __init__(self, *, error: Exception | None = None):
        self.error = error
        self.calls = 0

    async def generate(self, query, history, evidence):
        self.calls += 1
        if self.error:
            raise self.error
        return GeneratedAnswer("Grounded [D1]", ["D1"])


def _result():
    return SearchResult(
        contents="evidence",
        title="Title",
        url="https://example.test",
        score=0.9,
    )


def test_pipeline_builds_session_context_and_returns_existing_tuple():
    retrieval = Retrieval([_result()])
    pipeline = SearchPipeline(retrieval, Ranking(), Inference())
    history = [ChatMessage(role="user", content="Tell me about FAISS")]

    result = asyncio.run(
        pipeline.run(
            "What about its GPU support?", history, {"owner": "u1"}, 3, "retrieval"
        )
    )

    answer, citations, documents, intent, extra = result
    assert answer == "Grounded [D1]"
    assert citations == ["D1"]
    assert documents[0].title == "Title"
    assert intent == "search"
    assert retrieval.calls[0][0] == "Tell me about FAISS\nWhat about its GPU support?"
    assert retrieval.calls[0][2] == {"owner": "u1"}
    assert extra["source_provider"] == "retrieval"
    assert extra["retrieval_query"] == (
        "Tell me about FAISS\nWhat about its GPU support?"
    )


def test_pipeline_does_not_infer_without_evidence():
    inference = Inference()
    result = asyncio.run(
        SearchPipeline(Retrieval(), Ranking(), inference).run(
            "missing", [], None, 5, "retrieval"
        )
    )

    assert result[0] == "No results found for: missing"
    assert result[1:4] == ([], [], "search")
    assert inference.calls == 0


def test_pipeline_returns_deterministic_unreachable_result():
    result = asyncio.run(
        SearchPipeline(
            Retrieval(error=ConnectionError("down")), Ranking(), Inference()
        ).run("q", [], None, 5, "auto")
    )

    assert result[0] == "No sources are reachable right now. Please try again shortly."
    assert result[1:4] == ([], [], "search")
    assert result[4]["search_fallback"] == "retrieval_unavailable"


def test_pipeline_exposes_reranker_degradation():
    result = asyncio.run(
        SearchPipeline(
            Retrieval([_result()]),
            Ranking(metadata={"rerank_status": "timeout", "degraded": True}),
            Inference(),
        ).run("q", [], None, 5, "retrieval")
    )

    assert result[4]["ranking"]["rerank_status"] == "timeout"
    assert result[4]["ranking"]["degraded"] is True


def test_pipeline_falls_back_to_evidence_when_inference_fails():
    result = asyncio.run(
        SearchPipeline(
            Retrieval([_result()]),
            Ranking(),
            Inference(error=RuntimeError("model down")),
        ).run("q", [], None, 5, "retrieval")
    )

    assert "Title" in result[0]
    assert result[1] == ["[D1]"]
    assert result[4]["inference_fallback"] == "synthesis_failed"
