from __future__ import annotations

import httpx
import pytest

from src.context import ContextDocument
from src.context.search import SearchResult
from src.internal.search.models import CandidateSet, RankedEvidence
from src.internal.search.ranking import DefaultRankingStage


def _candidates(*rows: tuple[str, str, float]) -> CandidateSet:
    return CandidateSet(
        query="query",
        provider="test",
        candidates=[
            SearchResult(contents=content, title=title, url=url, score=score)
            for title, url, score in rows
            for content in [f"content for {title}"]
        ],
    )


@pytest.mark.asyncio
async def test_rank_removes_duplicates_and_reports_counts():
    candidates = CandidateSet(
        query="query",
        provider="test",
        candidates=[
            SearchResult(contents="same", title="first", url="https://same.test"),
            SearchResult(contents="same", title="duplicate", url="https://same.test"),
            SearchResult(contents="other", title="other", url="https://other.test"),
        ],
    )

    ranked = await DefaultRankingStage().rank("query", candidates, 5)

    assert [doc.title for doc in ranked.evidence] == ["first", "other"]
    assert ranked.metadata["candidate_count"] == 3
    assert ranked.metadata["deduplicated_count"] == 2


@pytest.mark.asyncio
async def test_rank_uses_optional_reranker_order_and_records_operation():
    candidates = _candidates(
        ("first", "https://first.test", 0.9),
        ("second", "https://second.test", 0.8),
    )

    class ReverseReranker:
        async def rank(self, query, candidates, top_k):
            docs = DefaultRankingStage.documents(candidates)
            evidence = [
                ContextDocument(
                    id=doc.id,
                    title=doc.title,
                    content=doc.content,
                    url=doc.url,
                    score=score,
                    metadata=doc.metadata,
                )
                for doc, score in zip(reversed(docs), (1.0, 0.0), strict=True)
            ]
            return RankedEvidence(query=query, evidence=evidence)

    ranked = await DefaultRankingStage(reranker=ReverseReranker()).rank(
        "query", candidates, 2
    )

    assert [doc.title for doc in ranked.evidence] == ["second", "first"]
    assert ranked.metadata["operations"] == ["deduplicate", "external_rerank", "mmr"]
    assert ranked.metadata["rerank_status"] == "applied"


@pytest.mark.asyncio
async def test_rank_applies_mmr_and_truncates_to_top_k():
    candidates = _candidates(
        ("one", "https://one.test", 0.9),
        ("two", "https://two.test", 0.8),
        ("three", "https://three.test", 0.7),
    )

    ranked = await DefaultRankingStage().rank("query", candidates, 2)

    assert len(ranked.evidence) == 2
    assert [doc.metadata["mmr_rank"] for doc in ranked.evidence] == [1, 2]
    assert ranked.metadata["returned_count"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [httpx.ReadTimeout("slow reranker"), RuntimeError("broken reranker")],
    ids=["timeout", "error"],
)
async def test_rank_preserves_pre_rerank_order_when_reranker_degrades(failure):
    candidates = _candidates(
        ("first", "https://first.test", 0.1),
        ("second", "https://second.test", 0.9),
        ("third", "https://third.test", 0.5),
    )

    class FailingReranker:
        async def rank(self, query, candidates, top_k):
            raise failure

    ranked = await DefaultRankingStage(reranker=FailingReranker()).rank(
        "query", candidates, 2
    )

    assert [doc.title for doc in ranked.evidence] == ["first", "second"]
    assert ranked.metadata["rerank_status"] == (
        "timeout" if isinstance(failure, httpx.TimeoutException) else "error"
    )
    assert ranked.metadata["degraded"] is True
