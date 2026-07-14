from __future__ import annotations

import asyncio

from src.context import ChatMessage, ContextDocument
from src.context.search import SearchResult
from src.internal.search_pipeline.models import (
    CandidateSet,
    GeneratedAnswer,
    RankedEvidence,
)
from src.internal.search_pipeline.stages import (
    FusionRankingStage,
    InferenceStage,
    RankingStage,
    RerankHTTPRankingStage,
    RetrievalStage,
    SearchClientRetrievalStage,
    ServingInferenceStage,
)


class _SearchClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None, dict | None]] = []

    async def retrieve_one(self, query, topk=None, filters=None):
        self.calls.append((query, topk, filters))
        return [
            SearchResult(
                contents="body",
                title="title",
                score=0.75,
                metadata={"source_provider": "retrieval", "chunk_id": "c1"},
            )
        ]


class _ServerManager:
    def __init__(self) -> None:
        self.calls = []

    async def generate(self, request_id, prompt_ids, sampling_params):
        self.calls.append((request_id, prompt_ids, sampling_params))
        return [8, 9]


class _HTTPResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "result": [
                [
                    {"document": {"_idx": "1"}, "score": 0.95},
                    {"document": {"_idx": "0"}, "score": 0.4},
                ]
            ]
        }


class _HTTPClient:
    calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, *, json, timeout):
        self.calls.append((url, json, timeout))
        return _HTTPResponse()


def test_candidate_set_serialization_preserves_scores_filters_and_provider_metadata():
    candidates = CandidateSet(
        query="q",
        candidates=[SearchResult(contents="body", score=0.4, metadata={"id": "1"})],
        provider="retrieval",
        filters={"document_sets": ["engineering"]},
        metadata={"mode": "hybrid"},
    )

    payload = candidates.to_dict()

    assert payload["candidates"][0]["score"] == 0.4
    assert payload["candidates"][0]["metadata"] == {"id": "1"}
    assert payload["filters"] == {"document_sets": ["engineering"]}
    assert payload["provider"] == "retrieval"
    assert payload["metadata"] == {"mode": "hybrid"}


def test_ranked_evidence_and_generated_answer_serialize_citations():
    evidence = ContextDocument(
        id="D1", title="Title", content="Body", score=0.9, metadata={"provider": "web"}
    )
    ranked = RankedEvidence(query="q", evidence=[evidence], metadata={"ranking": "rrf"})
    answer = GeneratedAnswer(
        answer="Answer [D1]", citations=["D1"], metadata={"model": "fake"}
    )

    assert ranked.to_dict()["evidence"][0]["citation"] == "[D1]"
    assert ranked.to_dict()["evidence"][0]["score"] == 0.9
    assert answer.to_dict() == {
        "answer": "Answer [D1]",
        "citations": ["D1"],
        "metadata": {"model": "fake"},
    }


def test_search_client_adapter_forwards_filters_and_normalizes_candidates():
    client = _SearchClient()
    stage = SearchClientRetrievalStage(client, provider="internal")

    result = asyncio.run(
        stage.retrieve(
            "retrieval q",
            [ChatMessage(role="user", content="original q")],
            {"access_acl": ["team-a"]},
            7,
        )
    )

    assert client.calls == [("retrieval q", 7, {"access_acl": ["team-a"]})]
    assert result.provider == "internal"
    assert result.candidates[0].score == 0.75
    assert result.candidates[0].metadata["chunk_id"] == "c1"
    assert result.metadata["history_messages"] == 1


def test_serving_adapter_builds_prompt_and_extracts_only_valid_citations():
    manager = _ServerManager()
    evidence = RankedEvidence(
        query="q",
        evidence=[ContextDocument(id="D1", title="T", content="B")],
    )
    stage = ServingInferenceStage(
        manager,
        encode=lambda text: [len(text)],
        decode=lambda ids: "Grounded [D1], dangling [D9]",
        sampling_params={"temperature": 0},
    )

    result = asyncio.run(
        stage.generate("q", [ChatMessage(role="user", content="prior")], evidence)
    )

    assert result.answer == "Grounded [D1], dangling [D9]"
    assert result.citations == ["D1"]
    assert manager.calls[0][1][0] > 0
    assert manager.calls[0][2] == {"temperature": 0}


def test_fusion_adapter_preserves_each_candidates_provider_metadata():
    primary = CandidateSet(
        query="q",
        candidates=[SearchResult(contents="internal", score=0.8)],
        provider="retrieval",
    )
    web = CandidateSet(
        query="q",
        candidates=[SearchResult(contents="web", score=0.7)],
        provider="serpapi",
    )

    result = asyncio.run(FusionRankingStage([web]).rank("q", primary, 2))

    assert {doc.metadata["source_provider"] for doc in result.evidence} == {
        "retrieval",
        "serpapi",
    }


def test_rerank_http_adapter_preserves_original_fields_and_translates_payload(
    monkeypatch,
):
    _HTTPClient.calls = []
    monkeypatch.setattr(
        "src.internal.search_pipeline.stages.httpx.AsyncClient", _HTTPClient
    )
    candidates = CandidateSet(
        query="q",
        candidates=[
            SearchResult(contents="one", title="One", score=0.2),
            SearchResult(
                contents="two",
                title="Two",
                url="https://two",
                score=0.3,
                metadata={"chunk_id": "c2"},
            ),
        ],
        provider="retrieval",
    )

    result = asyncio.run(
        RerankHTTPRankingStage("http://reranker/").rank("q", candidates, 2)
    )

    assert [doc.title for doc in result.evidence] == ["Two", "One"]
    assert result.evidence[0].score == 0.95
    assert result.evidence[0].url == "https://two"
    assert result.evidence[0].metadata == {
        "chunk_id": "c2",
        "source_provider": "retrieval",
    }
    url, payload, timeout = _HTTPClient.calls[0]
    assert url == "http://reranker/rerank"
    assert payload["documents"][0][1]["document"]["_idx"] == "1"
    assert payload["rerank_topk"] == 2
    assert timeout == 10.0


def test_adapters_satisfy_runtime_stage_protocols():
    assert isinstance(SearchClientRetrievalStage(_SearchClient()), RetrievalStage)
    assert isinstance(FusionRankingStage(), RankingStage)
    assert isinstance(
        ServingInferenceStage(
            _ServerManager(), encode=lambda _: [], decode=lambda _: ""
        ),
        InferenceStage,
    )
