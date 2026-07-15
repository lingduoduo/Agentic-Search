"""Tests for structured, evidence-grounded RAG answer primitives."""

from __future__ import annotations

import json

import pytest

from src.context.models import (
    AnswerClaim,
    AnswerDraft,
    ContextDocument,
    EvidenceSource,
    SearchContextBundle,
    VerificationStatus,
)
from src.context.safety import (
    CANONICAL_ABSTENTION,
    evidence_from_context,
    parse_answer_draft,
    render_verified_answer,
    verify_answer_draft,
)


def _evidence() -> list[EvidenceSource]:
    return [
        EvidenceSource(
            id="D1",
            text="FAISS is a vector similarity search library.",
            title="FAISS overview",
            url="https://example.test/faiss",
            provenance="retrieval",
            metadata={"collection": "docs"},
        ),
        EvidenceSource(
            id="T1",
            text="The service currently has 12 active indexes.",
            title="Index status",
            provenance="tool",
            tool_name="index_status",
        ),
    ]


def test_evidence_from_context_normalizes_documents_with_stable_ids():
    context = SearchContextBundle(
        query="What is FAISS?",
        documents=[
            ContextDocument(
                id="D1",
                title="FAISS overview",
                content="FAISS is a vector similarity search library.",
                url="https://example.test/faiss",
                score=0.9,
                metadata={"collection": "docs"},
            )
        ],
    )

    assert evidence_from_context(context) == [
        EvidenceSource(
            id="D1",
            text="FAISS is a vector similarity search library.",
            title="FAISS overview",
            url="https://example.test/faiss",
            provenance="retrieval",
            metadata={"collection": "docs"},
        )
    ]


def test_parse_answer_draft_accepts_only_the_exact_schema():
    payload = json.dumps(
        {
            "claims": [
                {"text": "FAISS supports similarity search.", "evidence_ids": ["D1"]}
            ],
            "missing_information": ["Its current release version"],
            "abstain": False,
        }
    )

    draft = parse_answer_draft(payload, _evidence())

    assert draft == AnswerDraft(
        claims=[
            AnswerClaim(text="FAISS supports similarity search.", evidence_ids=["D1"])
        ],
        missing_information=["Its current release version"],
        abstain=False,
    )


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        '{"claims": [], "missing_information": [], "abstain": false, "extra": 1}',
        '{"claims": [{"text": "", "evidence_ids": ["D1"]}], "missing_information": [], "abstain": false}',
        '{"claims": [{"text": "A claim", "evidence_ids": []}], "missing_information": [], "abstain": false}',
    ],
)
def test_parse_answer_draft_rejects_malformed_payloads(payload: str):
    with pytest.raises(ValueError):
        parse_answer_draft(payload, _evidence())


def test_parse_answer_draft_rejects_unknown_evidence_ids():
    payload = json.dumps(
        {
            "claims": [{"text": "An invented fact.", "evidence_ids": ["D99"]}],
            "missing_information": [],
            "abstain": False,
        }
    )

    with pytest.raises(ValueError, match="D99"):
        parse_answer_draft(payload, _evidence())


def test_verify_answer_draft_marks_supported_and_unsupported_claims():
    draft = AnswerDraft(
        claims=[
            AnswerClaim(
                text="FAISS is a vector similarity search library.", evidence_ids=["D1"]
            ),
            AnswerClaim(text="FAISS was invented on Mars.", evidence_ids=["D1"]),
        ]
    )

    result = verify_answer_draft(draft, _evidence(), overlap_threshold=0.5)

    assert result.status is VerificationStatus.PARTIAL
    assert result.supported_claims == [draft.claims[0]]
    assert result.unsupported_claims == [draft.claims[1]]
    assert result.verdicts[0].supported is True
    assert result.verdicts[1].supported is False
    assert result.verdicts[1].reason == "insufficient lexical support"


def test_render_verified_answer_includes_only_supported_claims_and_citations():
    draft = AnswerDraft(
        claims=[
            AnswerClaim(
                text="FAISS is a vector similarity search library.", evidence_ids=["D1"]
            ),
            AnswerClaim(text="It was invented on Mars.", evidence_ids=["D1"]),
        ]
    )
    result = verify_answer_draft(draft, _evidence(), overlap_threshold=0.5)

    assert (
        render_verified_answer(result)
        == "FAISS is a vector similarity search library. [D1]"
    )


def test_render_verified_answer_uses_canonical_abstention_without_support():
    result = verify_answer_draft(
        AnswerDraft(
            claims=[AnswerClaim(text="It was invented on Mars.", evidence_ids=["D1"])]
        ),
        _evidence(),
        overlap_threshold=0.5,
    )

    assert result.status is VerificationStatus.ABSTAINED
    assert result.confidence == 0.0
    assert render_verified_answer(result) == CANONICAL_ABSTENTION
    assert CANONICAL_ABSTENTION == "I don't know based on the available evidence."


def test_confidence_is_deterministic_from_support_coverage_and_sufficiency():
    draft = AnswerDraft(
        claims=[
            AnswerClaim(
                text="FAISS is a vector similarity search library.", evidence_ids=["D1"]
            ),
            AnswerClaim(
                text="The service currently has 12 active indexes.", evidence_ids=["T1"]
            ),
            AnswerClaim(text="It was invented on Mars.", evidence_ids=["D1"]),
        ]
    )

    result = verify_answer_draft(
        draft,
        _evidence(),
        overlap_threshold=0.5,
        evidence_sufficiency=0.5,
    )

    # support=2/3, coverage=2/2, sufficiency=1/2
    assert result.confidence == pytest.approx(0.7)


def test_confidence_substitutes_support_ratio_when_sufficiency_is_unavailable():
    draft = AnswerDraft(
        claims=[
            AnswerClaim(
                text="FAISS is a vector similarity search library.", evidence_ids=["D1"]
            ),
            AnswerClaim(text="It was invented on Mars.", evidence_ids=["D1"]),
        ]
    )

    result = verify_answer_draft(draft, _evidence(), overlap_threshold=0.5)

    # support=1/2, coverage=1/2, substituted sufficiency=1/2
    assert result.confidence == pytest.approx(0.5)
