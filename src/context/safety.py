"""Structured answer parsing and deterministic evidence verification."""

from __future__ import annotations

import json
from typing import Any

from .grounding import _overlap, _tokenize
from .models import (
    AnswerClaim,
    AnswerDraft,
    ClaimVerdict,
    EvidenceSource,
    SearchContextBundle,
    VerificationResult,
    VerificationStatus,
)

CANONICAL_ABSTENTION = "I don't know based on the available evidence."
TIMEOUT_DEGRADED_ANSWER = "I couldn't complete an answer in time. Please try again."


def evidence_from_context(context: SearchContextBundle) -> list[EvidenceSource]:
    """Normalize retrieved documents with stable IDs based on input order."""
    return [
        EvidenceSource(
            id=f"D{index}",
            text=document.content,
            title=document.title,
            url=document.url,
            provenance="retrieval",
            metadata=document.metadata,
        )
        for index, document in enumerate(context.documents, 1)
    ]


def build_claim(raw_claim: object, known_ids: set[str]) -> AnswerClaim:
    """Validate and build one claim from parsed JSON.

    Extracted from `parse_answer_draft`'s loop so a streaming reader parsing a
    partially-received draft can validate a claim using exactly the code the
    full parse uses, rather than a parallel implementation that could drift.
    """
    if not isinstance(raw_claim, dict) or set(raw_claim) != {
        "text",
        "evidence_ids",
    }:
        raise ValueError("each claim must contain exactly text and evidence_ids")
    text = raw_claim["text"]
    evidence_ids = raw_claim["evidence_ids"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("claim text must be a non-empty string")
    if not _is_string_list(evidence_ids) or not evidence_ids:
        raise ValueError("claim evidence_ids must be a non-empty list of strings")
    unknown = sorted(set(evidence_ids) - known_ids)
    if unknown:
        raise ValueError(f"unknown evidence IDs: {', '.join(unknown)}")
    return AnswerClaim(text=text.strip(), evidence_ids=list(evidence_ids))


def parse_answer_draft(
    payload: str | dict[str, object], evidence: list[EvidenceSource]
) -> AnswerDraft:
    """Parse an exact AnswerDraft JSON schema and validate all evidence IDs."""
    try:
        value: Any = json.loads(payload) if isinstance(payload, str) else payload
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("answer draft must be valid JSON") from exc

    if not isinstance(value, dict) or set(value) != {
        "claims",
        "missing_information",
        "abstain",
    }:
        raise ValueError("answer draft must contain exactly the required keys")
    if not isinstance(value["claims"], list):
        raise ValueError("claims must be a list")
    if not _is_string_list(value["missing_information"]):
        raise ValueError("missing_information must be a list of strings")
    if not isinstance(value["abstain"], bool):
        raise ValueError("abstain must be a boolean")

    known_ids = {item.id for item in evidence}
    claims = [build_claim(raw_claim, known_ids) for raw_claim in value["claims"]]

    return AnswerDraft(
        claims=claims,
        missing_information=list(value["missing_information"]),
        abstain=value["abstain"],
    )


def verify_claim(
    claim: AnswerClaim,
    evidence_by_id: dict[str, EvidenceSource],
    *,
    overlap_threshold: float = 0.15,
) -> ClaimVerdict:
    """Verify one claim against the evidence map.

    Extracted from `verify_answer_draft`'s loop so the streaming path can judge a
    claim the moment it arrives using the same code the batch path uses. Verdicts
    carry no cross-claim dependency, which is what makes this safe to split out.
    """
    missing = sorted(set(claim.evidence_ids) - evidence_by_id.keys())
    if missing:
        return ClaimVerdict(
            claim=claim,
            supported=False,
            reason=f"unknown evidence IDs: {', '.join(missing)}",
        )

    claim_tokens = _tokenize(claim.text)
    scores = {
        evidence_id: _overlap(claim_tokens, _tokenize(evidence_by_id[evidence_id].text))
        for evidence_id in claim.evidence_ids
    }
    supported = bool(scores) and all(
        score >= overlap_threshold for score in scores.values()
    )
    return ClaimVerdict(
        claim=claim,
        supported=supported,
        overlap_scores=scores,
        reason=None if supported else "insufficient lexical support",
    )


def verify_answer_draft(
    draft: AnswerDraft,
    evidence: list[EvidenceSource],
    *,
    overlap_threshold: float = 0.15,
    evidence_sufficiency: float | None = None,
    retry_occurred: bool = False,
) -> VerificationResult:
    """Verify every claim and compute deterministic confidence."""
    if draft.abstain:
        return VerificationResult(
            verdicts=[
                ClaimVerdict(
                    claim=claim,
                    supported=False,
                    reason="draft explicitly abstained",
                )
                for claim in draft.claims
            ],
            supported_claims=[],
            unsupported_claims=list(draft.claims),
            unknown_evidence_ids=[],
            confidence=0.0,
            retry_occurred=retry_occurred,
            status=VerificationStatus.ABSTAINED,
        )

    evidence_by_id = {item.id: item for item in evidence}
    verdicts = [
        verify_claim(claim, evidence_by_id, overlap_threshold=overlap_threshold)
        for claim in draft.claims
    ]
    unknown_ids = {
        evidence_id
        for verdict in verdicts
        for evidence_id in set(verdict.claim.evidence_ids) - evidence_by_id.keys()
    }

    supported_claims = [item.claim for item in verdicts if item.supported]
    unsupported_claims = [item.claim for item in verdicts if not item.supported]
    claim_count = len(verdicts)
    support_ratio = len(supported_claims) / claim_count if claim_count else 0.0
    cited_supported_ids = {
        evidence_id for claim in supported_claims for evidence_id in claim.evidence_ids
    }
    evidence_coverage = (
        len(cited_supported_ids) / len(evidence_by_id) if evidence_by_id else 0.0
    )

    if not supported_claims:
        status = VerificationStatus.ABSTAINED
        confidence = 0.0
    else:
        status = (
            VerificationStatus.VERIFIED
            if not unsupported_claims
            else VerificationStatus.PARTIAL
        )
        sufficiency = (
            support_ratio if evidence_sufficiency is None else evidence_sufficiency
        )
        confidence = _clamp(
            0.6 * support_ratio + 0.2 * evidence_coverage + 0.2 * sufficiency
        )

    return VerificationResult(
        verdicts=verdicts,
        supported_claims=supported_claims,
        unsupported_claims=unsupported_claims,
        unknown_evidence_ids=sorted(unknown_ids),
        confidence=confidence,
        retry_occurred=retry_occurred,
        status=status,
    )


def render_claim(claim: AnswerClaim) -> str:
    """Render one claim exactly as it appears in the final answer."""
    return f"{claim.text} {' '.join(f'[{item}]' for item in claim.evidence_ids)}"


def render_claims(claims: list[AnswerClaim]) -> str:
    """Join rendered claims, or abstain when there are none."""
    if not claims:
        return CANONICAL_ABSTENTION
    return " ".join(render_claim(claim) for claim in claims)


def render_verified_answer(result: VerificationResult) -> str:
    """Render only supported claims, or the canonical abstention."""
    return render_claims(result.supported_claims)


def _is_string_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
