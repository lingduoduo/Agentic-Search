"""Time-to-first-token and time-to-first-claim on the grounded answer path."""

from __future__ import annotations

import time

from src.context.models import (
    AnswerGenerationRequest,
    ContextDocument,
    SearchContextBundle,
)
from src.context.pipeline import generate_answer

_DRAFT = (
    '{"abstain": false, "missing_information": [], "claims": '
    '[{"text": "FAISS is a vector search library", "evidence_ids": ["D1"]}]}'
)


class _StreamingLLM:
    """Streams the draft in pieces, slowly enough to time."""

    structured_output_capability = "prompt_only"

    def complete(self, messages, **kw):
        return _DRAFT

    def stream_complete(self, messages, **kw):
        time.sleep(0.02)  # provider think time before the first token
        for i in range(0, len(_DRAFT), 40):
            yield _DRAFT[i : i + 40]


def _request() -> AnswerGenerationRequest:
    context = SearchContextBundle(
        query="what is FAISS",
        documents=[
            ContextDocument(
                id="D1",
                title="FAISS",
                content="FAISS is a vector search library for dense retrieval.",
                url=None,
                score=1.0,
                metadata={},
            )
        ],
    )
    return AnswerGenerationRequest(question="what is FAISS", context=context)


def test_streaming_answer_reports_first_token_and_first_claim():
    claims: list[str] = []
    result = generate_answer(_request(), llm=_StreamingLLM(), on_claim=claims.append)

    assert result.timings is not None
    assert result.timings.llm_first_token_ms is not None
    assert result.timings.llm_first_token_ms >= 20
    if claims:
        assert result.timings.time_to_first_claim_ms is not None
        assert (
            result.timings.time_to_first_claim_ms >= result.timings.llm_first_token_ms
        )


def test_non_streaming_answer_reports_no_first_token():
    class _PlainLLM:
        structured_output_capability = "prompt_only"

        def complete(self, messages, **kw):
            return _DRAFT

    result = generate_answer(_request(), llm=_PlainLLM())
    assert result.timings is not None
    assert result.timings.llm_first_token_ms is None
