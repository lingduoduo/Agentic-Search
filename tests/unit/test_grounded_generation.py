"""Guarded answer-generation behavior."""

from __future__ import annotations

import json

import src.context as context_api
from src.context import (
    CANONICAL_ABSTENTION,
    AnswerGenerationRequest,
    ChatMessage,
    LLMResponse,
    VerificationStatus,
    build_context_bundle,
    generate_answer,
)
from src.context.search import SearchResult


class SequenceLLM:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls: list[list[object]] = []

    def complete(self, messages, **kwargs):
        self.calls.append(messages)
        return LLMResponse(self.responses.pop(0))


def _bundle():
    return build_context_bundle(
        "What is FAISS?",
        [
            SearchResult(
                title="FAISS",
                contents='"FAISS"\nFAISS is a library for efficient vector similarity search.',
                score=0.9,
            )
        ],
    )


def _draft(*claims: tuple[str, list[str]], abstain: bool = False) -> str:
    return json.dumps(
        {
            "claims": [
                {"text": text, "evidence_ids": evidence_ids}
                for text, evidence_ids in claims
            ],
            "missing_information": [],
            "abstain": abstain,
        }
    )


def test_guarded_generation_renders_valid_supported_draft():
    llm = SequenceLLM(_draft(("FAISS enables vector similarity search.", ["D1"])))

    result = generate_answer(
        AnswerGenerationRequest(question="What is FAISS?", context=_bundle()), llm=llm
    )

    assert result.answer == "FAISS enables vector similarity search. [D1]"
    assert result.citations == ["D1"]
    assert result.confidence > 0
    assert result.verification_status is VerificationStatus.VERIFIED
    assert result.abstained is False
    assert len(llm.calls) == 1


def test_guarded_generation_retries_once_with_verifier_feedback():
    llm = SequenceLLM(
        _draft(("Tomorrow's forecast predicts rain.", ["D1"])),
        _draft(("FAISS enables vector similarity search.", ["D1"])),
    )

    result = generate_answer(
        AnswerGenerationRequest(question="What is FAISS?", context=_bundle()), llm=llm
    )

    assert result.answer == "FAISS enables vector similarity search. [D1]"
    assert len(llm.calls) == 2
    retry_text = "\n".join(message.content for message in llm.calls[1])
    assert "Tomorrow's forecast predicts rain" in retry_text
    assert "insufficient lexical support" in retry_text


def test_guarded_initial_prompt_preserves_history_before_current_request():
    history = [
        ChatMessage(role="user", content="Earlier user question"),
        ChatMessage(role="assistant", content="Earlier assistant answer"),
    ]
    llm = SequenceLLM(_draft(("FAISS enables vector similarity search.", ["D1"])))

    generate_answer(
        AnswerGenerationRequest(
            question="What is FAISS?", context=_bundle(), chat_history=history
        ),
        llm=llm,
    )

    messages = llm.calls[0]
    assert [message.role for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert [message.content for message in messages[1:3]] == [
        "Earlier user question",
        "Earlier assistant answer",
    ]
    assert messages[-1].content.startswith("Question:\nWhat is FAISS?")
    assert "evidence_ids" in messages[0].content


def test_guarded_corrective_prompt_preserves_history_before_current_request():
    history = [
        ChatMessage(role="user", content="Earlier user question"),
        ChatMessage(role="assistant", content="Earlier assistant answer"),
    ]
    llm = SequenceLLM(
        _draft(("Tomorrow's forecast predicts rain.", ["D1"])),
        _draft(("FAISS enables vector similarity search.", ["D1"])),
    )

    generate_answer(
        AnswerGenerationRequest(
            question="What is FAISS?", context=_bundle(), chat_history=history
        ),
        llm=llm,
    )

    messages = llm.calls[1]
    assert [message.role for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert [message.content for message in messages[1:3]] == [
        "Earlier user question",
        "Earlier assistant answer",
    ]
    assert messages[-1].content.startswith("Question:\nWhat is FAISS?")
    assert "Verifier feedback:" in messages[-1].content
    assert "evidence_ids" in messages[0].content


def test_guarded_generation_removes_unsupported_claims_after_retry():
    mixed = _draft(
        ("FAISS enables vector similarity search.", ["D1"]),
        ("Tomorrow's forecast predicts rain.", ["D1"]),
    )
    llm = SequenceLLM(mixed, mixed)

    result = generate_answer(
        AnswerGenerationRequest(question="What is FAISS?", context=_bundle()), llm=llm
    )

    assert result.answer == "FAISS enables vector similarity search. [D1]"
    assert "weather" not in result.answer
    assert result.verification_status is VerificationStatus.PARTIAL
    assert len(llm.calls) == 2


def test_guarded_generation_retries_malformed_output_then_abstains():
    llm = SequenceLLM("not json", "still not json")

    result = generate_answer(
        AnswerGenerationRequest(question="What is FAISS?", context=_bundle()), llm=llm
    )

    assert result.answer == CANONICAL_ABSTENTION
    assert result.confidence == 0.0
    assert result.verification_status is VerificationStatus.ABSTAINED
    assert result.abstained is True
    assert len(llm.calls) == 2


def test_guarded_generation_honors_explicit_total_abstention():
    llm = SequenceLLM(_draft(abstain=True))

    result = generate_answer(
        AnswerGenerationRequest(question="What is FAISS?", context=_bundle()), llm=llm
    )

    assert result.answer == CANONICAL_ABSTENTION
    assert result.abstained is True
    assert len(llm.calls) == 1


def test_guarded_generation_abstains_without_calling_llm_when_evidence_is_empty():
    empty = build_context_bundle("What is FAISS?", [])
    llm = SequenceLLM(_draft(("Invented answer.", ["D1"])))

    result = generate_answer(
        AnswerGenerationRequest(question="What is FAISS?", context=empty), llm=llm
    )

    assert result.answer == CANONICAL_ABSTENTION
    assert result.abstained is True
    assert llm.calls == []


def test_guarded_generation_never_exceeds_two_llm_calls():
    llm = SequenceLLM("bad", "bad", _draft(("FAISS is a library.", ["D1"])))

    generate_answer(
        AnswerGenerationRequest(
            question="What is FAISS?",
            context=_bundle(),
            grounded_generation=context_api.GroundedGenerationConfig(max_retries=10),
        ),
        llm=llm,
    )

    assert len(llm.calls) == 2


def test_disabling_guard_preserves_legacy_free_text_generation():
    llm = SequenceLLM("Legacy free text [D1].")

    result = generate_answer(
        AnswerGenerationRequest(
            question="What is FAISS?",
            context=_bundle(),
            grounded_generation=context_api.GroundedGenerationConfig(enabled=False),
        ),
        llm=llm,
    )

    assert result.answer == "Legacy free text [D1]."
    assert result.citations == ["D1"]
    assert result.confidence is None
    assert result.verification_status is None
    assert result.abstained is False


def test_structured_and_corrective_prompts_require_evidence_ids_and_uncertainty():
    from src.context import (
        build_corrective_answer_prompt,
        build_structured_answer_prompt,
    )

    initial = build_structured_answer_prompt("What is FAISS?", _bundle())
    corrective = build_corrective_answer_prompt(
        "What is FAISS?",
        _bundle(),
        original_draft="bad draft",
        verifier_feedback="unsupported claim",
    )

    assert "evidence_ids" in initial.system
    assert "uncertain" in (initial.system + initial.user).lower()
    assert "bad draft" in corrective.user
    assert "unsupported claim" in corrective.user
    assert "remove unsupported" in corrective.user.lower()


def test_extractive_fallback_abstains_without_relevant_snippet():
    unrelated = build_context_bundle(
        "What is FAISS?",
        [SearchResult(title="Cooking", contents="Bake the bread for thirty minutes.")],
    )

    result = generate_answer(
        AnswerGenerationRequest(question="What is FAISS?", context=unrelated), llm=None
    )

    assert result.answer == CANONICAL_ABSTENTION
    assert result.abstained is True


def test_result_metadata_defaults_remain_backward_compatible():
    from src.context.models import AnswerGenerationResult
    from src.context.prompts import build_answer_prompt

    bundle = _bundle()
    result = AnswerGenerationResult(
        answer="answer",
        citations=[],
        context=bundle,
        prompt=build_answer_prompt("q", bundle),
    )

    assert result.confidence is None
    assert result.verification_status is None
    assert result.abstained is False
    assert result.tool_evidence == []
