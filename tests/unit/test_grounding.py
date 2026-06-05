"""Unit tests for the citation grounding verifier."""

from __future__ import annotations

from src.context.grounding import GroundingVerifier
from src.context.models import (
    ContextDocument,
    GroundingReport,
    SearchContextBundle,
)


def _bundle(*doc_contents: str) -> SearchContextBundle:
    docs = [
        ContextDocument(
            id=f"D{i}",
            title=f"Doc {i}",
            content=content,
            score=0.9,
        )
        for i, content in enumerate(doc_contents, 1)
    ]
    return SearchContextBundle(query="test", documents=docs)


# ---------------------------------------------------------------------------
# GroundingVerifier.verify — grounded citations
# ---------------------------------------------------------------------------


def test_verify_grounded_citation():
    bundle = _bundle(
        "FAISS is a vector similarity search library developed by Facebook."
    )
    answer = "FAISS enables fast similarity search. [D1]"
    report = GroundingVerifier().verify(answer, bundle)
    assert len(report.verdicts) == 1
    v = report.verdicts[0]
    assert v.citation == "D1"
    assert v.document_found is True
    assert v.overlap_score > 0.0
    assert v.is_grounded is True


def test_verify_dangling_citation_not_in_context():
    bundle = _bundle("Some content about FAISS.")
    answer = "The model was trained on 1B parameters. [D9]"
    report = GroundingVerifier().verify(answer, bundle)
    assert len(report.verdicts) == 1
    v = report.verdicts[0]
    assert v.citation == "D9"
    assert v.document_found is False
    assert v.overlap_score == 0.0
    assert v.is_grounded is False


def test_verify_dangling_citation_stripped_from_answer_clean():
    bundle = _bundle("FAISS content.")
    answer = "The answer is 42. [D99] More text here."
    report = GroundingVerifier().verify(answer, bundle)
    assert "[D99]" not in report.answer_clean
    assert "The answer is 42." in report.answer_clean


def test_verify_valid_citation_not_stripped():
    bundle = _bundle("FAISS is a similarity search library.")
    answer = "FAISS enables fast search. [D1]"
    report = GroundingVerifier().verify(answer, bundle)
    assert "[D1]" in report.answer_clean


def test_verify_multiple_citations_mixed():
    bundle = _bundle(
        "Dense retrieval uses embeddings.",
        "Sparse retrieval uses BM25 term matching.",
    )
    answer = (
        "Dense retrieval uses embeddings for search. [D1] "
        "Sparse retrieval relies on term frequency. [D2] "
        "Magic happens at night. [D99]"
    )
    report = GroundingVerifier().verify(answer, bundle)
    citations = {v.citation: v for v in report.verdicts}
    assert citations["D1"].document_found is True
    assert citations["D2"].document_found is True
    assert citations["D99"].document_found is False
    assert "[D99]" not in report.answer_clean
    assert "[D1]" in report.answer_clean
    assert "[D2]" in report.answer_clean


def test_grounding_rate_all_grounded():
    bundle = _bundle("FAISS is a similarity search library developed by Facebook AI.")
    answer = "FAISS is a similarity search library. [D1]"
    report = GroundingVerifier().verify(answer, bundle)
    assert report.grounding_rate == 1.0


def test_grounding_rate_none_grounded():
    bundle = _bundle("Some unrelated content here.")
    answer = "The sky is blue today. [D99]"
    report = GroundingVerifier().verify(answer, bundle)
    assert report.grounding_rate == 0.0


def test_grounding_rate_empty_answer():
    bundle = _bundle("content")
    report = GroundingVerifier().verify("", bundle)
    assert report.verdicts == []
    assert report.grounding_rate == 1.0
    assert report.answer_clean == ""


def test_dangling_citations_property():
    bundle = _bundle("content")
    answer = "Fact one [D1]. Fact two [D7]. Fact three [D8]."
    report = GroundingVerifier().verify(answer, bundle)
    assert set(report.dangling_citations) == {"D7", "D8"}


def test_ungrounded_citations_includes_dangling_and_low_overlap():
    bundle = _bundle("Completely unrelated document about cooking.")
    answer = "FAISS is used for vector search. [D1]"
    report = GroundingVerifier(overlap_threshold=0.5).verify(answer, bundle)
    assert "D1" in report.ungrounded_citations


# ---------------------------------------------------------------------------
# Integration: AnswerGenerationRequest.verify_grounding → generate_answer
# ---------------------------------------------------------------------------


def test_generate_answer_attaches_grounding_report_when_requested():
    from src.context.pipeline import generate_answer
    from src.context.models import AnswerGenerationRequest

    bundle = _bundle("FAISS is a vector similarity search library.")
    req = AnswerGenerationRequest(
        question="What is FAISS?",
        context=bundle,
        verify_grounding=True,
    )
    result = generate_answer(req, llm=None)
    assert result.grounding_report is not None
    assert isinstance(result.grounding_report, GroundingReport)


def test_generate_answer_no_grounding_report_by_default():
    from src.context.pipeline import generate_answer
    from src.context.models import AnswerGenerationRequest

    bundle = _bundle("FAISS content.")
    req = AnswerGenerationRequest(question="What is FAISS?", context=bundle)
    result = generate_answer(req, llm=None)
    assert result.grounding_report is None
