"""Runtime citation grounding verifier."""

from __future__ import annotations

import re

from .models import CitationVerdict, GroundingReport, SearchContextBundle

_CITATION_RE = re.compile(r"\[(D\d+)\]")
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can need dare ought used "
    "to of in on at for by with from about into through during before "
    "after above below between among and or but nor so yet both either "
    "neither not only also just more most some any such other each every "
    "both few more most other some such no nor not only own same so than "
    "too very i me my we our you your he she it its they them their "
    "what which who whom this that these those i s t don doesn won couldn "
    "how when where why".split()
)


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def _overlap(sentence_tokens: set[str], doc_tokens: set[str]) -> float:
    if not sentence_tokens or not doc_tokens:
        return 0.0
    return len(sentence_tokens & doc_tokens) / len(sentence_tokens)


def _split_sentences(text: str) -> list[str]:
    # Don't split immediately before a citation like [D1] — keep citation with its claim.
    raw = re.split(r"(?<=[.!?])\s+(?!\[D\d+\])", text.strip())
    return [s.strip() for s in raw if s.strip()]


class GroundingVerifier:
    """Verifies that each [Dx] citation in an answer is supported by the cited document.

    Uses stopword-filtered lexical overlap as a cheap entailment proxy — no NLI
    model required, runs in < 1 ms per answer.  Dangling citations (referencing
    documents not present in the context) are always flagged regardless of threshold.

    Args:
        overlap_threshold: Minimum fraction of sentence tokens that must appear in
            the cited document for the citation to be considered grounded.
            Default 0.15 is intentionally lenient to avoid false positives on
            paraphrase-style citations.
    """

    def __init__(self, *, overlap_threshold: float = 0.15) -> None:
        self.overlap_threshold = overlap_threshold

    def verify(self, answer: str, context: SearchContextBundle) -> GroundingReport:
        doc_map = {doc.id: doc for doc in context.documents}
        sentences = _split_sentences(answer)

        verdicts: list[CitationVerdict] = []
        for sentence in sentences:
            for citation in _CITATION_RE.findall(sentence):
                doc = doc_map.get(citation)
                if doc is None:
                    verdicts.append(
                        CitationVerdict(
                            citation=citation,
                            document_found=False,
                            overlap_score=0.0,
                            is_grounded=False,
                            sentence=sentence,
                        )
                    )
                    continue
                score = _overlap(_tokenize(sentence), _tokenize(doc.content))
                verdicts.append(
                    CitationVerdict(
                        citation=citation,
                        document_found=True,
                        overlap_score=score,
                        is_grounded=score >= self.overlap_threshold,
                        sentence=sentence,
                    )
                )

        dangling = {v.citation for v in verdicts if not v.document_found}
        answer_clean = answer
        for cit in sorted(dangling):
            answer_clean = re.sub(rf"\[{re.escape(cit)}\]", "", answer_clean)
        answer_clean = re.sub(r" {2,}", " ", answer_clean).strip()

        return GroundingReport(verdicts=verdicts, answer_clean=answer_clean)
