"""High-level retrieval, prompt, and answer-generation pipeline."""

from __future__ import annotations

from .models import AnswerGenerationRequest
from .models import AnswerGenerationResult
from .models import ChatMessage
from .models import LLMClient
from .models import LLMResponse
from .models import SearchContextBundle
from .models import SearchFilters
from .models import SearchRequest
from .prompts import build_chat_prompt
from .retrieval.search_runner import build_search_context
from .utils import extract_citations


async def retrieve_context(
    question: str,
    *,
    search_url: str = "http://localhost:8000/retrieve",
    top_k: int = 5,
    filters: SearchFilters | None = None,
) -> SearchContextBundle:
    return await build_search_context(
        SearchRequest(query=question, top_k=top_k, filters=filters),
        search_url=search_url,
    )


def generate_answer(
    request: AnswerGenerationRequest,
    *,
    llm: LLMClient | None = None,
) -> AnswerGenerationResult:
    prompt = build_chat_prompt(
        request.question,
        request.context,
        history=request.chat_history,
        config=request.behavior,
    )
    if llm is None:
        answer = synthesize_answer_from_context(request.question, request.context)
    else:
        raw = llm.complete(prompt.messages)
        answer = raw.text if isinstance(raw, LLMResponse) else str(raw)
    return AnswerGenerationResult(
        answer=answer,
        citations=extract_citations(answer),
        context=request.context,
        prompt=prompt,
    )


async def answer_with_retrieval(
    question: str,
    *,
    llm: LLMClient | None = None,
    chat_history: list[ChatMessage] | None = None,
    search_url: str = "http://localhost:8000/retrieve",
    top_k: int = 5,
    filters: SearchFilters | None = None,
) -> AnswerGenerationResult:
    context = await retrieve_context(
        question,
        search_url=search_url,
        top_k=top_k,
        filters=filters,
    )
    return generate_answer(
        AnswerGenerationRequest(
            question=question,
            context=context,
            chat_history=chat_history or [],
        ),
        llm=llm,
    )


def synthesize_answer_from_context(question: str, context: SearchContextBundle) -> str:
    """Extractive fallback answer when no LLM is available.

    Scores every sentence in the retrieved documents by keyword overlap with
    the question, then assembles the top sentences into a grounded answer with
    inline citations.  This is intentionally conservative — it never fabricates
    information not present in the retrieved context.
    """
    if not context.documents:
        return f"I could not find retrieved context to answer: {question}"

    question_tokens = _tokenize(question)
    if not question_tokens:
        # Degenerate query — fall back to the top document's lead sentence.
        doc = context.documents[0]
        lead = _first_sentence(doc.content)
        return f"{lead} {doc.citation}"

    # Score every sentence across all documents.
    scored: list[tuple[float, str, str]] = []  # (score, citation, sentence)
    for doc in context.documents:
        for sentence in _split_sentences(doc.content):
            score = _overlap_score(question_tokens, _tokenize(sentence))
            if score > 0:
                scored.append((score, doc.citation, sentence))

    if not scored:
        # No sentence matched any keyword — return the lead of the top doc.
        doc = context.documents[0]
        return f"Based on {doc.citation} ({doc.title}): {_first_sentence(doc.content)}"

    scored.sort(key=lambda t: t[0], reverse=True)

    # Take up to 3 best sentences, deduplicating by citation.
    seen_citations: set[str] = set()
    selected: list[tuple[str, str]] = []
    for _, citation, sentence in scored:
        if citation not in seen_citations:
            selected.append((citation, sentence))
            seen_citations.add(citation)
        if len(selected) >= 3:
            break

    parts = [f"{sentence} {citation}" for citation, sentence in selected]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Extractive helpers
# ---------------------------------------------------------------------------

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
    import re

    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def _overlap_score(query_tokens: set[str], sentence_tokens: set[str]) -> float:
    if not query_tokens or not sentence_tokens:
        return 0.0
    shared = query_tokens & sentence_tokens
    # Jaccard-style but biased toward query recall
    return len(shared) / len(query_tokens)


def _split_sentences(text: str) -> list[str]:
    import re

    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 20]


def _first_sentence(text: str) -> str:
    sentences = _split_sentences(text)
    return sentences[0] if sentences else text[:200].strip()
