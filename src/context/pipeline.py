"""High-level retrieval, prompt, and answer-generation pipeline."""

from __future__ import annotations

from .models import AnswerGenerationRequest
from .models import AnswerGenerationResult
from .models import ChatMessage
from .models import LLMClient
from .models import LLMResponse
from .models import SearchContextBundle
from .models import SearchRequest
from .prompts import build_chat_prompt
from .retrieval.search_runner import build_search_context
from .utils import extract_citations


async def retrieve_context(
    question: str,
    *,
    search_url: str = "http://localhost:8000/retrieve",
    top_k: int = 5,
) -> SearchContextBundle:
    return await build_search_context(
        SearchRequest(query=question, top_k=top_k),
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
) -> AnswerGenerationResult:
    context = await retrieve_context(question, search_url=search_url, top_k=top_k)
    return generate_answer(
        AnswerGenerationRequest(
            question=question,
            context=context,
            chat_history=chat_history or [],
        ),
        llm=llm,
    )


def synthesize_answer_from_context(question: str, context: SearchContextBundle) -> str:
    if not context.documents:
        return f"I do not have retrieved context to answer: {question}"
    snippets = []
    for document in context.documents[:3]:
        content = " ".join(document.content.split())
        snippets.append(f"{document.citation} {document.title}\n{content}")
    return "\n\n".join(snippets)
