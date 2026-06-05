"""Agentic RAG loop: iterative hybrid retrieval + LLM-driven sufficiency assessment."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.context.models import (
    AnswerGenerationRequest,
    ChatMessage,
    ContextDocument,
    LLMClient,
    SearchContextBundle,
)
from src.context.pipeline import generate_answer, retrieve_context
from src.context.query_enhancer import QueryEnhancer

logger = logging.getLogger(__name__)

_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+[.)]\s*|[-*•]\s*)")
_ARTIFACT_RE = re.compile(r'[\[\]"\'`]')

_SUFFICIENCY_PROMPT = """You are evaluating whether retrieved documents are sufficient to fully answer a question.
Respond with exactly "yes" or "no".

Question: {question}

Retrieved context (first 1500 chars):
{context}""".strip()

_GAP_ANALYSIS_PROMPT = """You are analyzing whether retrieved documents fully answer a question.

Question: {question}

Retrieved context (first 1000 chars):
{context}

Step 1 — List the specific pieces of information the question requires but the context does NOT provide.
         Write each gap as a short phrase (e.g. "training cost of GPT-4").
         If nothing is missing, write "none".

Step 2 — For each gap, write one focused search query that would retrieve the missing information.
         Format: one query per line, no numbering, no extra text.
         Queries only — do not repeat the gap phrases.

Output format:
GAPS:
<gap 1>
<gap 2>

QUERIES:
<query 1>
<query 2>""".strip()


def _llm_text(response: object) -> str:
    if hasattr(response, "text"):
        return response.text
    if hasattr(response, "content"):
        return response.content
    return str(response)


def _clean_line(line: str) -> str:
    return _ARTIFACT_RE.sub("", _LIST_MARKER_RE.sub("", line)).strip()


def _parse_gap_queries(raw: str) -> list[str]:
    """Extract the QUERIES section from a structured gap-analysis response.

    Falls back to treating every non-empty line as a query when the
    structured format is absent (e.g. legacy LLM response).
    """
    if "QUERIES:" in raw:
        queries_section = raw.split("QUERIES:", 1)[1]
    elif "GAPS:" in raw:
        return []
    else:
        queries_section = raw
    return [
        _clean_line(line) for line in queries_section.splitlines() if _clean_line(line)
    ]


@dataclass(frozen=True)
class AgenticRAGConfig:
    max_rounds: int = 3
    topk: int = 5
    retrieval_url: str = "http://localhost:8000/retrieve"


@dataclass
class AgenticRAGResult:
    answer: str
    citations: list[str]
    rounds_used: int
    context: SearchContextBundle


class AgenticRAGLoop:
    """Iterative RAG loop with query enhancement and evidence sufficiency gating.

    Flow per run():
      1. Enhance query (decompose + HyDE via QueryEnhancer)
      2. Retrieve for every current query; accumulate unique docs by id
      3. Ask LLM: is evidence sufficient? → if yes, break; if no, generate follow-ups
      4. Repeat up to max_rounds
      5. Synthesize grounded answer from all accumulated evidence
    """

    def __init__(self, config: AgenticRAGConfig, llm: LLMClient | None = None) -> None:
        self.config = config
        self.llm = llm
        self._enhancer = QueryEnhancer(llm)

    async def run(
        self,
        question: str,
        *,
        chat_history: list[ChatMessage] | None = None,
    ) -> AgenticRAGResult:
        bundle = self._enhancer.enhance(question)
        current_queries = bundle.all_queries()

        accumulated: dict[str, ContextDocument] = {}
        seen_queries: set[str] = set()
        rounds_used = 0

        for round_idx in range(self.config.max_rounds):
            rounds_used += 1
            novel_queries = [q for q in current_queries if q not in seen_queries]
            if not novel_queries:
                break
            seen_queries.update(novel_queries)

            for q in novel_queries:
                try:
                    ctx = await retrieve_context(
                        q,
                        search_url=self.config.retrieval_url,
                        top_k=self.config.topk,
                    )
                    for doc in ctx.documents:
                        if doc.id not in accumulated:
                            accumulated[doc.id] = doc
                except Exception as exc:
                    logger.warning("Retrieval failed for query %r: %s", q, exc)

            merged = SearchContextBundle(
                query=question, documents=list(accumulated.values())
            )

            # On the last round always proceed to synthesis; otherwise check sufficiency.
            is_last = round_idx == self.config.max_rounds - 1
            if not is_last:
                if self._is_sufficient(question, merged):
                    break
                follow_ups = self._generate_followup(question, merged)
                novel_follow_ups = [q for q in follow_ups if q not in seen_queries]
                if not novel_follow_ups:
                    break
                current_queries = novel_follow_ups

        gen_result = generate_answer(
            AnswerGenerationRequest(
                question=question,
                context=merged,
                chat_history=chat_history or [],
            ),
            llm=self.llm,
        )
        return AgenticRAGResult(
            answer=gen_result.answer,
            citations=gen_result.citations,
            rounds_used=rounds_used,
            context=merged,
        )

    def _is_sufficient(self, question: str, context: SearchContextBundle) -> bool:
        if not context.documents:
            return False
        if self.llm is None:
            return True
        prompt = _SUFFICIENCY_PROMPT.format(
            question=question,
            context=context.to_context_text()[:1500],
        )
        try:
            raw = (
                _llm_text(self.llm.complete([ChatMessage(role="user", content=prompt)]))
                .strip()
                .lower()
            )
            return raw.startswith("yes")
        except Exception as exc:
            logger.warning("Sufficiency check failed: %s", exc)
            return True  # assume sufficient → stop looping on LLM error

    def _generate_followup(
        self, question: str, context: SearchContextBundle
    ) -> list[str]:
        if self.llm is None:
            return []
        prompt = _GAP_ANALYSIS_PROMPT.format(
            question=question,
            context=context.to_context_text()[:1000],
        )
        try:
            raw = _llm_text(
                self.llm.complete([ChatMessage(role="user", content=prompt)])
            ).strip()
            return _parse_gap_queries(raw)
        except Exception as exc:
            logger.warning("Gap analysis failed: %s", exc)
            return []
