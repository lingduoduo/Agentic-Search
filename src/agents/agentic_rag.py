"""Agentic RAG loop: iterative hybrid retrieval + LLM-driven sufficiency assessment."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.control_flow_trace import ControlFlowRecorder

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


def _norm_query(q: str) -> str:
    return " ".join(q.lower().split())


def _dedupe_novel(queries: list[str], seen: set[str]) -> list[str]:
    """Return queries whose normalized form is new; record each into `seen`.

    Dedupes both within this batch and against earlier rounds. Returned
    strings are the original queries — retrieval must use the raw text.
    """
    novel: list[str] = []
    for q in queries:
        norm = _norm_query(q)
        if norm not in seen:
            seen.add(norm)
            novel.append(q)
    return novel


def _doc_key(doc: ContextDocument) -> str:
    if doc.url:
        return doc.url.strip().lower()
    return hashlib.sha256(doc.content.encode("utf-8")).hexdigest()


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
    queries: list[str] = []
    for line in queries_section.splitlines():
        cleaned = _clean_line(line)
        if cleaned:
            queries.append(cleaned)
    return queries


@dataclass(frozen=True)
class AgenticRAGConfig:
    max_rounds: int = 3
    topk: int = 5
    retrieval_url: str = "http://localhost:8001/retrieve"
    max_followups_per_round: int = 5
    sufficiency_timeout_s: float = 5.0


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
        recorder: "ControlFlowRecorder | None" = None,
    ) -> AgenticRAGResult:
        def _emit(
            component: str,
            action: str,
            status: str,
            duration_ms: int | None = None,
            **details: object,
        ) -> None:
            if recorder is not None:
                recorder.record(
                    turn=max(rounds_used, 1),
                    component=component,
                    action=action,
                    status=status,
                    duration_ms=duration_ms,
                    details=details,
                )

        accumulated: dict[str, ContextDocument] = {}
        seen_queries: set[str] = set()
        rounds_used = 0

        t0 = time.perf_counter()
        bundle = self._enhancer.enhance(question)
        current_queries = bundle.all_queries()
        # Key by content fingerprint so docs from different retrieve_context() calls
        # (which all produce ephemeral D1-D5 IDs) are deduplicated correctly.
        _emit(
            "query_enhancer",
            "enhance",
            "completed",
            duration_ms=round((time.perf_counter() - t0) * 1000),
            query_count=len(current_queries),
        )

        merged = SearchContextBundle(query=question, documents=[])

        for round_idx in range(self.config.max_rounds):
            rounds_used += 1
            # current_queries is already deduped+recorded into seen_queries when
            # it originates from the follow-up branch below; only the initial
            # enhance() batch (round_idx == 0) still needs deduping here.
            if round_idx == 0:
                novel_queries = _dedupe_novel(current_queries, seen_queries)
            else:
                novel_queries = current_queries
            if not novel_queries:
                break

            t_retr = time.perf_counter()
            for q in novel_queries:
                try:
                    ctx = await retrieve_context(
                        q,
                        search_url=self.config.retrieval_url,
                        top_k=self.config.topk,
                    )
                    for doc in ctx.documents:
                        key = _doc_key(doc)
                        if key not in accumulated:
                            accumulated[key] = doc
                except Exception as exc:
                    logger.warning("Retrieval failed for query %r: %s", q, exc)
            retr_ms = round((time.perf_counter() - t_retr) * 1000)

            # Re-assign stable D1..DN IDs so citations are consistent across rounds.
            stable_docs = [
                ContextDocument(
                    id=f"D{i}",
                    title=doc.title,
                    content=doc.content,
                    url=doc.url,
                    score=doc.score,
                    metadata=doc.metadata,
                )
                for i, doc in enumerate(accumulated.values(), 1)
            ]
            merged = SearchContextBundle(query=question, documents=stable_docs)
            _emit(
                "search_tool",
                "vector_db_search",
                "completed",
                duration_ms=retr_ms,
                query_count=len(novel_queries),
                document_count=len(stable_docs),
                search_round=rounds_used,
            )

            # On the last round always proceed to synthesis; otherwise check sufficiency.
            is_last = round_idx == self.config.max_rounds - 1
            if not is_last:
                t_suff = time.perf_counter()
                sufficient = await self._is_sufficient(question, merged)
                _emit(
                    "evidence_judge",
                    "sufficiency_check",
                    "decided",
                    duration_ms=round((time.perf_counter() - t_suff) * 1000),
                    sufficient=sufficient,
                    search_round=rounds_used,
                )
                if sufficient:
                    break
                follow_ups = self._generate_followup(question, merged)
                novel_follow_ups = _dedupe_novel(follow_ups, seen_queries)
                if not novel_follow_ups:
                    break
                current_queries = novel_follow_ups[
                    : self.config.max_followups_per_round
                ]

        t_syn = time.perf_counter()
        gen_result = generate_answer(
            AnswerGenerationRequest(
                question=question,
                context=merged,
                chat_history=chat_history or [],
            ),
            llm=self.llm,
        )
        _emit(
            "answer_generator",
            "synthesize",
            "completed",
            duration_ms=round((time.perf_counter() - t_syn) * 1000),
            citation_count=len(gen_result.citations),
            document_count=len(merged.documents),
        )
        return AgenticRAGResult(
            answer=gen_result.answer,
            citations=gen_result.citations,
            rounds_used=rounds_used,
            context=merged,
        )

    async def _is_sufficient(self, question: str, context: SearchContextBundle) -> bool:
        if not context.documents:
            return False
        if self.llm is None:
            return True
        prompt = _SUFFICIENCY_PROMPT.format(
            question=question,
            context=context.to_context_text()[:1500],
        )
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.llm.complete,
                    [ChatMessage(role="user", content=prompt)],
                ),
                timeout=self.config.sufficiency_timeout_s,
            )
            return _llm_text(response).strip().lower().startswith("yes")
        except Exception as exc:  # includes asyncio.TimeoutError
            logger.warning("Sufficiency check failed or timed out: %s", exc)
            return True  # fail-open → stop looping on error/timeout

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
