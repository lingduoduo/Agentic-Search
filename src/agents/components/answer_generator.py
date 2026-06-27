"""AnswerGenerator: make the final answer + citation step explicit.

The policy LM emits an answer containing inline citation markers of the form
``[R{round}Q{query}D{doc}]``. This component links those markers back to the
retrieved results (reusing :class:`AgentContext`'s citation accounting) and
produces structured :class:`Citation` objects, written into the agent state.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...context.search import AgentContext
from ..state import AgentState, Citation


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    citations: list[Citation]


class AnswerGenerator:
    """Resolve inline citation markers in an answer into structured citations."""

    def generate(self, answer_text: str, ctx: AgentContext) -> AnswerResult:
        valid_keys = ctx.cited_result_ids(answer_text)
        # Collect every valid (key, contents) the answer references.
        candidates: list[tuple[str, str]] = []
        for round_idx, round_ctxs in enumerate(ctx.rounds, 1):
            for q_idx, search_ctx in enumerate(round_ctxs, 1):
                for d_idx, result in enumerate(search_ctx.results, 1):
                    key = f"R{round_idx}Q{q_idx}D{d_idx}"
                    if key in valid_keys:
                        candidates.append((key, result.contents))

        # Order by first appearance of the marker in the answer text, then drop
        # later citations whose contents already appeared (collapse duplicates).
        candidates.sort(key=lambda kc: answer_text.find(f"[{kc[0]}]"))
        seen_contents: set[str] = set()
        citations: list[Citation] = []
        for key, contents in candidates:
            if contents in seen_contents:
                continue
            seen_contents.add(contents)
            citations.append(Citation(doc_id=key, marker=f"[{key}]", text=contents))
        return AnswerResult(answer=answer_text, citations=citations)

    def update_state(
        self, state: AgentState, answer_text: str, ctx: AgentContext
    ) -> AnswerResult:
        result = self.generate(answer_text, ctx)
        state.set_citations(result.citations)
        return result
