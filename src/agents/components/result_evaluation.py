"""Heuristics for evaluating whether search results are good enough to stop."""

from __future__ import annotations

from dataclasses import dataclass

from ...context.search import SearchContext


@dataclass(frozen=True)
class SearchEvaluationConfig:
    """Thresholds used to judge whether a search round returned enough signal."""

    min_results_per_query: int = 1
    min_total_results: int = 2
    min_top_score: float = 0.0
    min_avg_score: float = 0.0
    # Require at least this many characters of content per result (strips whitespace first).
    min_content_length: int = 10
    require_scores: bool = False


@dataclass(frozen=True)
class QueryEvaluation:
    """Per-query assessment of returned results."""

    query: str
    result_count: int
    nonempty_count: int
    top_score: float | None
    avg_score: float | None
    is_sufficient: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchRoundEvaluation:
    """Assessment of a full search round, possibly containing parallel queries."""

    queries: tuple[QueryEvaluation, ...]
    total_results: int
    is_sufficient: bool
    reasons: tuple[str, ...] = ()

    @property
    def insufficient_queries(self) -> list[QueryEvaluation]:
        return [q for q in self.queries if not q.is_sufficient]

    def to_feedback_block(self) -> str:
        """Render evaluator feedback for reinjection into the agent loop."""
        verdict = "SUFFICIENT" if self.is_sufficient else "INSUFFICIENT"
        lines = [
            f"Verdict: {verdict}",
            f"Total results: {self.total_results}",
        ]
        if self.reasons:
            lines.append("Round notes: " + "; ".join(self.reasons))

        for index, qe in enumerate(self.queries, 1):
            status = "OK" if qe.is_sufficient else "NEEDS_MORE_SEARCH"
            lines.append(
                f"Query {index}: {status} | query={qe.query!r} | "
                f"results={qe.result_count} (non-empty: {qe.nonempty_count}) | "
                f"top_score={qe.top_score} | avg_score={qe.avg_score}"
            )
            if qe.reasons:
                lines.append("  Notes: " + "; ".join(qe.reasons))

        if self.is_sufficient:
            lines.append(
                "Action: evidence is likely good enough to synthesize if it answers the question."
            )
        else:
            lines.append(
                "Action: refine keywords and keep searching before finalizing an answer."
            )
        return "\n".join(lines)


class SearchResultEvaluator:
    """Evaluate whether returned search results are good enough to stop searching."""

    def __init__(self, config: SearchEvaluationConfig | None = None) -> None:
        self.config = config or SearchEvaluationConfig()

    def evaluate_round(
        self, search_contexts: list[SearchContext]
    ) -> SearchRoundEvaluation:
        query_evals = tuple(self._evaluate_query(ctx) for ctx in search_contexts)
        total_results = sum(qe.result_count for qe in query_evals)

        reasons: list[str] = []
        if total_results < self.config.min_total_results:
            reasons.append(
                f"total results {total_results} below minimum {self.config.min_total_results}"
            )
        if any(not qe.is_sufficient for qe in query_evals):
            reasons.append("at least one query did not return enough strong evidence")

        is_sufficient = bool(query_evals) and not reasons
        return SearchRoundEvaluation(
            queries=query_evals,
            total_results=total_results,
            is_sufficient=is_sufficient,
            reasons=tuple(reasons),
        )

    def _evaluate_query(self, ctx: SearchContext) -> QueryEvaluation:
        results = ctx.results
        result_count = len(results)
        nonempty_count = sum(
            1
            for r in results
            if len(r.contents.strip()) >= self.config.min_content_length
        )
        scored = [r.score for r in results if r.score > 0.0]
        top_score = max(scored) if scored else None
        avg_score = sum(scored) / len(scored) if scored else None

        reasons: list[str] = []
        if result_count < self.config.min_results_per_query:
            reasons.append(
                f"result count {result_count} below minimum {self.config.min_results_per_query}"
            )
        if nonempty_count < result_count:
            reasons.append(
                f"{result_count - nonempty_count} result(s) had insufficient content "
                f"(min {self.config.min_content_length} chars)"
            )
        if self.config.require_scores and not scored:
            reasons.append("no retrieval scores returned")
        if top_score is not None and top_score < self.config.min_top_score:
            reasons.append(
                f"top score {top_score:.4f} below minimum {self.config.min_top_score:.4f}"
            )
        if avg_score is not None and avg_score < self.config.min_avg_score:
            reasons.append(
                f"average score {avg_score:.4f} below minimum {self.config.min_avg_score:.4f}"
            )

        return QueryEvaluation(
            query=ctx.query,
            result_count=result_count,
            nonempty_count=nonempty_count,
            top_score=top_score,
            avg_score=avg_score,
            is_sufficient=not reasons,
            reasons=tuple(reasons),
        )
