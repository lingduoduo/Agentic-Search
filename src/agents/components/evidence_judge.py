"""EvidenceJudge: turn the heuristic sufficiency check into a continuous score.

Wraps the existing :class:`SearchResultEvaluator` (a boolean, threshold-based
gate) and maps its verdict to a continuous ``evidence_score`` in ``[0, 1]`` that
the GRPO policy can condition on. The boolean ``is_sufficient`` is preserved as
the safety rail that gates premature answers.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...context.search import SearchContext
from ...training.evaluation import (
    SearchEvaluationConfig,
    SearchResultEvaluator,
    SearchRoundEvaluation,
)
from ..core.state import AgentState


@dataclass(frozen=True)
class EvidenceVerdict:
    """Continuous + boolean view of how good a search round's evidence is."""

    is_sufficient: bool
    score: float
    round_eval: SearchRoundEvaluation


def _squash(score: float | None) -> float:
    """Map a non-negative retrieval score into [0, 1) via x / (1 + x)."""
    if not score or score <= 0.0:
        return 0.0
    return score / (1.0 + score)


class EvidenceJudge:
    """Score the sufficiency of retrieved evidence for the current question."""

    def __init__(
        self,
        config: SearchEvaluationConfig | None = None,
        evaluator: SearchResultEvaluator | None = None,
    ) -> None:
        self._evaluator = evaluator or SearchResultEvaluator(config)

    def judge(self, contexts: list[SearchContext]) -> EvidenceVerdict:
        round_eval = self._evaluator.evaluate_round(contexts)
        return EvidenceVerdict(
            is_sufficient=round_eval.is_sufficient,
            score=self._to_score(round_eval),
            round_eval=round_eval,
        )

    def update_state(
        self, state: AgentState, contexts: list[SearchContext]
    ) -> EvidenceVerdict:
        verdict = self.judge(contexts)
        state.set_evidence(verdict.score)
        return verdict

    @staticmethod
    def marginal_gain(prev: float, curr: float) -> float:
        """Improvement in evidence_score from one round to the next."""
        return curr - prev

    @staticmethod
    def should_stop(prev: float, curr: float, min_gain: float) -> bool:
        """True when the latest round's gain falls below ``min_gain`` (plateau).

        A plateau means another search round is unlikely to help, so the policy
        (or an opt-in loop) can choose to stop searching. This never forces an
        answer; it only signals that more searching has diminishing returns.
        """
        return EvidenceJudge.marginal_gain(prev, curr) < min_gain

    @staticmethod
    def score_round(round_eval: SearchRoundEvaluation) -> float:
        """Public continuous score for an already-computed round evaluation.

        Lets the agent loop reuse the heuristic→[0,1] mapping without re-running
        the evaluator it already called.
        """
        return EvidenceJudge._to_score(round_eval)

    @staticmethod
    def _to_score(round_eval: SearchRoundEvaluation) -> float:
        """Blend the fraction of sufficient queries with squashed top scores.

        Both terms are in [0, 1], so the blend is too. The score term makes the
        result monotonic in retrieval quality even when every query already
        clears the boolean sufficiency thresholds.
        """
        queries = round_eval.queries
        if not queries:
            return 0.0
        frac_sufficient = sum(q.is_sufficient for q in queries) / len(queries)
        tops = [_squash(q.top_score) for q in queries if q.top_score is not None]
        score_signal = sum(tops) / len(tops) if tops else 0.0
        return max(0.0, min(1.0, 0.5 * frac_sufficient + 0.5 * score_signal))
