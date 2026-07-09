"""Simulated, deterministic pointwise judge standing in for an LLM-as-judge.

The judge scores an answer's quality/form from the answer text alone
(reference-free) and returns a scalar in ``[0, 1]``.  It is a drop-in for the
``BatchJudgeFn`` seam GRPO already consumes (``score_prompt_group`` /
``score_prompt_batch`` in :mod:`src.training.grpo`); a real LLM judge can
replace it behind the same :meth:`SimulatedPreferenceJudge.as_batch_judge_fn`
interface.

Scores are deterministic: identical answers always produce identical scores.
Tie-break jitter is derived from a SHA-256 digest, never the salted built-in
``hash`` (which varies per process) and never ``random`` — so tests and cached
runs are reproducible.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from src.training.reward import BatchJudgeFn

_HEDGES = (
    "i don't know",
    "i do not know",
    "cannot determine",
    "not sure",
    "unknown",
)


@dataclass(frozen=True)
class SimulatedPreferenceJudge:
    """Reference-free, deterministic pointwise answer-quality judge.

    ``max_words``: answers up to this many words get full length credit;
    longer answers are penalized.  ``jitter_scale``: magnitude of the
    deterministic tie-break term added to the base score.
    """

    max_words: int = 40
    jitter_scale: float = 0.05

    def score(self, answer: str) -> float:
        """Return a quality score in ``[0, 1]`` from the answer text alone."""
        text = answer.strip()
        if not text:
            return 0.0
        words = text.split()
        n = len(words)
        if n < 2:
            length_score = 0.3
        elif n <= self.max_words:
            length_score = 1.0
        else:
            length_score = max(0.2, 1.0 - (n - self.max_words) / 100.0)
        unique_ratio = len({w.lower() for w in words}) / n
        lowered = text.lower()
        hedge_penalty = 0.5 if any(h in lowered for h in _HEDGES) else 0.0
        base = 0.5 * length_score + 0.5 * unique_ratio - hedge_penalty
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        jitter = (digest[0] / 255.0) * self.jitter_scale
        return max(0.0, min(1.0, base + jitter))

    def as_batch_judge_fn(self) -> BatchJudgeFn:
        """Adapt to the ``BatchJudgeFn`` GRPO expects (ground truth ignored)."""

        def _judge(answers: list[str], ground_truths: list[str]) -> list[float]:
            return [self.score(a) for a in answers]

        return _judge


def judge_gold_agreement(pairs: list[tuple[float, bool]]) -> dict[str, float]:
    """Summarise how well judge scores separate correct from incorrect answers.

    ``pairs`` is a list of ``(judge_score, is_correct)``.  A positive ``gap``
    means the judge scores correct answers higher on average — evidence the
    (simulated) judge tracks correctness rather than being nonsense.  On hard
    factual questions a reference-free judge may show a small or zero gap; that
    is an informative result, not a failure.
    """
    correct = [s for s, ok in pairs if ok]
    incorrect = [s for s, ok in pairs if not ok]
    mean_correct = sum(correct) / len(correct) if correct else 0.0
    mean_incorrect = sum(incorrect) / len(incorrect) if incorrect else 0.0
    return {
        "mean_score_correct": mean_correct,
        "mean_score_incorrect": mean_incorrect,
        "gap": mean_correct - mean_incorrect,
        "n_correct": float(len(correct)),
        "n_incorrect": float(len(incorrect)),
    }
