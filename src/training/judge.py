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
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

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


# ---------------------------------------------------------------------------
# Reference-based judging.
#
# Everything above scores an answer from its own text. Everything below
# compares it against the gold answer, which is what a training signal for
# correctness actually requires.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldAgreementJudge:
    """Deterministic reference-based judge: does the answer match the gold?

    Graded rather than binary, because GRPO normalises advantages *within* a
    rollout group: if every rollout for a prompt scores 0.0, every advantage is
    0.0 and that prompt contributes no gradient at all. A partial-credit signal
    keeps near-misses informative instead of silently dropping the prompt.

    The ladder, highest first:

    - exact match after normalisation -> 1.0
    - gold contained in the prediction -> ``containment_score``
    - otherwise token F1, scaled by ``partial_weight``

    No network, no model, no randomness. This is the offline fallback for
    :class:`LLMJudge` and a usable judge in its own right.
    """

    containment_score: float = 0.7
    partial_weight: float = 0.6

    def score(self, answer: str, gold: str) -> float:
        from src.training.reward import normalize_answer_text, token_f1_score

        pred_norm = normalize_answer_text(answer)
        gold_norm = normalize_answer_text(gold)
        if not gold_norm:
            return 0.0
        if pred_norm == gold_norm:
            return 1.0
        if gold_norm in pred_norm:
            return self.containment_score
        return self.partial_weight * token_f1_score(answer, gold)

    def as_batch_judge_fn(self) -> BatchJudgeFn:
        def _judge(answers: list[str], ground_truths: list[str]) -> list[float]:
            return [self.score(a, g) for a, g in zip(answers, ground_truths)]

        return _judge


class JudgeParseError(RuntimeError):
    """An LLM judge response could not be read as a score."""


def parse_judge_score(raw: str) -> float:
    """Read a score in [0, 1] from a judge response, or raise.

    Raises rather than returning a default **on purpose**. A judge that quietly
    returns a middling score for every unparseable reply gives every rollout in
    a group the same value, every within-group advantage becomes 0.0, and
    training turns into a no-op that still logs as if it were working. A loud
    failure that the caller decides how to handle is the only safe contract
    here — see :class:`LLMJudge`, which falls back per item and counts it.
    """
    text = raw.strip()
    if not text:
        raise JudgeParseError("empty judge response")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match is None:
        raise JudgeParseError(f"no number in judge response: {text[:80]!r}")
    value = float(match.group(0))
    if not 0.0 <= value <= 1.0:
        raise JudgeParseError(f"judge score out of range [0, 1]: {value}")
    return value


def is_degenerate_group(scores: Sequence[float], *, tolerance: float = 1e-9) -> bool:
    """True when every score in a rollout group is effectively identical.

    GRPO advantages are ``score_i - mean(group)``, so a group whose scores are
    all equal produces all-zero advantages and contributes nothing to the
    update. That is indistinguishable from a working step in the logs, which is
    why it is worth naming and asserting on rather than discovering later as
    "training ran but the model did not move".
    """
    values = list(scores)
    if len(values) < 2:
        return True
    return (max(values) - min(values)) <= tolerance


@dataclass
class LLMJudge:
    """LLM-as-judge over (answer, gold), with a deterministic offline fallback.

    Three properties this has to get right, none of them about prompt wording:

    **It must see the gold.** The reference-free judge it replaces scored answer
    *shape* — length, vocabulary variety, absence of hedging — so a confidently
    worded wrong answer outscored a correct hedged one by construction. Any
    judge used as a training signal for correctness has to read the reference.

    **It must fail per item, not per batch.** An unparseable reply falls back to
    :class:`GoldAgreementJudge` for that answer only, and increments
    ``parse_failures``. Substituting a constant would flatten the group and zero
    every advantage; aborting the batch would throw away the rollouts that did
    parse.

    **It must be cheap on repeats.** GRPO scores G rollouts per prompt against
    one gold, and prompts recur across steps, so results are cached by
    ``(answer, gold)``.

    With ``llm=None`` this is exactly ``GoldAgreementJudge`` plus bookkeeping,
    which is what keeps the no-network smoke path working.
    """

    llm: object | None = None
    fallback: GoldAgreementJudge = field(default_factory=GoldAgreementJudge)
    parse_failures: int = 0
    llm_calls: int = 0
    _cache: dict[tuple[str, str], float] = field(default_factory=dict, repr=False)

    def score(self, answer: str, gold: str) -> float:
        key = (answer, gold)
        if key in self._cache:
            return self._cache[key]
        value = self._score_uncached(answer, gold)
        self._cache[key] = value
        return value

    def _score_uncached(self, answer: str, gold: str) -> float:
        if self.llm is None:
            return self.fallback.score(answer, gold)
        try:
            self.llm_calls += 1
            raw = self.llm.complete(
                [
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _JUDGE_USER_TEMPLATE.format(
                            gold=gold.strip(), answer=answer.strip()
                        ),
                    },
                ]
            )
            text = raw if isinstance(raw, str) else getattr(raw, "text", "")
            return parse_judge_score(text)
        except Exception:
            # Includes JudgeParseError and any provider/network error. Falling
            # back keeps the group's scores spread out and the step meaningful;
            # the counter is what makes a silently-degraded run visible.
            self.parse_failures += 1
            return self.fallback.score(answer, gold)

    def as_batch_judge_fn(self) -> BatchJudgeFn:
        def _judge(answers: list[str], ground_truths: list[str]) -> list[float]:
            return [self.score(a, g) for a, g in zip(answers, ground_truths)]

        return _judge


_JUDGE_SYSTEM_PROMPT = (
    "You grade short factual answers against a reference answer. "
    "Reply with a single number between 0 and 1 and nothing else. "
    "1 means the answer conveys the same fact as the reference, including "
    "paraphrases and differences of formatting. 0 means it does not. "
    "Judge only factual agreement — never length, fluency, confidence, or style."
)

_JUDGE_USER_TEMPLATE = (
    "Reference answer: {gold}\n\nCandidate answer: {answer}\n\nScore:"
)
