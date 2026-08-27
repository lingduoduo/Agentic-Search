"""GRPO losses, grouped rollout sampling and scoring, and judges."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from typing import Any, Callable, overload
from uuid import uuid4

import torch

from ....agents.core.base import AgentLoopBase, AgentLoopOutput
from ..data import PromptBatch
from ..ppo.core_algos import (
    PPOPolicyLossConfig,
    compute_trajectory_policy_loss,
    masked_mean,
    masked_whiten,
)
from ..reward import BatchJudgeFn, JudgeFn, SearchRewardFunction, _score_answers


def _compute_grpo_outcome_advantage_tensor(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    clip_advantages: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group-normalized outcome advantages expanded over response tokens.

    Args:
        token_level_rewards: ``(batch, seq_len)`` sparse reward tensor.
        eos_mask: ``(batch, seq_len)`` binary mask over response tokens.
        index: ``(batch,)`` integer group IDs.
        epsilon: Denominator stabiliser for std normalization.
        clip_advantages: If given, clip normalized advantages to
            ``[-clip_advantages, +clip_advantages]`` before expansion.
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    advantages = torch.zeros_like(token_level_rewards)

    with torch.no_grad():
        for group_id in torch.unique(index).tolist():
            group_mask = index == int(group_id)
            group_scores = scores[group_mask]
            if group_scores.numel() == 0:
                continue
            mean = group_scores.mean()
            std = (
                group_scores.std(unbiased=False)
                if group_scores.numel() > 1
                else torch.tensor(1.0, device=scores.device, dtype=scores.dtype)
            )
            normalized = (group_scores - mean) / (std + epsilon)
            if clip_advantages is not None:
                normalized = normalized.clamp(
                    -float(clip_advantages), float(clip_advantages)
                )
            advantages[group_mask] = (
                normalized.unsqueeze(-1).expand(-1, response_length)
                * eos_mask[group_mask]
            )

    return advantages, advantages


def compute_reinforce_policy_loss_core(
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
) -> torch.Tensor:
    """Unclipped REINFORCE policy-gradient loss over masked response tokens."""

    return -masked_mean(log_prob * advantages, eos_mask)


def compute_reinforce_policy_loss(
    *,
    log_probs: list[float],
    rewards: list[float],
    response_mask: list[int],
    baseline: float = 0.0,
) -> dict[str, float]:
    """Compute a trajectory-level REINFORCE loss from aligned token lists."""

    n = len(log_probs)
    if not (len(rewards) == len(response_mask) == n):
        raise ValueError(
            "log_probs, rewards, and response_mask must all have the same length, "
            f"got lengths {n}, {len(rewards)}, {len(response_mask)}."
        )

    log_prob_tensor = torch.tensor(log_probs, dtype=torch.float32)
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    mask = torch.tensor(response_mask, dtype=torch.float32)
    advantages = reward_tensor - float(baseline)
    normalizer = max(float(mask.sum()), 1.0)
    loss = float(-((log_prob_tensor * advantages * mask).sum() / normalizer))
    return {
        "reinforce_policy_loss": loss,
        "total_loss": loss,
        "mean_reward": float((reward_tensor * mask).sum() / normalizer),
        "mean_advantage": float((advantages * mask).sum() / normalizer),
    }


def compute_grpo_policy_loss(
    *,
    new_log_probs: list[float],
    old_log_probs: list[float],
    advantages: list[float],
    response_mask: list[int],
    ref_log_probs: list[float] | None = None,
    config: PPOPolicyLossConfig | None = None,
) -> dict[str, float]:
    """Convenience wrapper that accepts a :class:`PPOPolicyLossConfig`.

    Combines the clipped policy loss, optional KL penalty, optional entropy
    bonus, and optional advantage whitening in a single call.  Delegates to
    :func:`compute_trajectory_policy_loss` for the core arithmetic.

    Args:
        new_log_probs: Current policy log-probs for response tokens.
        old_log_probs: Behaviour-policy log-probs (from rollout time).
        advantages: Per-token advantage values (aligned with response_mask).
        response_mask: 1 for model-generated tokens, 0 for prompt / padding.
        ref_log_probs: Reference model log-probs for KL penalty.
            If ``None`` and ``config.kl_coefficient > 0``, ``old_log_probs``
            is used as the reference.
        config: Loss configuration.  Defaults to ``PPOPolicyLossConfig()``.

    Returns:
        Dict with ``grpo_policy_loss``, ``kl_penalty``, ``entropy_bonus``,
        ``total_loss``, ``clip_fraction``, and ``mean_ratio`` keys.
    """
    cfg = config or PPOPolicyLossConfig()

    adv = list(advantages)
    if cfg.whiten_advantages:
        adv_t = torch.tensor(adv, dtype=torch.float32)
        mask_t = torch.tensor(response_mask, dtype=torch.float32)
        adv_t = masked_whiten(adv_t, mask_t)
        adv = adv_t.tolist()

    base = compute_trajectory_policy_loss(
        new_log_probs=new_log_probs,
        old_log_probs=old_log_probs,
        advantages=adv,
        response_mask=response_mask,
        ref_log_probs=ref_log_probs,
        clip_epsilon=cfg.clip_epsilon,
        kl_beta=cfg.kl_coefficient,
    )

    entropy_bonus = 0.0
    if cfg.entropy_coefficient != 0.0:
        new_lp = torch.tensor(new_log_probs, dtype=torch.float32)
        mask_t = torch.tensor(response_mask, dtype=torch.float32)
        # Entropy approximation: H ≈ -E[log p], averaged over masked tokens.
        h = masked_mean(-new_lp, mask_t)
        entropy_bonus = float(cfg.entropy_coefficient * h)

    total = base["total_loss"] - entropy_bonus
    return {
        **base,
        "entropy_bonus": entropy_bonus,
        "total_loss": total,
    }


@dataclass(frozen=True)
class PromptGroupSamplingConfig:
    """How to generate multiple strategy rollouts for one prompt group."""

    num_rollouts: int = 4
    temperature_step: float = 0.15
    top_p_step: float = 0.03
    max_temperature: float = 1.1
    max_top_p: float = 1.0


@dataclass(frozen=True)
class GRPORolloutSample:
    """One rollout plus its sampling metadata."""

    group_id: str
    rollout_index: int
    sampling_params: dict[str, Any]
    output: AgentLoopOutput


@dataclass(frozen=True)
class ScoredGRPORollout:
    """One rollout with reward breakdown and within-group advantage."""

    group_id: str
    rollout_index: int
    sampling_params: dict[str, Any]
    output: AgentLoopOutput
    reward: float
    reward_component: str
    reward_components: dict[str, float]
    advantage: float


@dataclass(frozen=True)
class GRPOAdvantageConfig:
    """How grouped rollout rewards are converted into training advantages.

    Modes
    -----
    ``group_outcome``
        Mean-center within the group; no std normalization.  The original
        DeepSeek-R1 / GRPO formulation.
    ``group_std_normalized``
        Mean-center **and** divide by within-group std + ε.  Reduces gradient
        variance at the cost of scaling information.
    ``reinforce_baseline``
        Alias for ``group_outcome``.  Uses the within-group mean as a running
        baseline, matching the classical REINFORCE-with-baseline update.
    ``dapo``
        Direct Advantage Policy Optimization (DAPO / Dr. GRPO): reward is used
        as the advantage without any group normalization.  Suitable when group
        sizes are very small (G=1 or G=2) or when within-group variance is
        already low.

    Clipping and scaling
    --------------------
    ``reward_scale``
        Multiply each raw reward before advantage computation.
    ``clip_range``
        If set, advantages are clipped to ``[-clip_range, clip_range]`` after
        normalization.  Prevents extreme advantages from dominating gradient
        updates.
    """

    mode: str = "group_std_normalized"
    reward_component: str = "total"
    reward_scale: float = 1.0
    clip_range: float | None = None

    @classmethod
    def outcome_only(cls) -> "GRPOAdvantageConfig":
        """Preset for critic-free final-outcome GRPO (DeepSeek-R1 / sparse reward).

        Group-relative advantage from the terminal judge score only:

            A_i = r_i - mean({r_j : group_j == group_i})
            r_i = correctness_weight * judge(answer_i, ground_truth)

        No std normalisation, no process shaping, no value model.
        This is the sparse-reward variant described in the DeepSeek-R1 paper
        and suited for long-CoT / agent trajectories.
        """
        return cls(mode="group_outcome", reward_component="terminal_reward")

    @classmethod
    def std_normalized(cls) -> "GRPOAdvantageConfig":
        """Preset for std-normalized group-relative GRPO.

        Advantages are mean-centered and divided by within-group std:

            A_i = (r_i - mean(group)) / (std(group) + ε)

        Uses the shaped total reward, which includes process-shaping terms
        such as citation support, search efficiency, and duplicate-query
        penalties.  If you want std-normalisation with sparse rewards, use
        :meth:`outcome_only` and pass ``SearchRewardConfig.sparse_final_only()``.
        """
        return cls(mode="group_std_normalized", reward_component="total")

    @classmethod
    def reinforce_with_baseline(
        cls,
        *,
        reward_component: str = "total",
    ) -> "GRPOAdvantageConfig":
        """Preset for REINFORCE with within-group mean as baseline.

        Identical to ``outcome_only`` but uses the full shaped reward by
        default instead of the terminal-only score.  The within-group mean
        acts as a control variate that reduces variance without std scaling.

            A_i = r_i - mean(group_rewards)

        Choose this over :meth:`std_normalized` when you want the magnitude
        of the advantages to reflect actual reward differences (std scaling
        can suppress useful gradient signal when group variance is low).
        """
        return cls(mode="reinforce_baseline", reward_component=reward_component)

    @classmethod
    def dapo(
        cls,
        *,
        reward_component: str = "total",
        clip_range: float | None = 1.0,
    ) -> "GRPOAdvantageConfig":
        """Preset for DAPO / Dr. GRPO: reward directly as advantage.

        No group centering, no std normalization — each rollout's reward is
        used as its own advantage.  Apply ``clip_range`` to prevent outlier
        rewards from producing excessively large gradient steps.

            A_i = clip(reward_i, -clip_range, +clip_range)

        Use this when:
        - Group size G=1 or G=2 makes within-group statistics unreliable.
        - You have an absolute reward scale you want to preserve.
        - You want to debug reward shapes without group-normalization masking.
        """
        return cls(
            mode="dapo", reward_component=reward_component, clip_range=clip_range
        )


def build_grpo_sampling_params(
    base_sampling_params: dict[str, Any],
    *,
    num_rollouts: int,
    config: PromptGroupSamplingConfig | None = None,
) -> list[dict[str, Any]]:
    """Create slightly diversified sampling params for one prompt group."""
    cfg = config or PromptGroupSamplingConfig(num_rollouts=num_rollouts)
    if num_rollouts <= 0:
        raise ValueError("num_rollouts must be positive.")

    base_temp = float(base_sampling_params.get("temperature", 0.7))
    base_top_p = float(base_sampling_params.get("top_p", 0.95))

    variants: list[dict[str, Any]] = []
    for rollout_index in range(num_rollouts):
        params = dict(base_sampling_params)
        params["temperature"] = min(
            cfg.max_temperature,
            max(0.0, base_temp + rollout_index * cfg.temperature_step),
        )
        params["top_p"] = min(
            cfg.max_top_p,
            max(0.0, base_top_p + rollout_index * cfg.top_p_step),
        )
        variants.append(params)
    return variants


async def sample_prompt_group(
    loop_factory: Callable[[], AgentLoopBase],
    *,
    question: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    sampling_params: dict[str, Any],
    num_rollouts: int = 4,
    sampling_variants: list[dict[str, Any]] | None = None,
    group_id: str | None = None,
) -> list[GRPORolloutSample]:
    """Generate multiple rollouts for the same prompt group under one group id."""
    if num_rollouts <= 0:
        raise ValueError("num_rollouts must be positive.")
    if question is None and messages is None:
        raise ValueError("Either `question` or `messages` must be provided.")
    if question is not None and messages is not None:
        raise ValueError("Provide only one of `question` or `messages`.")

    resolved_group_id = group_id or f"prompt_group_{uuid4().hex}"
    variants = sampling_variants or build_grpo_sampling_params(
        sampling_params,
        num_rollouts=num_rollouts,
    )
    if len(variants) != num_rollouts:
        raise ValueError("sampling_variants length must equal num_rollouts.")

    resolved_messages = (
        [{"role": "user", "content": question}] if messages is None else list(messages)
    )

    async def _run_rollout(
        rollout_index: int, rollout_sampling_params: dict[str, Any]
    ) -> GRPORolloutSample:
        loop = loop_factory()
        output = await loop.run(
            messages=resolved_messages,
            sampling_params=rollout_sampling_params,
        )
        output.group_id = resolved_group_id
        output.rollout_index = rollout_index
        return GRPORolloutSample(
            group_id=resolved_group_id,
            rollout_index=rollout_index,
            sampling_params=dict(rollout_sampling_params),
            output=output,
        )

    return list(
        await asyncio.gather(*[_run_rollout(i, p) for i, p in enumerate(variants)])
    )


async def sample_prompt_batch(
    loop_factory: Callable[[], AgentLoopBase],
    batch: PromptBatch,
    *,
    sampling_params: dict[str, Any],
    num_rollouts: int = 4,
    sampling_variants: list[dict[str, Any]] | None = None,
    max_concurrent: int | None = None,
) -> list[list[GRPORolloutSample]]:
    """Generate rollout groups for every prompt in a DataLoader batch.

    Each item in the batch becomes one prompt group with ``num_rollouts``
    concurrently sampled rollouts.

    Args:
        max_concurrent: Maximum number of prompt groups to sample at the same
            time.  ``None`` (default) fires all groups concurrently.  Set this
            to a small value (e.g. ``4``–``8``) when the inference server has
            limited capacity to avoid request queue saturation.
    """

    async def _sample_one(messages: list[dict[str, Any]]) -> list[GRPORolloutSample]:
        return await sample_prompt_group(
            loop_factory,
            messages=messages,
            sampling_params=sampling_params,
            num_rollouts=num_rollouts,
            sampling_variants=sampling_variants,
        )

    if max_concurrent is None:
        return list(await asyncio.gather(*[_sample_one(m) for m in batch.messages]))

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _bounded(messages: list[dict[str, Any]]) -> list[GRPORolloutSample]:
        async with semaphore:
            return await _sample_one(messages)

    return list(await asyncio.gather(*[_bounded(m) for m in batch.messages]))


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
        from ..reward import normalize_answer_text, token_f1_score

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
    fallback: GoldAgreementJudge = dataclass_field(default_factory=GoldAgreementJudge)
    parse_failures: int = 0
    llm_calls: int = 0
    _cache: dict[tuple[str, str], float] = dataclass_field(
        default_factory=dict, repr=False
    )

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


def score_prompt_group(
    samples: list[GRPORolloutSample],
    *,
    ground_truth: str,
    judge_fn: JudgeFn,
    reward_fn: SearchRewardFunction | None = None,
    advantage_config: GRPOAdvantageConfig | None = None,
    batch_judge_fn: BatchJudgeFn | None = None,
    metadata: dict | None = None,
) -> list[ScoredGRPORollout]:
    """Score a prompt group and compute GRPO advantages within that group.

    All rollouts in the group share the same ``ground_truth`` (they were all
    sampled from the same prompt).  The judge is called once per sample, or
    once for the whole group when ``batch_judge_fn`` is supplied — pass this
    for LLM judges to avoid ``num_rollouts`` separate API calls.
    """
    if not samples:
        return []

    reward_function = reward_fn or SearchRewardFunction()
    resolved_advantage_config = advantage_config or GRPOAdvantageConfig()

    # Score all answers in one batch call if a batch judge is available.
    answers = [s.output.final_answer or "" for s in samples]
    gt_list = [ground_truth] * len(samples)
    correctness_scores = _score_answers(
        judge_fn, answers, gt_list, batch_judge_fn=batch_judge_fn
    )

    rewards: list[float] = []
    reward_components: list[dict[str, float]] = []
    group_ids: list[str] = []

    human_signal: float | None = metadata.get("human_signal") if metadata else None

    reward_scale = float(resolved_advantage_config.reward_scale)
    for sample, correctness in zip(samples, correctness_scores):
        components = reward_function._reward_components_from_correctness(
            sample.output, correctness, human_signal=human_signal
        )
        reward_components.append(components)
        raw_reward = _select_reward_component(
            components,
            component=resolved_advantage_config.reward_component,
        )
        rewards.append(raw_reward * reward_scale)
        group_ids.append(sample.group_id)

    advantages = _compute_advantages(
        reward_function,
        rewards,
        group_ids,
        mode=resolved_advantage_config.mode,
        clip_range=resolved_advantage_config.clip_range,
    )
    return [
        ScoredGRPORollout(
            group_id=sample.group_id,
            rollout_index=sample.rollout_index,
            sampling_params=sample.sampling_params,
            output=sample.output,
            reward=reward,
            reward_component=resolved_advantage_config.reward_component,
            reward_components=components,
            advantage=advantage,
        )
        for sample, reward, components, advantage in zip(
            samples, rewards, reward_components, advantages
        )
    ]


def _compute_grpo_outcome_advantage_list(rewards: list[float]) -> list[float]:
    """Compute outcome-based GRPO advantages for one prompt group.

    This is the critic-free core signal used by GRPO:

        advantage_i = reward_i - mean(group_rewards)

    For a single trajectory there is no relative comparison, so the returned
    advantage is `0.0`.
    """
    if not rewards:
        return []
    if len(rewards) == 1:
        return [0.0]
    mean = sum(rewards) / len(rewards)
    return [reward - mean for reward in rewards]


_NOT_GIVEN = object()


@overload
def compute_grpo_outcome_advantage(rewards: list[float]) -> list[float]: ...


@overload
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    clip_advantages: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]: ...


def compute_grpo_outcome_advantage(
    rewards: list[float] | torch.Tensor | object = _NOT_GIVEN,
    eos_mask: torch.Tensor | object = _NOT_GIVEN,
    index: torch.Tensor | object = _NOT_GIVEN,
    epsilon: float = 1e-6,
    clip_advantages: float | None = None,
    *,
    token_level_rewards: torch.Tensor | object = _NOT_GIVEN,
) -> list[float] | tuple[torch.Tensor, torch.Tensor]:
    """Compute the established list- or tensor-native GRPO advantages.

    The former rollout and loss modules exposed two call shapes under this
    name. Consolidation keeps both forms accepted by one public owner.
    """
    if token_level_rewards is not _NOT_GIVEN:
        if rewards is not _NOT_GIVEN:
            raise TypeError(
                "compute_grpo_outcome_advantage received both rewards and "
                "token_level_rewards"
            )
        rewards = token_level_rewards

    if rewards is _NOT_GIVEN:
        raise TypeError("compute_grpo_outcome_advantage missing required rewards")

    if isinstance(rewards, torch.Tensor):
        if eos_mask is _NOT_GIVEN or index is _NOT_GIVEN:
            raise TypeError(
                "tensor GRPO advantages require eos_mask and index arguments"
            )
        return _compute_grpo_outcome_advantage_tensor(
            rewards,
            eos_mask,
            index,
            epsilon=epsilon,
            clip_advantages=clip_advantages,
        )

    if eos_mask is not _NOT_GIVEN or index is not _NOT_GIVEN:
        raise TypeError("list GRPO advantages accept only the rewards argument")
    return _compute_grpo_outcome_advantage_list(rewards)


def compute_dapo_advantages(
    rewards: list[float],
    *,
    clip_range: float | None = None,
) -> list[float]:
    """DAPO / Dr. GRPO advantages: reward used directly, no group normalization.

    Each rollout's reward is its own advantage.  Useful when group sizes are
    too small for reliable within-group statistics (G=1 or G=2).

    Args:
        rewards: Raw scalar rewards, one per rollout.
        clip_range: If given, clip advantages to ``[-clip_range, +clip_range]``.

    Returns:
        Advantages aligned with *rewards*.
    """
    if clip_range is not None:
        lo, hi = -float(clip_range), float(clip_range)
        return [max(lo, min(hi, float(r))) for r in rewards]
    return [float(r) for r in rewards]


def _select_reward_component(
    reward_components: dict[str, float],
    *,
    component: str,
) -> float:
    """Pick which scalar reward should feed GRPO advantage estimation."""
    if component not in reward_components:
        available = ", ".join(sorted(reward_components))
        raise ValueError(
            f"Unsupported GRPO reward component: {component!r}. "
            f"Available components: {available}."
        )
    return float(reward_components[component])


def _compute_advantages(
    reward_function: SearchRewardFunction,
    rewards: list[float],
    group_ids: list[str],
    *,
    mode: str,
    clip_range: float | None = None,
) -> list[float]:
    """Resolve which GRPO advantage transform to apply to rollout rewards."""
    if mode in ("group_outcome", "reinforce_baseline"):
        advantages = reward_function.compute_grpo_outcome_advantages(rewards, group_ids)
    elif mode == "group_std_normalized":
        advantages = reward_function.compute_batch_advantages(rewards, group_ids)
    elif mode == "dapo":
        advantages = compute_dapo_advantages(rewards, clip_range=None)
    else:
        raise ValueError(
            f"Unsupported GRPO advantage mode: {mode!r}. "
            "Expected 'group_outcome', 'group_std_normalized', "
            "'reinforce_baseline', or 'dapo'."
        )
    if clip_range is not None and mode != "dapo":
        lo, hi = -float(clip_range), float(clip_range)
        advantages = [max(lo, min(hi, a)) for a in advantages]
    return advantages


def score_prompt_batch(
    grouped_samples: list[list[GRPORolloutSample]],
    *,
    ground_truths: list[str],
    judge_fn: JudgeFn,
    reward_fn: SearchRewardFunction | None = None,
    advantage_config: GRPOAdvantageConfig | None = None,
    batch_judge_fn: BatchJudgeFn | None = None,
) -> list[list[ScoredGRPORollout]]:
    """Score all rollout groups from a DataLoader batch.

    Args:
        grouped_samples: output of ``sample_prompt_batch`` — one list per prompt.
        ground_truths: aligned to ``batch.ground_truths``.
        batch_judge_fn: Optional batch judge called once per prompt group
            instead of once per rollout.  Pass this for LLM judges to reduce
            API calls from ``num_prompts * num_rollouts`` down to ``num_prompts``.
    """
    if len(grouped_samples) != len(ground_truths):
        raise ValueError("grouped_samples and ground_truths must have the same length.")
    reward_function = reward_fn or SearchRewardFunction()
    return [
        score_prompt_group(
            samples,
            ground_truth=gt,
            judge_fn=judge_fn,
            reward_fn=reward_function,
            advantage_config=advantage_config,
            batch_judge_fn=batch_judge_fn,
        )
        for samples, gt in zip(grouped_samples, ground_truths)
    ]


# ---------------------------------------------------------------------------
# On-policy GRPO batch assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OnPolicyGRPOConfig:
    """Configuration for on-policy GRPO batch assembly."""

    min_reward_range: float = 0.0
    normalize_globally: bool = False
    max_groups: int | None = None


@dataclass(frozen=True)
class OnPolicyBatchStats:
    """Diagnostics returned by ``compute_on_policy_batch_stats``."""

    n_groups_total: int
    n_groups_kept: int
    n_rollouts_kept: int
    pct_groups_kept: float
    mean_reward: float
    reward_std: float


def filter_zero_advantage_groups(
    scored_groups: list[list[ScoredGRPORollout]],
    *,
    min_reward_range: float = 0.0,
) -> list[list[ScoredGRPORollout]]:
    """Drop groups whose reward range is at or below *min_reward_range*.

    A group where every rollout receives the same reward produces zero
    advantage for all members and contributes nothing to the GRPO gradient.
    Filtering these groups before the loss step reduces wasted compute.
    """
    result = []
    for group in scored_groups:
        rewards = [r.reward for r in group]
        if max(rewards) - min(rewards) > min_reward_range:
            result.append(group)
    return result


def assemble_on_policy_batch(
    scored_groups: list[list[ScoredGRPORollout]],
    config: OnPolicyGRPOConfig | None = None,
) -> list[ScoredGRPORollout]:
    """Filter dead groups, optionally normalise globally, and flatten.

    Steps:
    1. Drop groups where ``max(reward) - min(reward) <= config.min_reward_range``.
    2. If ``config.normalize_globally``, re-centre and rescale advantages
       across all remaining rollouts using global mean/std.
    3. Truncate to ``config.max_groups`` groups if set.
    4. Flatten to a single list.
    """
    cfg = config or OnPolicyGRPOConfig()

    kept = filter_zero_advantage_groups(
        scored_groups, min_reward_range=cfg.min_reward_range
    )

    if cfg.max_groups is not None:
        kept = kept[: cfg.max_groups]

    flat: list[ScoredGRPORollout] = [r for group in kept for r in group]

    if cfg.normalize_globally and flat:
        adv_values = [r.advantage for r in flat]
        mean = sum(adv_values) / len(adv_values)
        variance = sum((a - mean) ** 2 for a in adv_values) / len(adv_values)
        std = math.sqrt(variance) if variance > 0 else 1.0
        flat = [replace(r, advantage=(r.advantage - mean) / std) for r in flat]

    return flat


def compute_on_policy_batch_stats(
    scored_groups_before_filter: list[list[ScoredGRPORollout]],
    flat_batch_after_filter: list[ScoredGRPORollout],
) -> OnPolicyBatchStats:
    """Compute diagnostics comparing the raw and filtered batches."""
    n_total = len(scored_groups_before_filter)
    n_kept = len({r.group_id for r in flat_batch_after_filter})
    n_rollouts = len(flat_batch_after_filter)
    pct = n_kept / n_total if n_total > 0 else 0.0

    rewards = [r.reward for r in flat_batch_after_filter]
    if rewards:
        mean = sum(rewards) / len(rewards)
        std = math.sqrt(sum((x - mean) ** 2 for x in rewards) / len(rewards))
    else:
        mean = 0.0
        std = 0.0

    return OnPolicyBatchStats(
        n_groups_total=n_total,
        n_groups_kept=n_kept,
        n_rollouts_kept=n_rollouts,
        pct_groups_kept=pct,
        mean_reward=mean,
        reward_std=std,
    )
