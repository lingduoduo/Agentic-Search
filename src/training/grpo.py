"""Utilities for grouped rollout sampling and scoring for GRPO-style training."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable
from uuid import uuid4

from ..agents.base import AgentLoopBase, AgentLoopOutput
from .data import PromptBatch
from .reward import BatchJudgeFn, JudgeFn, SearchRewardFunction, _score_answers


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
) -> list[list[GRPORolloutSample]]:
    """Generate rollout groups for every prompt in a DataLoader batch.

    Each item in the batch becomes one prompt group with `num_rollouts`
    concurrently sampled rollouts.  All groups are dispatched concurrently.
    """
    group_tasks = [
        sample_prompt_group(
            loop_factory,
            messages=messages,
            sampling_params=sampling_params,
            num_rollouts=num_rollouts,
            sampling_variants=sampling_variants,
        )
        for messages in batch.messages
    ]
    return list(await asyncio.gather(*group_tasks))


def score_prompt_group(
    samples: list[GRPORolloutSample],
    *,
    ground_truth: str,
    judge_fn: JudgeFn,
    reward_fn: SearchRewardFunction | None = None,
    advantage_config: GRPOAdvantageConfig | None = None,
    batch_judge_fn: BatchJudgeFn | None = None,
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

    reward_scale = float(resolved_advantage_config.reward_scale)
    for sample, correctness in zip(samples, correctness_scores):
        components = reward_function._reward_components_from_correctness(
            sample.output, correctness
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


def compute_grpo_outcome_advantage(rewards: list[float]) -> list[float]:
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
