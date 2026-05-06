"""Utilities for grouped rollout sampling and scoring for GRPO-style training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from uuid import uuid4

from .agent_loop import AgentLoopBase, AgentLoopOutput
from .reward import SearchRewardFunction


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
    reward_components: dict[str, float]
    advantage: float


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
    question: str,
    sampling_params: dict[str, Any],
    num_rollouts: int = 4,
    sampling_variants: list[dict[str, Any]] | None = None,
    group_id: str | None = None,
) -> list[GRPORolloutSample]:
    """Generate multiple rollouts for the same question under one group id."""
    if num_rollouts <= 0:
        raise ValueError("num_rollouts must be positive.")

    resolved_group_id = group_id or f"prompt_group_{uuid4().hex}"
    variants = sampling_variants or build_grpo_sampling_params(
        sampling_params,
        num_rollouts=num_rollouts,
    )
    if len(variants) != num_rollouts:
        raise ValueError("sampling_variants length must equal num_rollouts.")

    samples: list[GRPORolloutSample] = []
    for rollout_index, rollout_sampling_params in enumerate(variants):
        loop = loop_factory()
        output = await loop.run(
            messages=[{"role": "user", "content": question}],
            sampling_params=rollout_sampling_params,
        )
        output.group_id = resolved_group_id
        output.rollout_index = rollout_index
        samples.append(
            GRPORolloutSample(
                group_id=resolved_group_id,
                rollout_index=rollout_index,
                sampling_params=dict(rollout_sampling_params),
                output=output,
            )
        )
    return samples


def score_prompt_group(
    samples: list[GRPORolloutSample],
    *,
    ground_truth: str,
    judge_fn: Callable[[str, str], float],
    reward_fn: SearchRewardFunction | None = None,
) -> list[ScoredGRPORollout]:
    """Score a prompt group and compute GRPO advantages within that group."""
    if not samples:
        return []

    reward_function = reward_fn or SearchRewardFunction()
    rewards: list[float] = []
    reward_components: list[dict[str, float]] = []
    group_ids: list[str] = []

    for sample in samples:
        components = reward_function.reward_components(
            sample.output,
            ground_truth=ground_truth,
            judge_fn=judge_fn,
        )
        reward_components.append(components)
        rewards.append(components["total"])
        group_ids.append(sample.group_id)

    advantages = reward_function.compute_batch_advantages(rewards, group_ids)
    return [
        ScoredGRPORollout(
            group_id=sample.group_id,
            rollout_index=sample.rollout_index,
            sampling_params=sample.sampling_params,
            output=sample.output,
            reward=reward,
            reward_components=components,
            advantage=advantage,
        )
        for sample, reward, components, advantage in zip(
            samples, rewards, reward_components, advantages
        )
    ]
