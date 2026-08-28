"""A simulated user population with a conversion rule the analysis never sees.

This exists because the repository has two users and zero feedback rows (see the
design doc), so no statement about unseen users can be made from its data. The
generator supplies a population whose ground truth is known, which turns the
harness's output into a statement about the pipeline -- "it detects an effect of
this size at this power" -- rather than an unbacked claim about real users.

Three independent knobs, each of which can be set to zero to remove its effect:

    alignment        how strongly reward predicts conversion
    behavior_shift   how much less the trained policy searches
    instruction_gap  how much more often the trained policy complies

Zeroing all three gives the null cohort, on which the harness must fail to find
significance. That test is what stops this from being a machine for producing
p-values.

``converted`` is drawn once per (user, prompt) session and shared identically by
both policy arms: conversion is a property of the session being evaluated, not
of which policy answered it. A consequence follows directly from that choice --
comparing raw ``converted`` rates between the trained and baseline arms shows
exactly zero lift, by construction. Any trained-vs-baseline effect on
conversion has to be read off ``reward`` (via ``alignment``), never off
``converted`` directly.

``search_rounds`` and instruction compliance also carry a per-user latent
offset, drawn once per user and applied identically to both arms (the same
"shared, not knob-gated" pattern as conversion above). This is what makes a
user's sessions on those two axes correlated rather than i.i.d., which is what
a per-user cluster bootstrap over those fields is actually testing against.

This module is the seam. A reader over ``chat_sessions`` + ``retrieval_feedback``
producing the same ``EvalRecord`` changes what the numbers mean without changing
a line of the analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

_POLICIES = ("trained", "baseline")


@dataclass(frozen=True)
class EvalRecord:
    """One rollout. The contract between a population and the analysis."""

    user_id: str
    prompt_id: str
    policy: str
    reward: float
    converted: bool
    response: str
    metrics: dict[str, float] = field(default_factory=dict)
    cited_ids: frozenset[str] = frozenset()
    tool_calls: tuple[str, ...] = ()


@dataclass(frozen=True)
class CohortConfig:
    """Population shape and the size of each planted effect."""

    num_users: int = 40
    sessions_per_user: int = 12
    alignment: float = 2.0
    behavior_shift: float = 1.5
    instruction_gap: float = 0.25
    base_compliance: float = 0.6
    seed: int = 0


def null_cohort_config(config: CohortConfig) -> CohortConfig:
    """The same population with every planted effect removed."""
    return replace(config, alignment=0.0, behavior_shift=0.0, instruction_gap=0.0)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-x)))


def generate_cohort(config: CohortConfig) -> list[EvalRecord]:
    """Generate one population. Deterministic in ``config.seed``."""
    if config.num_users < 1 or config.sessions_per_user < 1:
        raise ValueError("num_users and sessions_per_user must both be positive.")

    rng = np.random.default_rng(config.seed)
    records: list[EvalRecord] = []

    for user_index in range(config.num_users):
        user_id = f"u{user_index:03d}"
        # Per-user latents: this is what makes sessions from one user
        # correlated, and therefore what makes clustering necessary. Each
        # applies identically to both arms -- only the effect knobs below are
        # allowed to create a trained-vs-baseline gap.
        affinity = float(rng.normal(0.0, 1.0))
        rounds_offset = float(rng.normal(0.0, 1.2))
        compliance_offset = float(rng.normal(0.0, 0.15))

        for session_index in range(config.sessions_per_user):
            prompt_id = f"p{session_index:03d}"
            # quality is the sole variable shared by reward and conversion.
            # alignment scales both of them; at alignment=0 neither carries
            # any information about the other, by construction.
            quality = float(rng.normal(0.0, 1.0))
            convert_p = _sigmoid(config.alignment * quality + affinity)
            converted = bool(rng.random() < convert_p)

            for policy in _POLICIES:
                is_trained = policy == "trained"
                # Reward's only link to conversion is through quality, scaled
                # by alignment -- this is the one signal alignment measures.
                reward = config.alignment * quality + float(rng.normal(0.0, 1.0))

                base_rate = max(0.5, 4.0 + rounds_offset)
                rounds = max(
                    0.0,
                    float(rng.poisson(base_rate))
                    - (config.behavior_shift if is_trained else 0.0),
                )
                compliance_p = (
                    config.base_compliance
                    + compliance_offset
                    + (config.instruction_gap if is_trained else 0.0)
                )
                complies = bool(rng.random() < min(1.0, max(0.0, compliance_p)))

                citation = "R1Q1D1"
                body = f"Retrieved evidence for {prompt_id}. [{citation}]"
                response = f"<answer>{body}</answer>" if complies else body
                cited = frozenset({citation}) if complies else frozenset()
                tool_calls = (
                    ('{"name": "search", "arguments": {"query": "q"}}',)
                    if complies
                    else ("{not json",)
                )

                records.append(
                    EvalRecord(
                        user_id=user_id,
                        prompt_id=prompt_id,
                        policy=policy,
                        reward=reward,
                        converted=converted,
                        response=response,
                        metrics={
                            "search_rounds": rounds,
                            "web_searches": float(rng.poisson(1.0)),
                            "vdb_searches": float(rng.poisson(2.0)),
                            "rerank_calls": float(rng.poisson(0.5)),
                            "repeated_search_queries": float(rng.poisson(0.5)),
                            "rounds_used": rounds,
                        },
                        cited_ids=cited,
                        tool_calls=tool_calls,
                    )
                )

    return records
