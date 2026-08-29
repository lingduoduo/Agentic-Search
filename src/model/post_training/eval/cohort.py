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

Every behavioral metric and instruction compliance also carry a per-user latent
offset, drawn once per user and applied identically to both arms (the same
"shared, not knob-gated" pattern as conversion above). This is what makes a
user's sessions correlated rather than i.i.d. on every axis the analysis
clusters over, which is what a per-user cluster bootstrap is actually testing
against. A component with no such latent would be exchangeable across users and
would make the clustering machinery look better than it is.

The three response-shaped constraints (answer tag, citations, tool calls) are
three *separate* Bernoulli draws sharing one probability. They stay correlated
-- a compliant policy tends to be compliant on all three -- without being the
same coin flip printed three times, which would put three exact duplicates
inside the analysis's Benjamini-Hochberg family.

This module is the seam. A reader over ``chat_sessions`` + ``retrieval_feedback``
producing the same ``EvalRecord`` changes what the numbers mean without changing
a line of the analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

_POLICIES = ("trained", "baseline")

# Base Poisson mean for each behavioral component, before the per-user latent.
_BEHAVIOR_LAMBDAS = {
    "web_searches": 1.0,
    "vdb_searches": 2.0,
    "rerank_calls": 0.5,
    "repeated_search_queries": 0.5,
}
# Spread of the per-user activity latent (log scale), and the baseline
# probability that a rollout blows its round budget.
_ACTIVITY_SIGMA = 0.6
_BASE_OVERRUN_RATE = 0.5


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

    num_users: int = 60
    sessions_per_user: int = 12
    alignment: float = 2.0
    behavior_shift: float = 1.5
    instruction_gap: float = 0.25
    base_compliance: float = 0.6
    seed: int = 0


def null_cohort_config(config: CohortConfig) -> CohortConfig:
    """The same population with every planted effect removed."""
    return replace(config, alignment=0.0, behavior_shift=0.0, instruction_gap=0.0)


def effect_size_summary(config: CohortConfig) -> str:
    """The planted effect sizes, for the report's provenance line.

    "Power 1.00" is uninterpretable without the size of the effect that power
    was achieved against, and the one sentence this harness is entitled to
    claim -- "the pipeline detects an effect of *this size*, on held-out users,
    at *this power*" -- cannot be completed from a report that omits it.
    """
    return (
        f"planted effect sizes: alignment={config.alignment:g}, "
        f"behavior_shift={config.behavior_shift:g}, "
        f"instruction_gap={config.instruction_gap:g}"
    )


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
        # allowed to create a trained-vs-baseline gap. Drawn here, outside the
        # policy loop, for exactly that reason.
        affinity = float(rng.normal(0.0, 1.0))
        rounds_offset = float(rng.normal(0.0, 1.2))
        compliance_offset = float(rng.normal(0.0, 0.15))
        activity = {
            name: float(np.exp(rng.normal(0.0, _ACTIVITY_SIGMA)))
            for name in _BEHAVIOR_LAMBDAS
        }

        for session_index in range(config.sessions_per_user):
            prompt_id = f"p{session_index:03d}"
            # quality is the sole variable shared by reward and conversion.
            # alignment scales both of them; at alignment=0 neither carries
            # any information about the other, by construction.
            quality = float(rng.normal(0.0, 1.0))
            convert_p = _sigmoid(config.alignment * quality + affinity)
            converted = bool(rng.random() < convert_p)
            # Half the default allowed_tools would never be exercised if every
            # call were a search; alternating by prompt keeps both names live
            # without spending an RNG draw or splitting the two arms.
            tool_name = "fetch" if session_index % 3 == 0 else "search"

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
                # A policy that searches less also issues fewer web/vdb
                # searches, invokes fewer reranks, and repeats fewer queries.
                # Reuse behavior_shift rather than inventing a new knob: scale
                # its fractional pull on search_rounds' base rate (4.0) onto
                # each metric's own per-user mean, so at behavior_shift=0 the
                # reduction is exactly zero for both arms, and each arm still
                # draws one Poisson sample per metric regardless of policy --
                # only the post-draw subtraction differs, keeping the two
                # arms' draw sequences aligned.
                shift_fraction = config.behavior_shift / 4.0
                behavior: dict[str, float] = {}
                for name, base_lambda in _BEHAVIOR_LAMBDAS.items():
                    lam = base_lambda * activity[name]
                    behavior[name] = max(
                        0.0,
                        float(rng.poisson(lam))
                        - (lam * shift_fraction if is_trained else 0.0),
                    )

                # Blowing the round budget is an instruction-following failure,
                # so it is gated by instruction_gap and gets its own draws.
                # Reusing search_rounds as rounds_used -- as an earlier version
                # did -- made round_budget_respected a threshold on the
                # behavioral component, responding to behavior_shift and not to
                # the instruction knob it is supposed to measure.
                overrun_size = float(rng.poisson(1.5))
                overrun_rate = min(
                    1.0,
                    max(
                        0.0,
                        _BASE_OVERRUN_RATE
                        - (config.instruction_gap if is_trained else 0.0),
                    ),
                )
                overruns = bool(rng.random() < overrun_rate)
                rounds_used = rounds + (1.0 + overrun_size if overruns else 0.0)

                compliance_p = min(
                    1.0,
                    max(
                        0.0,
                        config.base_compliance
                        + compliance_offset
                        + (config.instruction_gap if is_trained else 0.0),
                    ),
                )
                # Three draws, one probability: correlated, not identical.
                answer_ok = bool(rng.random() < compliance_p)
                citations_ok = bool(rng.random() < compliance_p)
                tools_ok = bool(rng.random() < compliance_p)

                citation = "R1Q1D1"
                body = f"Retrieved evidence for {prompt_id}. [{citation}]"
                response = f"<answer>{body}</answer>" if answer_ok else body
                cited = frozenset({citation}) if citations_ok else frozenset()
                tool_calls = (
                    (f'{{"name": "{tool_name}", "arguments": {{"query": "q"}}}}',)
                    if tools_ok
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
                            **behavior,
                            "rounds_used": rounds_used,
                        },
                        cited_ids=cited,
                        tool_calls=tool_calls,
                    )
                )

    return records
