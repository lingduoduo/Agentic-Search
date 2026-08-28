"""Contracts for the simulated cohort.

The generator holds the conversion rule; the analysis never sees it. These
tests check that the knobs do what they claim, because every downstream claim
about detected power is a claim about this generator being honest.
"""

from __future__ import annotations

import pytest

from src.model.post_training.eval.cohort import (
    CohortConfig,
    EvalRecord,
    generate_cohort,
    null_cohort_config,
)


def test_cohort_is_deterministic_for_a_fixed_seed():
    config = CohortConfig(num_users=6, sessions_per_user=4, seed=5)

    assert generate_cohort(config) == generate_cohort(config)


def test_a_different_seed_produces_a_different_cohort():
    first = generate_cohort(CohortConfig(num_users=6, sessions_per_user=4, seed=1))
    second = generate_cohort(CohortConfig(num_users=6, sessions_per_user=4, seed=2))

    assert first != second


def test_every_user_answers_every_prompt_under_both_policies():
    records = generate_cohort(CohortConfig(num_users=5, sessions_per_user=3, seed=0))

    by_policy = {"trained": set(), "baseline": set()}
    for record in records:
        by_policy[record.policy].add((record.user_id, record.prompt_id))

    assert by_policy["trained"] == by_policy["baseline"]
    assert len(by_policy["trained"]) == 15


def test_records_are_frozen():
    record = generate_cohort(CohortConfig(num_users=2, sessions_per_user=2))[0]

    with pytest.raises(AttributeError):
        record.reward = 1.0  # type: ignore[misc]


def test_alignment_makes_reward_track_conversion():
    records = generate_cohort(
        CohortConfig(num_users=30, sessions_per_user=10, alignment=3.0, seed=3)
    )

    converted = [r.reward for r in records if r.converted]
    unconverted = [r.reward for r in records if not r.converted]
    assert sum(converted) / len(converted) > sum(unconverted) / len(unconverted)


def test_zero_alignment_decouples_reward_from_conversion():
    records = generate_cohort(
        CohortConfig(num_users=40, sessions_per_user=10, alignment=0.0, seed=4)
    )

    converted = [r.reward for r in records if r.converted]
    unconverted = [r.reward for r in records if not r.converted]
    difference = abs(
        sum(converted) / len(converted) - sum(unconverted) / len(unconverted)
    )
    assert difference < 0.15


def test_behavior_shift_makes_the_trained_policy_search_less():
    records = generate_cohort(
        CohortConfig(num_users=30, sessions_per_user=8, behavior_shift=2.0, seed=6)
    )

    def mean_rounds(policy: str) -> float:
        values = [r.metrics["search_rounds"] for r in records if r.policy == policy]
        return sum(values) / len(values)

    assert mean_rounds("trained") < mean_rounds("baseline")


def test_instruction_gap_makes_trained_responses_more_compliant():
    records = generate_cohort(
        CohortConfig(num_users=30, sessions_per_user=8, instruction_gap=0.4, seed=7)
    )

    def tagged(policy: str) -> float:
        rows = [r for r in records if r.policy == policy]
        return sum("<answer>" in r.response for r in rows) / len(rows)

    assert tagged("trained") > tagged("baseline")


def test_most_users_have_both_outcomes_so_auc_is_defined():
    records = generate_cohort(CohortConfig(num_users=40, sessions_per_user=12, seed=8))

    outcomes: dict[str, set[bool]] = {}
    for record in records:
        outcomes.setdefault(record.user_id, set()).add(record.converted)
    mixed = sum(1 for values in outcomes.values() if len(values) == 2)

    assert mixed / len(outcomes) >= 0.8


def test_null_config_zeroes_all_three_effects():
    null = null_cohort_config(CohortConfig(num_users=9, sessions_per_user=5, seed=2))

    assert null.alignment == 0.0
    assert null.behavior_shift == 0.0
    assert null.instruction_gap == 0.0
    assert null.num_users == 9
    assert null.sessions_per_user == 5
    assert null.seed == 2


def test_every_cited_id_appears_as_a_label_in_its_response():
    records = generate_cohort(CohortConfig(num_users=8, sessions_per_user=4, seed=9))

    cited_anywhere = 0
    for record in records:
        for label in record.cited_ids:
            assert f"[{label}]" in record.response
            cited_anywhere += 1

    # Guard against the assertion loop never running.
    assert cited_anywhere > 0


def test_records_compare_by_value():
    """What the determinism tests above actually rely on.

    Not hashability: a frozen dataclass carrying a `dict` field is unhashable,
    and `metrics` is a dict.
    """
    fields = {
        "user_id": "u",
        "prompt_id": "p",
        "policy": "trained",
        "reward": 0.5,
        "converted": True,
        "response": "<answer>x</answer>",
        "metrics": {"search_rounds": 1.0},
        "cited_ids": frozenset({"R1Q1D1"}),
        "tool_calls": (),
    }

    assert EvalRecord(**fields) == EvalRecord(**fields)
    assert EvalRecord(**{**fields, "reward": 0.6}) != EvalRecord(**fields)
