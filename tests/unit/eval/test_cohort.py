"""Contracts for the simulated cohort.

The generator holds the conversion rule; the analysis never sees it. These
tests check that the knobs do what they claim, because every downstream claim
about detected power is a claim about this generator being honest.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.model.post_training.eval.cohort import (
    CohortConfig,
    EvalRecord,
    generate_cohort,
    null_cohort_config,
)


def _auc(scores: list[float], labels: list[bool]) -> float:
    """Rank-based AUC of ``scores`` predicting ``labels`` (ties count half).

    Scale-free, unlike a raw difference-of-means in reward units: an AUC of
    0.5 always means "no discrimination" regardless of how reward is scaled,
    which is what makes it a fair statistic to threshold on both when an
    effect should be present and when it should be exactly absent.
    """
    scores_arr = np.asarray(scores, dtype=float)
    labels_arr = np.asarray(labels, dtype=bool)
    positive = scores_arr[labels_arr]
    negative = scores_arr[~labels_arr]
    if len(positive) == 0 or len(negative) == 0:
        return 0.5
    greater = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(positive) * len(negative)))


def _variance_ratio(values_by_user: list[list[float]]) -> float:
    """Observed between-user variance divided by what i.i.d. sampling predicts.

    Near 1.0 means a user's sessions are exchangeable with anyone else's on
    this field (no per-user structure). Materially above 1.0 means a user's
    sessions correlate -- the assumption a per-user cluster bootstrap over
    that field is actually testing against.
    """
    arr = np.asarray(values_by_user, dtype=float)
    user_means = arr.mean(axis=1)
    between = user_means.var(ddof=1)
    within = arr.var(axis=1, ddof=1).mean()
    sessions = arr.shape[1]
    predicted_between_under_iid = within / sessions
    if predicted_between_under_iid <= 0:
        return float("inf")
    return float(between / predicted_between_under_iid)


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
    """Decisive across a seed sweep, not a coin flip at one pinned seed.

    A prior version of this test (mean-reward-of-converted > mean-reward-of-
    unconverted, at a single pinned seed) kept passing under two mutations
    that should have broken it: dropping reward's dependence on quality, and
    severing conversion's dependence on quality. Both left reward and
    converted genuinely uncorrelated, so the pinned seed's pass/fail was
    close to a 50/50 draw. AUC on a seed sweep does not have that problem --
    see the module's mutation-check notes in the task report.
    """
    for seed in (3, 13, 23, 33, 43):
        records = generate_cohort(
            CohortConfig(num_users=30, sessions_per_user=10, alignment=3.0, seed=seed)
        )
        auc = _auc([r.reward for r in records], [r.converted for r in records])
        assert auc > 0.7, f"seed={seed} auc={auc}"


def test_zero_alignment_decouples_reward_from_conversion():
    """Normalised on AUC so the tolerance isn't a fragile reward-unit magic
    number. A raw difference-of-means version of this assertion, tuned to
    the same population, exceeded its tolerance in about 1 seed in 20; AUC
    against 0.5 does not carry that same finite-sample scale sensitivity.
    """
    records = generate_cohort(
        CohortConfig(num_users=40, sessions_per_user=10, alignment=0.0, seed=4)
    )
    auc = _auc([r.reward for r in records], [r.converted for r in records])
    assert abs(auc - 0.5) < 0.1


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


def test_sessions_are_correlated_within_a_user_on_every_clustered_axis():
    """The cluster bootstrap is run over *every* behavioral component, not
    just search_rounds, so every component needs a per-user latent. A
    component drawn i.i.d. across users is exchangeable between users, which
    makes clustering it free -- and makes achieved power on that component
    look better than clustered data would actually give.
    """
    records = generate_cohort(CohortConfig(num_users=60, sessions_per_user=20, seed=11))

    by_user_converted: dict[str, list[float]] = {}
    by_user_complies: dict[str, list[float]] = {}
    by_user_metric: dict[str, dict[str, list[float]]] = {}
    seen_sessions: set[tuple[str, str]] = set()
    for record in records:
        session_key = (record.user_id, record.prompt_id)
        if session_key not in seen_sessions:
            seen_sessions.add(session_key)
            by_user_converted.setdefault(record.user_id, []).append(
                float(record.converted)
            )
        for name, value in record.metrics.items():
            by_user_metric.setdefault(name, {}).setdefault(record.user_id, []).append(
                value
            )
        by_user_complies.setdefault(record.user_id, []).append(
            float("<answer>" in record.response)
        )

    users = sorted(by_user_converted)
    converted_ratio = _variance_ratio([by_user_converted[u] for u in users])
    complies_ratio = _variance_ratio([by_user_complies[u] for u in users])

    assert converted_ratio > 1.5, converted_ratio
    assert complies_ratio > 1.5, complies_ratio
    for name in (
        "search_rounds",
        "web_searches",
        "vdb_searches",
        "rerank_calls",
        "repeated_search_queries",
        "rounds_used",
    ):
        ratio = _variance_ratio([by_user_metric[name][u] for u in users])
        assert ratio > 1.5, f"{name} shows no within-user correlation: {ratio}"


def test_the_three_response_constraints_are_not_one_shared_draw():
    """One `complies` boolean driving the answer tag, the citations and the
    tool call made three rows of the report identical on every record, and
    put three exact duplicates inside a Benjamini-Hochberg family of nine.
    They must be three draws -- correlated through the shared probability,
    but able to disagree.
    """
    records = generate_cohort(CohortConfig(num_users=40, sessions_per_user=12, seed=21))

    combinations = {
        (
            "<answer>" in record.response,
            bool(record.cited_ids),
            record.tool_calls[0].startswith('{"name"'),
        )
        for record in records
    }
    disagreements = sum(
        len(
            {
                "<answer>" in record.response,
                bool(record.cited_ids),
                record.tool_calls[0].startswith('{"name"'),
            }
        )
        > 1
        for record in records
    )

    # A single shared draw can only ever produce (T,T,T) and (F,F,F).
    assert len(combinations) > 2
    assert disagreements / len(records) > 0.1, disagreements


def test_rounds_used_is_not_a_copy_of_search_rounds():
    """`metrics["rounds_used"] = rounds` made round_budget_respected a
    threshold on the behavioral component, so the instruction constraint
    answered to behavior_shift and not to instruction_gap.
    """
    records = generate_cohort(CohortConfig(num_users=20, sessions_per_user=10, seed=22))

    differing = sum(
        record.metrics["rounds_used"] != record.metrics["search_rounds"]
        for record in records
    )

    assert differing > 0
    assert all(
        record.metrics["rounds_used"] >= record.metrics["search_rounds"]
        for record in records
    )


def test_the_round_budget_responds_to_the_instruction_knob_alone():
    """With the behavioral knob switched off entirely, the instruction knob
    must still move budget compliance -- that is what makes
    round_budget_respected an instruction-following measurement.
    """

    def respected(records, policy: str) -> float:
        rows = [r for r in records if r.policy == policy]
        return sum(r.metrics["rounds_used"] <= 5 for r in rows) / len(rows)

    with_gap = generate_cohort(
        CohortConfig(
            num_users=40,
            sessions_per_user=10,
            behavior_shift=0.0,
            instruction_gap=0.4,
            seed=23,
        )
    )
    # Averaged over seeds: one null cohort of this size carries a standard
    # error near 0.035 on this difference, so a single draw cannot
    # distinguish "no effect" from "small effect" at a useful tolerance.
    null_gaps = [
        respected(records, "trained") - respected(records, "baseline")
        for seed in range(5)
        for records in [
            generate_cohort(
                null_cohort_config(
                    CohortConfig(num_users=40, sessions_per_user=10, seed=23 + seed)
                )
            )
        ]
    ]

    assert respected(with_gap, "trained") > respected(with_gap, "baseline") + 0.05
    assert abs(sum(null_gaps) / len(null_gaps)) < 0.03, null_gaps


def test_both_default_tool_names_are_exercised():
    """`allowed_tools` defaults to {"search", "fetch"}; a generator that only
    ever emits searches leaves half that constant untested.
    """
    records = generate_cohort(CohortConfig(num_users=10, sessions_per_user=6, seed=24))

    names = {
        call.split('"name": "')[1].split('"')[0]
        for record in records
        for call in record.tool_calls
        if '"name": "' in call
    }

    assert names == {"search", "fetch"}


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
