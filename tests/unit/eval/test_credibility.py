"""The two tests that make every other number in this harness meaningful.

If a null cohort produces significance, the harness manufactures p-values and
nothing it prints can be trusted. If ignoring clustering does not inflate
significance, the design's central claim is unfounded and the complexity of
clustering buys nothing.
"""

from __future__ import annotations

import pytest

from src.model.post_training.eval.cohort import (
    CohortConfig,
    generate_cohort,
    null_cohort_config,
)
from src.model.post_training.eval.instruction_following import CONSTRAINT_NAMES
from src.model.post_training.eval.stats import paired_permutation_p
from src.model.post_training.eval.unseen_users import (
    BEHAVIOR_COMPONENTS,
    achieved_power,
    evaluate_unseen_users,
)


def test_a_null_cohort_does_not_produce_significance():
    """No planted effect: rejections must stay near the nominal rate."""
    config = null_cohort_config(
        CohortConfig(num_users=60, sessions_per_user=10, seed=100)
    )

    power = achieved_power(config, replications=60, resamples=100, seed=100)

    assert power["alignment"] <= 0.20
    for name, rate in power.items():
        assert rate <= 0.25, f"{name} rejected {rate:.2f} of the time under the null"


def test_a_planted_effect_is_detected_far_more_often_than_the_null():
    planted = CohortConfig(
        num_users=60, sessions_per_user=10, behavior_shift=2.5, seed=101
    )
    null = null_cohort_config(planted)

    detected = achieved_power(planted, replications=40, resamples=100, seed=101)
    baseline = achieved_power(null, replications=40, resamples=100, seed=101)

    assert detected["search_rounds"] > baseline["search_rounds"] + 0.4


def test_every_measurement_achieves_nonzero_power_under_a_planted_effect():
    """Guard against a component that can never reject.

    A measurement stuck at 0.0 power under both the null *and* a planted
    effect is indistinguishable from a metric no knob actually drives -- the
    null test alone cannot tell "genuinely quiet" from "structurally
    incapable of ever rejecting". This is the direct check: with every knob
    on, every behavioral component and every instruction constraint must
    reject at least sometimes.
    """
    planted = CohortConfig(
        num_users=60, sessions_per_user=10, behavior_shift=2.5, seed=101
    )

    power = achieved_power(planted, replications=40, resamples=100, seed=101)

    for name in (*BEHAVIOR_COMPONENTS, *CONSTRAINT_NAMES):
        assert power[name] > 0.0, f"{name} never rejected under a planted effect"


def test_ignoring_clustering_inflates_significance():
    """The bug the whole design exists to avoid, demonstrated.

    Sessions within a user share a latent offset. Treating each session as an
    independent unit finds a difference between two label groups that a
    user-level analysis correctly does not.
    """
    import numpy as np

    rng = np.random.default_rng(202)
    clustered_rejections = 0
    naive_rejections = 0
    trials = 120

    for trial in range(trials):
        # 12 users, 10 correlated sessions each, no real policy effect.
        offsets = rng.normal(0.0, 3.0, size=12)
        per_session = [
            offsets[user] + rng.normal(0.0, 0.5)
            for user in range(12)
            for _ in range(10)
        ]
        per_user = [
            float(np.mean(per_session[user * 10 : (user + 1) * 10]))
            for user in range(12)
        ]

        naive_p = paired_permutation_p(
            per_session, resamples=100, seed=trial, alternative="two-sided"
        )
        clustered_p = paired_permutation_p(
            per_user, resamples=100, seed=trial, alternative="two-sided"
        )
        naive_rejections += int(naive_p < 0.05)
        clustered_rejections += int(clustered_p < 0.05)

    assert naive_rejections > clustered_rejections


def test_power_reports_every_measurement():
    config = CohortConfig(num_users=30, sessions_per_user=6, seed=103)

    power = achieved_power(config, replications=5, resamples=50, seed=103)

    assert "alignment" in power
    assert "search_rounds" in power
    assert "answer_tag_present" in power
    assert all(0.0 <= rate <= 1.0 for rate in power.values())


def test_power_is_deterministic():
    config = CohortConfig(num_users=20, sessions_per_user=6, seed=104)

    first = achieved_power(config, replications=5, resamples=50, seed=104)
    second = achieved_power(config, replications=5, resamples=50, seed=104)

    assert first == second


def test_each_replication_uses_a_fresh_cohort():
    """Otherwise 'power' is one lucky draw counted N times.

    Pins the exact contract: replication *i* uses ``seed + i``. Reproduce two
    replications by hand and require the reported rate to equal their mean.
    """
    from dataclasses import replace

    config = CohortConfig(num_users=25, sessions_per_user=6, seed=105)

    by_hand = []
    for index in range(2):
        records = generate_cohort(replace(config, seed=105 + index))
        report = evaluate_unseen_users(
            records, seed=105 + index, resamples=100, provenance="test cohort"
        )
        rounds = next(r for r in report.behavior if r.name == "search_rounds")
        by_hand.append(int(rounds.p_adjusted < 0.05))

    power = achieved_power(config, replications=2, resamples=100, seed=105)

    assert power["search_rounds"] == sum(by_hand) / 2
    # And the two replications must not be the same cohort.
    assert generate_cohort(replace(config, seed=105)) != generate_cohort(
        replace(config, seed=106)
    )


def test_power_skips_replications_whose_split_leaves_no_holdout():
    """A small cohort can hash every user into the training half. That used
    to raise ValueError("records produced an empty holdout frame") out of
    achieved_power and kill the run; it must be skipped and counted instead.
    """
    config = CohortConfig(num_users=6, sessions_per_user=6, seed=200)

    power = achieved_power(config, replications=40, resamples=50, seed=200)

    assert power["skipped_replication_rate"] > 0.0
    assert all(0.0 <= rate <= 1.0 for rate in power.values())


def test_power_raises_only_when_no_replication_survives():
    config = CohortConfig(num_users=1, sessions_per_user=4, seed=201)

    with pytest.raises(ValueError, match="empty holdout"):
        achieved_power(
            config, replications=5, resamples=20, seed=201, holdout_fraction=0.01
        )


def test_power_makes_no_alignment_claim_below_the_minimum_user_count():
    """Strong alignment, too few held-out users to support the interval: the
    harness must report no rejections rather than a claim it cannot back.
    """
    config = CohortConfig(num_users=15, sessions_per_user=10, alignment=4.0, seed=202)

    power = achieved_power(config, replications=20, resamples=50, seed=202)

    assert power["alignment"] == 0.0
