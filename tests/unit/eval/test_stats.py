"""Contracts for the resampling primitives.

These are checked against answers computable by hand or by construction, not
against a reference implementation, because there is no reference here to
disagree with.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.model.post_training.eval.stats import (
    benjamini_hochberg,
    cliffs_delta,
    cluster_bootstrap_ci,
    paired_permutation_p,
    roc_auc,
)


def test_auc_is_one_when_every_positive_outranks_every_negative():
    assert roc_auc([0.1, 0.2, 0.9, 1.0], [False, False, True, True]) == 1.0


def test_auc_is_zero_when_the_ranking_is_exactly_backwards():
    assert roc_auc([0.9, 1.0, 0.1, 0.2], [False, False, True, True]) == 0.0


def test_auc_gives_half_credit_for_ties():
    # One positive and one negative sharing a score is a coin flip.
    assert roc_auc([1.0, 1.0], [True, False]) == 0.5


def test_auc_needs_both_outcomes_present():
    with pytest.raises(ValueError, match="both"):
        roc_auc([0.1, 0.2], [True, True])


def test_cliffs_delta_is_plus_one_for_disjoint_ordered_samples():
    assert cliffs_delta([10.0, 11.0], [1.0, 2.0]) == 1.0


def test_cliffs_delta_is_minus_one_when_reversed():
    assert cliffs_delta([1.0, 2.0], [10.0, 11.0]) == -1.0


def test_cliffs_delta_is_zero_for_identical_samples():
    assert cliffs_delta([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_benjamini_hochberg_matches_a_hand_worked_example():
    # p * n / rank, then enforced monotone from the largest down.
    # raw:  0.01, 0.02, 0.03, 0.04     n = 4
    # p*n/i: 0.04, 0.04, 0.04, 0.04
    assert benjamini_hochberg([0.01, 0.02, 0.03, 0.04]) == pytest.approx(
        [0.04, 0.04, 0.04, 0.04]
    )


def test_benjamini_hochberg_preserves_input_order():
    adjusted = benjamini_hochberg([0.04, 0.01])
    assert adjusted[0] > adjusted[1]


def test_benjamini_hochberg_never_exceeds_one():
    assert all(value <= 1.0 for value in benjamini_hochberg([0.9, 0.95, 0.99]))


def test_benjamini_hochberg_pulls_a_non_monotone_scaled_value_down_to_its_neighbour():
    # raw:    0.01,  0.5,   0.02,  0.6     n = 4, ranks by raw value: 1, 3, 2, 4
    # p*n/i:  0.04,  0.667, 0.04,  0.6
    # index 1's scaled value (0.667) exceeds index 3's (0.6) despite having a
    # smaller rank, so enforcement must pull it down to 0.6.
    assert benjamini_hochberg([0.01, 0.5, 0.02, 0.6]) == pytest.approx(
        [0.04, 0.6, 0.04, 0.6]
    )


def test_bootstrap_ci_brackets_the_point_estimate():
    units = [float(x) for x in range(100)]
    point, low, high = cluster_bootstrap_ci(
        units, lambda sample: float(np.mean(sample)), resamples=500, seed=1
    )

    assert low < point < high
    assert point == pytest.approx(49.5)


def test_bootstrap_ci_covers_a_known_mean_at_about_the_nominal_rate():
    rng = np.random.default_rng(7)
    covered = 0
    trials = 200
    for trial in range(trials):
        sample = rng.normal(loc=5.0, scale=1.0, size=60).tolist()
        _, low, high = cluster_bootstrap_ci(
            sample, lambda s: float(np.mean(s)), resamples=300, seed=trial
        )
        covered += int(low <= 5.0 <= high)

    assert 0.85 <= covered / trials <= 1.0


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    units = [1.0, 5.0, 2.0, 8.0]
    first = cluster_bootstrap_ci(units, lambda s: float(np.mean(s)), seed=3)
    second = cluster_bootstrap_ci(units, lambda s: float(np.mean(s)), seed=3)

    assert first == second


def test_bootstrap_resample_size_matches_the_unit_count():
    # A statistic that reports its own sample size pins the resample draw
    # size directly: every replicate must be built from as many units as the
    # input has, not a fraction of it.
    units = list(range(37))
    point, low, high = cluster_bootstrap_ci(
        units, lambda sample: float(len(sample)), resamples=50, seed=2
    )

    assert point == len(units)
    assert low == pytest.approx(point)
    assert high == pytest.approx(point)


def test_permutation_p_is_small_for_a_consistent_positive_shift():
    assert paired_permutation_p([1.0] * 20, resamples=500, seed=0) < 0.01


def test_permutation_p_is_large_when_differences_are_symmetric():
    assert paired_permutation_p([1.0, -1.0] * 10, resamples=500, seed=0) > 0.2


def test_permutation_p_is_never_zero():
    assert paired_permutation_p([5.0] * 30, resamples=100, seed=0) > 0.0


def test_permutation_p_is_uniform_under_exchangeability():
    """The property that makes a null-cohort result trustworthy."""
    rng = np.random.default_rng(11)
    pvalues = [
        paired_permutation_p(
            rng.normal(size=25).tolist(), resamples=200, seed=int(trial)
        )
        for trial in range(300)
    ]

    # A uniform p-value rejects at about the nominal rate.
    assert 0.01 <= sum(p < 0.05 for p in pvalues) / len(pvalues) <= 0.12


def test_permutation_alternative_less_mirrors_greater():
    greater = paired_permutation_p([1.0] * 15, seed=0, alternative="greater")
    less = paired_permutation_p([-1.0] * 15, seed=0, alternative="less")

    assert greater == pytest.approx(less)


def test_permutation_two_sided_is_one_when_the_observed_difference_is_zero():
    # A symmetric set of differences has observed mean 0.0 exactly, and
    # |permuted| >= |0.0| holds for every possible permutation, so the
    # two-sided p-value must sit at its ceiling of 1.0.
    p = paired_permutation_p(
        [1.0, -1.0] * 10, resamples=500, seed=0, alternative="two-sided"
    )
    assert p == 1.0


def test_permutation_two_sided_is_at_least_the_matching_one_sided_p():
    # {permuted >= observed} is a subset of {|permuted| >= |observed|} when
    # observed > 0, so for the same data and seed two-sided can never be more
    # significant (smaller) than the matching "greater" one-sided test.
    differences = [3.0, 1.0]
    greater = paired_permutation_p(
        differences, resamples=500, seed=0, alternative="greater"
    )
    two_sided = paired_permutation_p(
        differences, resamples=500, seed=0, alternative="two-sided"
    )

    assert two_sided > greater
    assert two_sided < 1.0


def test_permutation_rejects_an_unknown_alternative():
    with pytest.raises(ValueError, match="alternative"):
        paired_permutation_p([1.0], alternative="sideways")
