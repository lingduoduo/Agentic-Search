"""Resampling primitives for user-clustered evaluation.

Deliberately ignorant of rewards, agents and policies: these functions take
grouped numbers and return numbers, which is what makes them testable against
answers that can be worked out by hand.

Everything here is non-parametric. The user counts this harness works with are
small and the quantities (action counts, compliance rates) are skewed and
bounded, so a t-test's normality assumption would be doing real work with no
justification. Permutation and bootstrap assume only exchangeability.

numpy only -- no scipy, and no torch: this package must import in the CI job
that installs neither.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

_ALTERNATIVES = ("greater", "less", "two-sided")


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Probability a random positive outranks a random negative.

    Rank-based (Mann-Whitney), so ties score exactly half credit rather than
    being broken arbitrarily by sort order.

    Raises:
        ValueError: if *labels* is not mixed -- AUC is undefined without both
            a positive and a negative to compare.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length.")
    positives = int(sum(1 for label in labels if label))
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUC needs both outcomes present in labels.")

    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    position = 0
    while position < len(order):
        end = position
        while (
            end + 1 < len(order) and scores[order[end + 1]] == scores[order[position]]
        ):
            end += 1
        average_rank = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[order[index]] = average_rank
        position = end + 1

    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Ordinal effect size in ``[-1, 1]``: how often *a* exceeds *b*.

    Chosen over a standardized mean difference because it makes no assumption
    about spread, which matters for bounded rates and long-tailed action counts.
    """
    if not a or not b:
        raise ValueError("cliffs_delta needs two non-empty samples.")
    left = np.asarray(a, dtype=float)[:, None]
    right = np.asarray(b, dtype=float)[None, :]
    return float((np.sum(left > right) - np.sum(left < right)) / (len(a) * len(b)))


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, in the caller's input order."""
    count = len(pvalues)
    if count == 0:
        return []
    order = sorted(range(count), key=lambda index: pvalues[index])
    adjusted = [0.0] * count
    running_min = 1.0
    for rank in range(count - 1, -1, -1):
        index = order[rank]
        scaled = pvalues[index] * count / (rank + 1)
        running_min = min(running_min, scaled)
        adjusted[index] = min(1.0, running_min)
    return adjusted


def cluster_bootstrap_ci(
    units: Sequence[Any],
    statistic: Callable[[Sequence[Any]], float],
    *,
    resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI, resampling whole *units*.

    A unit is one user with everything they contributed. Resampling users
    rather than their individual sessions is what keeps the interval honest:
    sessions from one user are correlated, and treating them as independent
    draws would shrink the interval toward a width the data does not support.

    Returns:
        ``(point_estimate, lower, upper)``.
    """
    if not units:
        raise ValueError("cluster_bootstrap_ci needs at least one unit.")
    if resamples < 1:
        raise ValueError("resamples must be at least 1.")

    point = float(statistic(units))
    rng = np.random.default_rng(seed)
    count = len(units)
    replicates = np.empty(resamples, dtype=float)
    for draw in range(resamples):
        picks = rng.integers(0, count, size=count)
        replicates[draw] = statistic([units[index] for index in picks])

    low = float(np.percentile(replicates, 100 * alpha / 2))
    high = float(np.percentile(replicates, 100 * (1 - alpha / 2)))
    return point, low, high


def paired_permutation_p(
    differences: Sequence[float],
    *,
    resamples: int = 2000,
    seed: int = 0,
    alternative: str = "greater",
) -> float:
    """Exact-form paired permutation test by sign flipping.

    Each element is one user's (trained - baseline) difference. Under the null
    the policy label is arbitrary within a user, so flipping a difference's sign
    is exactly relabelling that user -- which is why one sign-flip test serves
    both the behavioral and instruction-following comparisons.

    The p-value uses the ``(1 + count) / (1 + resamples)`` correction, so it is
    never 0: the observed arrangement is itself one of the arrangements.
    """
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_ALTERNATIVES}.")
    if not differences:
        raise ValueError("paired_permutation_p needs at least one difference.")

    values = np.asarray(differences, dtype=float)
    observed = float(np.mean(values))
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(resamples, values.size)) * 2 - 1
    permuted = (signs * values).mean(axis=1)

    if alternative == "greater":
        extreme = int(np.sum(permuted >= observed))
    elif alternative == "less":
        extreme = int(np.sum(permuted <= observed))
    else:
        extreme = int(np.sum(np.abs(permuted) >= abs(observed)))
    return (1.0 + extreme) / (1.0 + resamples)
