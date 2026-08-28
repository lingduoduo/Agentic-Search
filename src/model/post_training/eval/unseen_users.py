"""Evaluate a policy on users held out from training.

Everything here obeys one rule: the unit of independence is the user. A user
contributes many sessions and those sessions are correlated, so a per-session
analysis reports an ``n`` the data does not have and finds significance in
noise. Metrics therefore collapse to one number per user before any test runs,
and every resample draws users rather than sessions.

The comparisons are paired by user: both policies answer the same prompts, so a
user's own difference cancels their idiosyncratic difficulty, and the null
hypothesis is that the policy label attached to each difference is arbitrary.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

from .cohort import CohortConfig, EvalRecord, generate_cohort
from .instruction_following import CONSTRAINT_NAMES, check_constraints
from .stats import (
    benjamini_hochberg,
    cliffs_delta,
    cluster_bootstrap_ci,
    paired_permutation_p,
    roc_auc,
)

BEHAVIOR_COMPONENTS: tuple[str, ...] = (
    "search_rounds",
    "web_searches",
    "vdb_searches",
    "rerank_calls",
    "repeated_search_queries",
)


@dataclass(frozen=True)
class AlignmentResult:
    auc: float
    ci_low: float
    ci_high: float
    n_users: int
    n_excluded: int


@dataclass(frozen=True)
class ComparisonResult:
    name: str
    trained_mean: float
    baseline_mean: float
    effect: float
    p_value: float
    p_adjusted: float


@dataclass(frozen=True)
class UnseenUserReport:
    n_holdout_users: int
    alignment: AlignmentResult
    behavior: tuple[ComparisonResult, ...]
    instruction: tuple[ComparisonResult, ...]
    provenance: str


def split_users(
    user_ids: Iterable[str],
    *,
    holdout_fraction: float = 0.3,
    seed: int = 0,
) -> tuple[frozenset[str], frozenset[str]]:
    """Partition users deterministically by hash.

    Hashing rather than shuffling makes the split independent of iteration
    order, so the same user lands on the same side no matter how the records
    reached this function.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be strictly between 0 and 1.")

    train: set[str] = set()
    holdout: set[str] = set()
    for user_id in set(user_ids):
        digest = hashlib.sha256(f"{seed}:{user_id}".encode()).digest()
        position = int.from_bytes(digest[:8], "big") / float(1 << 64)
        (holdout if position < holdout_fraction else train).add(user_id)
    return frozenset(train), frozenset(holdout)


def _per_user(records: Sequence[EvalRecord]) -> dict[str, list[EvalRecord]]:
    grouped: dict[str, list[EvalRecord]] = {}
    for record in records:
        grouped.setdefault(record.user_id, []).append(record)
    return grouped


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _alignment(
    grouped: dict[str, list[EvalRecord]], *, resamples: int, seed: int
) -> AlignmentResult:
    """Per-user AUC of reward against conversion, averaged over users.

    A user whose sessions all converted (or none did) has no ranking to score,
    so they are dropped -- and counted, because a cohort that is mostly
    degenerate must not read as a clean result.
    """
    per_user_auc: list[float] = []
    excluded = 0
    for rows in grouped.values():
        trained = [row for row in rows if row.policy == "trained"]
        labels = [row.converted for row in trained]
        if len(set(labels)) < 2:
            excluded += 1
            continue
        per_user_auc.append(roc_auc([row.reward for row in trained], labels))

    if not per_user_auc:
        return AlignmentResult(0.5, 0.5, 0.5, 0, excluded)

    point, low, high = cluster_bootstrap_ci(
        per_user_auc, _mean, resamples=resamples, seed=seed
    )
    return AlignmentResult(point, low, high, len(per_user_auc), excluded)


def _paired_means(
    grouped: dict[str, list[EvalRecord]],
    value_of: Callable[[EvalRecord], float],
) -> tuple[list[float], list[float]]:
    """One trained mean and one baseline mean per user, aligned by index."""
    trained_means: list[float] = []
    baseline_means: list[float] = []
    for rows in grouped.values():
        trained = [value_of(row) for row in rows if row.policy == "trained"]
        baseline = [value_of(row) for row in rows if row.policy == "baseline"]
        if not trained or not baseline:
            continue
        trained_means.append(_mean(trained))
        baseline_means.append(_mean(baseline))
    return trained_means, baseline_means


def _compare(
    name: str,
    trained: list[float],
    baseline: list[float],
    *,
    seed: int,
    resamples: int,
    alternative: str,
) -> ComparisonResult:
    differences = [t - b for t, b in zip(trained, baseline)]
    p_value = paired_permutation_p(
        differences, resamples=resamples, seed=seed, alternative=alternative
    )
    return ComparisonResult(
        name=name,
        trained_mean=_mean(trained),
        baseline_mean=_mean(baseline),
        effect=cliffs_delta(trained, baseline),
        p_value=p_value,
        p_adjusted=p_value,  # replaced after the whole family is collected
    )


def evaluate_unseen_users(
    records: Sequence[EvalRecord],
    *,
    holdout_fraction: float = 0.3,
    seed: int = 0,
    resamples: int = 2000,
    allowed_tools: frozenset[str] = frozenset({"search", "fetch"}),
    max_search_rounds: int = 5,
    provenance: str = "",
) -> UnseenUserReport:
    """Run every measurement on the held-out users only."""
    if not records:
        raise ValueError("records must not be empty.")

    _, holdout = split_users(
        {record.user_id for record in records},
        holdout_fraction=holdout_fraction,
        seed=seed,
    )
    frame = [record for record in records if record.user_id in holdout]
    if not frame:
        raise ValueError("records produced an empty holdout frame.")
    grouped = _per_user(frame)

    alignment = _alignment(grouped, resamples=resamples, seed=seed)

    behavior: list[ComparisonResult] = []
    for index, component in enumerate(BEHAVIOR_COMPONENTS):
        trained, baseline = _paired_means(
            grouped, lambda row, key=component: row.metrics.get(key, 0.0)
        )
        behavior.append(
            _compare(
                component,
                trained,
                baseline,
                seed=seed + index,
                resamples=resamples,
                alternative="two-sided",
            )
        )

    instruction: list[ComparisonResult] = []
    for index, constraint in enumerate(CONSTRAINT_NAMES):

        def value_of(row: EvalRecord, key: str = constraint) -> float:
            verdicts = check_constraints(
                row, allowed_tools=allowed_tools, max_search_rounds=max_search_rounds
            )
            return float(verdicts[key])

        trained, baseline = _paired_means(grouped, value_of)
        instruction.append(
            _compare(
                constraint,
                trained,
                baseline,
                seed=seed + 100 + index,
                resamples=resamples,
                alternative="greater",
            )
        )

    family = [*behavior, *instruction]
    adjusted = benjamini_hochberg([result.p_value for result in family])
    corrected = [
        ComparisonResult(
            result.name,
            result.trained_mean,
            result.baseline_mean,
            result.effect,
            result.p_value,
            value,
        )
        for result, value in zip(family, adjusted)
    ]

    return UnseenUserReport(
        n_holdout_users=len(holdout),
        alignment=alignment,
        behavior=tuple(corrected[: len(behavior)]),
        instruction=tuple(corrected[len(behavior) :]),
        provenance=provenance,
    )


def format_report(report: UnseenUserReport) -> str:
    """Render the report, provenance first.

    Provenance leads because a number from this harness must never be quoted
    without the population that produced it.
    """
    lines = [
        "# Unseen-user evaluation",
        "",
        f"Provenance: {report.provenance or 'unspecified'}",
        f"Measured on {report.n_holdout_users} held-out users "
        "(unit of independence: user, not session).",
        "",
        "## Conversion alignment",
        f"AUC {report.alignment.auc:.3f} "
        f"[{report.alignment.ci_low:.3f}, {report.alignment.ci_high:.3f}] "
        f"over {report.alignment.n_users} users "
        f"({report.alignment.n_excluded} excluded: no outcome variation)",
        "",
        "## Behavioral separation",
        "| component | trained | baseline | effect (Cliff's d) | p | p (BH) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in report.behavior:
        lines.append(
            f"| `{result.name}` | {result.trained_mean:.3f} "
            f"| {result.baseline_mean:.3f} | {result.effect:+.3f} "
            f"| {result.p_value:.4f} | {result.p_adjusted:.4f} |"
        )
    lines += [
        "",
        "## Instruction following (vs larger baseline)",
        "| constraint | trained | baseline | effect (Cliff's d) | p | p (BH) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in report.instruction:
        lines.append(
            f"| `{result.name}` | {result.trained_mean:.3f} "
            f"| {result.baseline_mean:.3f} | {result.effect:+.3f} "
            f"| {result.p_value:.4f} | {result.p_adjusted:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def achieved_power(
    config: CohortConfig,
    *,
    replications: int = 200,
    alpha: float = 0.05,
    resamples: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Fraction of freshly generated cohorts on which each measurement fires.

    This is what turns "statistically significant" into a statement about the
    pipeline rather than about one lucky draw. Each replication regenerates the
    cohort from a new seed; reusing one cohort would report the same result N
    times and call it power.

    Only meaningful for a generator whose ground truth is known -- which is why
    it takes a config rather than records.
    """
    if replications < 1:
        raise ValueError("replications must be at least 1.")

    counts: dict[str, int] = {"alignment": 0}
    for name in (*BEHAVIOR_COMPONENTS, *CONSTRAINT_NAMES):
        counts[name] = 0

    for index in range(replications):
        records = generate_cohort(replace(config, seed=seed + index))
        report = evaluate_unseen_users(records, seed=seed + index, resamples=resamples)
        counts["alignment"] += int(report.alignment.ci_low > 0.5)
        for result in (*report.behavior, *report.instruction):
            counts[result.name] += int(result.p_adjusted < alpha)

    return {name: value / replications for name, value in counts.items()}
