"""Evaluate a policy on users held out from training.

Everything here obeys one rule: the unit of independence is the user. A user
contributes many sessions and those sessions are correlated, so a per-session
analysis reports an ``n`` the data does not have and finds significance in
noise. Metrics therefore collapse to one number per user before any test runs,
and every resample draws users rather than sessions.

The comparisons are paired twice over: by user, and inside a user by
``prompt_id``. Both policies answer the same prompts, so pairing on the prompt
cancels task difficulty, and pairing on the user cancels their idiosyncratic
behaviour. The null hypothesis is that the policy label attached to each
user's difference is arbitrary.
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

DEFAULT_ALLOWED_TOOLS = frozenset({"search", "fetch"})
DEFAULT_MAX_SEARCH_ROUNDS = 5

# Below this many held-out users with a defined AUC, the alignment claim is not
# made at all. The bound is empirical: on null cohorts (alignment planted at
# zero), the rate at which the bootstrap's lower bound still cleared 0.5 --
# a false positive, against a 0.025 nominal one-sided level -- was
#
#   users with a defined AUC:  1-2    3-5    6-8    9-11   12-17   18+
#   null false-positive rate:  0.287  0.056  0.037  0.029  0.027   0.018
#
# (1385 null replications, sessions_per_user=10, resamples=200). The interval
# is anti-conservative at small n and degenerate at n=1: every resample of a
# single unit returns that unit, so the interval collapses to zero width and
# its lower bound clears 0.5 unconditionally. Printing
# "AUC 1.000 [1.000, 1.000] over 1 users" is worse than printing nothing, so
# below this threshold the harness prints nothing and counts no rejection.
MIN_ALIGNMENT_USERS = 12


@dataclass(frozen=True)
class AlignmentResult:
    auc: float
    ci_low: float
    ci_high: float
    n_users: int
    n_excluded: int
    min_users: int = MIN_ALIGNMENT_USERS

    @property
    def sufficient(self) -> bool:
        """Whether enough users remain for the interval to mean anything."""
        return self.n_users >= self.min_users


@dataclass(frozen=True)
class ComparisonResult:
    name: str
    trained_mean: float
    baseline_mean: float
    effect: float
    p_value: float
    p_adjusted: float
    n_users: int = 0
    mean_difference: float = 0.0
    diff_ci_low: float = 0.0
    diff_ci_high: float = 0.0


@dataclass(frozen=True)
class UnseenUserReport:
    n_holdout_users: int
    alignment: AlignmentResult
    behavior: tuple[ComparisonResult, ...]
    instruction: tuple[ComparisonResult, ...]
    provenance: str
    baseline_label: str = "baseline"


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
    grouped: dict[str, list[EvalRecord]], *, resamples: int, alpha: float, seed: int
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
        per_user_auc, _mean, resamples=resamples, alpha=alpha, seed=seed
    )
    return AlignmentResult(point, low, high, len(per_user_auc), excluded)


def _paired_means(
    grouped: dict[str, list[EvalRecord]],
    value_of: Callable[[EvalRecord], float],
) -> tuple[list[float], list[float]]:
    """One trained mean and one baseline mean per user, aligned by index.

    Within a user, only prompts both arms answered are counted. On today's
    cohort every arm answers every prompt, so the intersection is a no-op --
    but on real rollouts with unequal coverage, averaging over each arm's own
    prompt set would silently compare different tasks and still call the
    comparison paired.
    """
    trained_means: list[float] = []
    baseline_means: list[float] = []
    for rows in grouped.values():
        by_policy: dict[str, dict[str, list[float]]] = {"trained": {}, "baseline": {}}
        for row in rows:
            if row.policy in by_policy:
                by_policy[row.policy].setdefault(row.prompt_id, []).append(
                    value_of(row)
                )
        shared = by_policy["trained"].keys() & by_policy["baseline"].keys()
        if not shared:
            continue
        trained_means.append(
            _mean([_mean(by_policy["trained"][key]) for key in sorted(shared)])
        )
        baseline_means.append(
            _mean([_mean(by_policy["baseline"][key]) for key in sorted(shared)])
        )
    return trained_means, baseline_means


def _compare(
    name: str,
    trained: list[float],
    baseline: list[float],
    *,
    seed: int,
    resamples: int,
    alpha: float,
    alternative: str,
) -> ComparisonResult:
    differences = [t - b for t, b in zip(trained, baseline)]
    p_value = paired_permutation_p(
        differences, resamples=resamples, seed=seed, alternative=alternative
    )
    # The spec's effect size for the instruction comparison: the mean paired
    # difference with a cluster bootstrap CI. Cliff's delta answers a different
    # question (how often one arm exceeds the other) and carries no interval,
    # so both are reported rather than one standing in for the other.
    difference, diff_low, diff_high = cluster_bootstrap_ci(
        differences, _mean, resamples=resamples, alpha=alpha, seed=seed
    )
    return ComparisonResult(
        name=name,
        trained_mean=_mean(trained),
        baseline_mean=_mean(baseline),
        effect=cliffs_delta(trained, baseline),
        p_value=p_value,
        p_adjusted=p_value,  # replaced after the whole family is collected
        n_users=len(differences),
        mean_difference=difference,
        diff_ci_low=diff_low,
        diff_ci_high=diff_high,
    )


def evaluate_unseen_users(
    records: Sequence[EvalRecord],
    *,
    provenance: str,
    holdout_fraction: float = 0.3,
    seed: int = 0,
    resamples: int = 2000,
    alpha: float = 0.05,
    allowed_tools: frozenset[str] = DEFAULT_ALLOWED_TOOLS,
    max_search_rounds: int = DEFAULT_MAX_SEARCH_ROUNDS,
    baseline_label: str = "baseline",
) -> UnseenUserReport:
    """Run every measurement on the held-out users only.

    ``provenance`` is required and must be non-empty: a number from this
    harness is meaningless without the population that produced it, and a
    report that renders "Provenance: unspecified" invites exactly the quotation
    this design exists to prevent.
    """
    if not records:
        raise ValueError("records must not be empty.")
    if not provenance.strip():
        raise ValueError("provenance must be a non-empty description of the cohort.")

    _, holdout = split_users(
        {record.user_id for record in records},
        holdout_fraction=holdout_fraction,
        seed=seed,
    )
    frame = [record for record in records if record.user_id in holdout]
    if not frame:
        raise ValueError("records produced an empty holdout frame.")
    grouped = _per_user(frame)

    alignment = _alignment(grouped, resamples=resamples, alpha=alpha, seed=seed)

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
                alpha=alpha,
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
                alpha=alpha,
                alternative="greater",
            )
        )

    family = [*behavior, *instruction]
    adjusted = benjamini_hochberg([result.p_value for result in family])
    corrected = [
        replace(result, p_adjusted=value) for result, value in zip(family, adjusted)
    ]

    return UnseenUserReport(
        n_holdout_users=len(holdout),
        alignment=alignment,
        behavior=tuple(corrected[: len(behavior)]),
        instruction=tuple(corrected[len(behavior) :]),
        provenance=provenance,
        baseline_label=baseline_label,
    )


def _comparison_rows(results: Sequence[ComparisonResult]) -> list[str]:
    return [
        f"| `{result.name}` | {result.n_users} | {result.trained_mean:.3f} "
        f"| {result.baseline_mean:.3f} | {result.mean_difference:+.3f} "
        f"[{result.diff_ci_low:+.3f}, {result.diff_ci_high:+.3f}] "
        f"| {result.effect:+.3f} "
        f"| {result.p_value:.4f} | {result.p_adjusted:.4f} |"
        for result in results
    ]


_TABLE_HEADER = (
    "| name | n users | trained | baseline | mean paired diff [CI] "
    "| effect (Cliff's d) | p | p (BH) |\n"
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
)


def format_report(report: UnseenUserReport) -> str:
    """Render the report, provenance first.

    Provenance leads because a number from this harness must never be quoted
    without the population that produced it -- which includes the size of the
    effect that was planted in it.
    """
    if report.alignment.sufficient:
        alignment_line = (
            f"AUC {report.alignment.auc:.3f} "
            f"[{report.alignment.ci_low:.3f}, {report.alignment.ci_high:.3f}] "
            f"over {report.alignment.n_users} users "
            f"({report.alignment.n_excluded} excluded: no outcome variation)"
        )
    else:
        alignment_line = (
            f"Undefined: {report.alignment.n_users} users with a defined AUC "
            f"({report.alignment.n_excluded} excluded: no outcome variation), "
            f"below the {report.alignment.min_users} this harness requires "
            "before it will state an interval."
        )

    lines = [
        "# Unseen-user evaluation",
        "",
        f"Provenance: {report.provenance}",
        f"Measured on {report.n_holdout_users} held-out users "
        "(unit of independence: user, not session).",
        "",
        "## Conversion alignment",
        alignment_line,
        "",
        "## Behavioral separation",
        _TABLE_HEADER,
        *_comparison_rows(report.behavior),
        "",
        f"## Instruction following (vs {report.baseline_label})",
        _TABLE_HEADER,
        *_comparison_rows(report.instruction),
        "",
    ]
    return "\n".join(lines)


def achieved_power(
    config: CohortConfig,
    *,
    replications: int = 200,
    alpha: float = 0.05,
    resamples: int = 200,
    seed: int = 0,
    holdout_fraction: float = 0.3,
    allowed_tools: frozenset[str] = DEFAULT_ALLOWED_TOOLS,
    max_search_rounds: int = DEFAULT_MAX_SEARCH_ROUNDS,
) -> dict[str, float]:
    """Fraction of freshly generated cohorts on which each measurement fires.

    This is what turns "statistically significant" into a statement about the
    pipeline rather than about one lucky draw. Each replication regenerates the
    cohort from a new seed; reusing one cohort would report the same result N
    times and call it power.

    A small cohort can hash every user into the training half, leaving nothing
    to evaluate. Such a replication is skipped rather than raised, and the
    fraction skipped is reported as ``skipped_replication_rate`` so a power
    figure computed over a handful of surviving draws cannot pass for one
    computed over all of them. Rates are over the replications that ran.

    Only meaningful for a generator whose ground truth is known -- which is why
    it takes a config rather than records.
    """
    if replications < 1:
        raise ValueError("replications must be at least 1.")

    counts: dict[str, int] = {"alignment": 0}
    for name in (*BEHAVIOR_COMPONENTS, *CONSTRAINT_NAMES):
        counts[name] = 0

    completed = 0
    for index in range(replications):
        records = generate_cohort(replace(config, seed=seed + index))
        _, holdout = split_users(
            {record.user_id for record in records},
            holdout_fraction=holdout_fraction,
            seed=seed + index,
        )
        if not holdout:
            continue
        report = evaluate_unseen_users(
            records,
            provenance="power replication",
            holdout_fraction=holdout_fraction,
            seed=seed + index,
            resamples=resamples,
            alpha=alpha,
            allowed_tools=allowed_tools,
            max_search_rounds=max_search_rounds,
        )
        completed += 1
        counts["alignment"] += int(
            report.alignment.sufficient and report.alignment.ci_low > 0.5
        )
        for result in (*report.behavior, *report.instruction):
            counts[result.name] += int(result.p_adjusted < alpha)

    if completed == 0:
        raise ValueError("every replication produced an empty holdout frame.")

    power = {name: value / completed for name, value in counts.items()}
    power["skipped_replication_rate"] = (replications - completed) / replications
    return power
