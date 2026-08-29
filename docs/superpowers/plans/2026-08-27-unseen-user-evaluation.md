# Unseen-User Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an evaluation harness that measures conversion alignment, behavioral separation, and instruction following on users held out from training, with every statistic clustered by user.

**Architecture:** Four torch-free modules under `src/model/post_training/eval/`. `stats.py` holds generic resampling primitives that know nothing about agents; `cohort.py` generates a simulated user population from a conversion rule the analysis never sees; `instruction_following.py` holds four objective predicates; `unseen_users.py` splits users, runs the measurements, corrects for multiplicity, and renders a report that states its own provenance.

**Tech Stack:** Python 3.10+, numpy 1.26.4, pytest, Ruff. No torch, no transformers, no scipy.

**Spec:** `docs/superpowers/specs/2026-08-27-unseen-user-evaluation-design.md`

## Global Constraints

- **No torch, transformers, or scipy imports anywhere in this work.** `reward.py` and `eval/` are the torch-free half of post-training and the CI unit-test job installs no torch. A torch import silently drops these tests from that gate rather than failing loudly. Verify with the blocker snippet in Task 6.
- The unit of independence is the **user**. Every statistic aggregates to one value per user before any test; every resample draws users. Reported `n` is a user count, never a session count.
- All randomness goes through `numpy.random.default_rng(seed)`. No `random`, no unseeded `np.random`. Identical seed ⇒ identical output.
- p-values use the `(1 + count) / (1 + resamples)` correction; a permutation p-value is never 0.
- Every new test must fail when the behavior it describes is removed. Verify by deleting the behavior, watching it go red, then restoring.
- Follow `eval/action_eval.py`'s style: frozen dataclass aggregates, plain dict samples, module docstring that says why the module exists.
- Do not modify reward functions, trainers, or agent loops. This is evaluation only.
- Default sizes: 2000 bootstrap resamples, 2000 permutations, 200 power replications. All configurable.

---

### Task 1: Statistical Primitives

**Files:**
- Create: `src/model/post_training/eval/stats.py`
- Create: `tests/unit/eval/__init__.py`
- Create: `tests/unit/eval/test_stats.py`

**Interfaces:**
- Consumes: nothing from this repo. numpy only.
- Produces:
  - `roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float`
  - `cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float`
  - `benjamini_hochberg(pvalues: Sequence[float]) -> list[float]`
  - `cluster_bootstrap_ci(units: Sequence[Any], statistic: Callable[[Sequence[Any]], float], *, resamples: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]` returning `(point, low, high)`
  - `paired_permutation_p(differences: Sequence[float], *, resamples: int = 2000, seed: int = 0, alternative: str = "greater") -> float`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/eval/__init__.py` as an empty file, then `tests/unit/eval/test_stats.py`:

```python
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


def test_permutation_rejects_an_unknown_alternative():
    with pytest.raises(ValueError, match="alternative"):
        paired_permutation_p([1.0], alternative="sideways")
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest -q tests/unit/eval/test_stats.py`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.model.post_training.eval.stats'`.

- [ ] **Step 3: Implement the primitives**

Create `src/model/post_training/eval/stats.py`:

```python
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
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[position]]:
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
```

- [ ] **Step 4: Run the tests and verify GREEN**

Run: `python -m pytest -q tests/unit/eval/test_stats.py`

Expected: PASS.

- [ ] **Step 5: Mutation-check the uniformity test**

Break the p-value correction and confirm a test notices:

```bash
python - <<'EOF'
from pathlib import Path
p = Path("src/model/post_training/eval/stats.py")
p.write_text(p.read_text().replace(
    "return (1.0 + extreme) / (1.0 + resamples)",
    "return extreme / resamples"))
EOF
python -m pytest -q tests/unit/eval/test_stats.py::test_permutation_p_is_never_zero
git checkout src/model/post_training/eval/stats.py
```

Expected: FAIL under the mutation, then restore.

- [ ] **Step 6: Lint and commit**

```bash
ruff check src/model/post_training/eval/stats.py tests/unit/eval --fix
ruff format src/model/post_training/eval/stats.py tests/unit/eval
python -m pytest -q tests/unit/eval/test_stats.py
git add src/model/post_training/eval/stats.py tests/unit/eval
git commit -m "feat(eval): add user-clustered resampling primitives"
```

---

### Task 2: Simulated Cohort

**Files:**
- Create: `src/model/post_training/eval/cohort.py`
- Create: `tests/unit/eval/test_cohort.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `EvalRecord` frozen dataclass with fields `user_id: str`, `prompt_id: str`, `policy: str`, `reward: float`, `converted: bool`, `response: str`, `metrics: dict[str, float]`, `cited_ids: frozenset[str]`, `tool_calls: tuple[str, ...]`
  - `CohortConfig` frozen dataclass with fields `num_users: int = 40`, `sessions_per_user: int = 12`, `alignment: float = 2.0`, `behavior_shift: float = 1.5`, `instruction_gap: float = 0.25`, `base_compliance: float = 0.6`, `seed: int = 0`
  - `generate_cohort(config: CohortConfig) -> list[EvalRecord]`
  - `null_cohort_config(config: CohortConfig) -> CohortConfig` — same shape, all three effects zeroed

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/eval/test_cohort.py`:

```python
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
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest -q tests/unit/eval/test_cohort.py`

Expected: FAIL with `ModuleNotFoundError` for `...eval.cohort`.

- [ ] **Step 3: Implement the generator**

Create `src/model/post_training/eval/cohort.py`:

```python
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
        # Per-user latent propensity: this is what makes sessions from one user
        # correlated, and therefore what makes clustering necessary.
        affinity = float(rng.normal(0.0, 1.0))

        for session_index in range(config.sessions_per_user):
            prompt_id = f"p{session_index:03d}"
            quality = float(rng.normal(0.0, 1.0))
            convert_p = _sigmoid(config.alignment * quality + affinity)
            converted = bool(rng.random() < convert_p)

            for policy in _POLICIES:
                is_trained = policy == "trained"
                # Reward tracks latent quality; the trained policy reads it
                # slightly better, which is the signal alignment measures.
                reward = quality + float(rng.normal(0.0, 0.5))
                if is_trained:
                    reward += 0.25 * config.alignment * quality

                rounds = max(
                    0.0,
                    float(rng.poisson(4.0))
                    - (config.behavior_shift if is_trained else 0.0),
                )
                compliance_p = config.base_compliance + (
                    config.instruction_gap if is_trained else 0.0
                )
                complies = bool(rng.random() < min(1.0, compliance_p))

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
```

- [ ] **Step 4: Run the tests and verify GREEN**

Run: `python -m pytest -q tests/unit/eval/test_cohort.py`

Expected: PASS.

- [ ] **Step 5: Mutation-check the null config**

```bash
python - <<'EOF'
from pathlib import Path
p = Path("src/model/post_training/eval/cohort.py")
p.write_text(p.read_text().replace(
    "return replace(config, alignment=0.0, behavior_shift=0.0, instruction_gap=0.0)",
    "return replace(config, alignment=0.0, behavior_shift=0.0)"))
EOF
python -m pytest -q tests/unit/eval/test_cohort.py::test_null_config_zeroes_all_three_effects
git checkout src/model/post_training/eval/cohort.py
```

Expected: FAIL under the mutation, then restore.

- [ ] **Step 6: Lint and commit**

```bash
ruff check src/model/post_training/eval/cohort.py tests/unit/eval/test_cohort.py --fix
ruff format src/model/post_training/eval/cohort.py tests/unit/eval/test_cohort.py
python -m pytest -q tests/unit/eval
git add src/model/post_training/eval/cohort.py tests/unit/eval/test_cohort.py
git commit -m "feat(eval): add simulated cohort with a hidden conversion rule"
```

---

### Task 3: Instruction-Following Predicates

**Files:**
- Create: `src/model/post_training/eval/instruction_following.py`
- Create: `tests/unit/eval/test_instruction_following.py`

**Interfaces:**
- Consumes: `EvalRecord` from Task 2.
- Produces:
  - `CONSTRAINT_NAMES: tuple[str, ...]` = `("answer_tag_present", "citations_wellformed", "tool_calls_parseable", "round_budget_respected")`
  - `check_constraints(record: EvalRecord, *, allowed_tools: frozenset[str], max_search_rounds: int) -> dict[str, bool]` — one boolean per name in `CONSTRAINT_NAMES`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/eval/test_instruction_following.py`:

```python
"""Contracts for the four objective instruction-following predicates.

Objective on purpose: no judge, so the headline number does not depend on a
third model and does not move when that model is swapped.
"""

from __future__ import annotations

import pytest

from src.model.post_training.eval.cohort import EvalRecord
from src.model.post_training.eval.instruction_following import (
    CONSTRAINT_NAMES,
    check_constraints,
)

TOOLS = frozenset({"search", "fetch"})


def _record(**overrides) -> EvalRecord:
    base = {
        "user_id": "u0",
        "prompt_id": "p0",
        "policy": "trained",
        "reward": 0.0,
        "converted": False,
        "response": "<answer>grounded [R1Q1D1]</answer>",
        "metrics": {"rounds_used": 2.0},
        "cited_ids": frozenset({"R1Q1D1"}),
        "tool_calls": ('{"name": "search", "arguments": {}}',),
    }
    base.update(overrides)
    return EvalRecord(**base)


def test_a_fully_compliant_record_passes_everything():
    assert check_constraints(
        _record(), allowed_tools=TOOLS, max_search_rounds=5
    ) == dict.fromkeys(CONSTRAINT_NAMES, True)


def test_result_always_reports_every_constraint():
    result = check_constraints(
        _record(response="", tool_calls=()), allowed_tools=TOOLS, max_search_rounds=5
    )

    assert set(result) == set(CONSTRAINT_NAMES)


@pytest.mark.parametrize(
    "response",
    ["no tag at all", "<answer>unclosed", "answer</answer>", "<answer></answer>"],
)
def test_malformed_answer_tags_fail(response: str):
    result = check_constraints(
        _record(response=response), allowed_tools=TOOLS, max_search_rounds=5
    )

    assert result["answer_tag_present"] is False


def test_a_citation_label_that_was_never_retrieved_fails():
    """Parsing is not enough -- the label must resolve to a retrieved doc."""
    result = check_constraints(
        _record(response="<answer>x [R9Q9D9]</answer>", cited_ids=frozenset()),
        allowed_tools=TOOLS,
        max_search_rounds=5,
    )

    assert result["citations_wellformed"] is False


def test_an_answer_citing_nothing_fails_the_citation_constraint():
    result = check_constraints(
        _record(response="<answer>ungrounded</answer>", cited_ids=frozenset()),
        allowed_tools=TOOLS,
        max_search_rounds=5,
    )

    assert result["citations_wellformed"] is False


def test_unparseable_tool_calls_fail():
    result = check_constraints(
        _record(tool_calls=("{not json",)), allowed_tools=TOOLS, max_search_rounds=5
    )

    assert result["tool_calls_parseable"] is False


def test_valid_json_naming_an_unregistered_tool_fails():
    result = check_constraints(
        _record(tool_calls=('{"name": "rm_rf", "arguments": {}}',)),
        allowed_tools=TOOLS,
        max_search_rounds=5,
    )

    assert result["tool_calls_parseable"] is False


def test_a_record_with_no_tool_calls_vacuously_satisfies_the_constraint():
    result = check_constraints(
        _record(tool_calls=()), allowed_tools=TOOLS, max_search_rounds=5
    )

    assert result["tool_calls_parseable"] is True


def test_exceeding_the_round_budget_fails():
    result = check_constraints(
        _record(metrics={"rounds_used": 9.0}), allowed_tools=TOOLS, max_search_rounds=5
    )

    assert result["round_budget_respected"] is False


def test_exactly_the_budget_is_allowed():
    result = check_constraints(
        _record(metrics={"rounds_used": 5.0}), allowed_tools=TOOLS, max_search_rounds=5
    )

    assert result["round_budget_respected"] is True


def test_a_missing_rounds_metric_counts_as_zero_rounds():
    result = check_constraints(
        _record(metrics={}), allowed_tools=TOOLS, max_search_rounds=5
    )

    assert result["round_budget_respected"] is True
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest -q tests/unit/eval/test_instruction_following.py`

Expected: FAIL with `ModuleNotFoundError` for `...eval.instruction_following`.

- [ ] **Step 3: Implement the predicates**

Create `src/model/post_training/eval/instruction_following.py`:

```python
"""Objectively checkable instruction-following constraints.

Each predicate is decidable from the record alone, with no model in the loop.
That is the point: an LLM judge would make the headline comparison depend on a
third model's mood, and re-running the report a month later would produce a
different number for reasons unrelated to either policy.

The citation check deliberately requires more than a well-formed label. A model
that emits ``[R9Q9D9]`` having retrieved nothing has produced syntax, not a
citation, and rewarding it teaches exactly the wrong lesson.
"""

from __future__ import annotations

import json
import re

from .cohort import CohortConfig, EvalRecord, generate_cohort

CONSTRAINT_NAMES: tuple[str, ...] = (
    "answer_tag_present",
    "citations_wellformed",
    "tool_calls_parseable",
    "round_budget_respected",
)

# Mirrors the search-agent contract: <answer>...</answer> with a body.
_ANSWER_TAG_RE = re.compile(r"<answer>\s*\S.*?</answer>", re.DOTALL | re.IGNORECASE)
# Mirrors src/context/search.py's citation labels.
_CITATION_RE = re.compile(r"\[(?:D\d+|R\d+Q\d+D\d+)\]")


def check_constraints(
    record: EvalRecord,
    *,
    allowed_tools: frozenset[str],
    max_search_rounds: int,
) -> dict[str, bool]:
    """Evaluate every constraint for one record.

    Always returns a verdict for every name in :data:`CONSTRAINT_NAMES`; a
    missing key would silently shrink a compliance rate's denominator.
    """
    labels = set(_CITATION_RE.findall(record.response))
    citations_ok = bool(labels) and all(
        label.strip("[]") in record.cited_ids for label in labels
    )

    tool_calls_ok = True
    for payload in record.tool_calls:
        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            tool_calls_ok = False
            break
        if not isinstance(parsed, dict) or parsed.get("name") not in allowed_tools:
            tool_calls_ok = False
            break

    return {
        "answer_tag_present": bool(_ANSWER_TAG_RE.search(record.response)),
        "citations_wellformed": citations_ok,
        "tool_calls_parseable": tool_calls_ok,
        "round_budget_respected": (
            float(record.metrics.get("rounds_used", 0.0)) <= float(max_search_rounds)
        ),
    }
```

- [ ] **Step 4: Run the tests and verify GREEN**

Run: `python -m pytest -q tests/unit/eval/test_instruction_following.py`

Expected: PASS.

- [ ] **Step 5: Mutation-check the citation resolution**

```bash
python - <<'EOF'
from pathlib import Path
p = Path("src/model/post_training/eval/instruction_following.py")
p.write_text(p.read_text().replace(
    "    citations_ok = bool(labels) and all(\n"
    "        label.strip(\"[]\") in record.cited_ids for label in labels\n"
    "    )",
    "    citations_ok = bool(labels)"))
EOF
python -m pytest -q tests/unit/eval/test_instruction_following.py
git checkout src/model/post_training/eval/instruction_following.py
```

Expected: `test_a_citation_label_that_was_never_retrieved_fails` FAILS, then restore.

- [ ] **Step 6: Lint and commit**

```bash
ruff check src/model/post_training/eval tests/unit/eval --fix
ruff format src/model/post_training/eval tests/unit/eval
python -m pytest -q tests/unit/eval
git add src/model/post_training/eval/instruction_following.py tests/unit/eval/test_instruction_following.py
git commit -m "feat(eval): add objective instruction-following predicates"
```

---

### Task 4: Split, Measurements, and Report

**Files:**
- Create: `src/model/post_training/eval/unseen_users.py`
- Create: `tests/unit/eval/test_unseen_users.py`
- Modify: `src/model/post_training/eval/__init__.py`

**Interfaces:**
- Consumes: `roc_auc`, `cliffs_delta`, `benjamini_hochberg`, `cluster_bootstrap_ci`, `paired_permutation_p` (Task 1); `EvalRecord` (Task 2); `CONSTRAINT_NAMES`, `check_constraints` (Task 3).
- Produces:
  - `BEHAVIOR_COMPONENTS: tuple[str, ...]` = `("search_rounds", "web_searches", "vdb_searches", "rerank_calls", "repeated_search_queries")`
  - `split_users(user_ids: Iterable[str], *, holdout_fraction: float = 0.3, seed: int = 0) -> tuple[frozenset[str], frozenset[str]]` returning `(train_users, holdout_users)`
  - `AlignmentResult` frozen dataclass: `auc: float`, `ci_low: float`, `ci_high: float`, `n_users: int`, `n_excluded: int`
  - `ComparisonResult` frozen dataclass: `name: str`, `trained_mean: float`, `baseline_mean: float`, `effect: float`, `p_value: float`, `p_adjusted: float`
  - `UnseenUserReport` frozen dataclass: `n_holdout_users: int`, `alignment: AlignmentResult`, `behavior: tuple[ComparisonResult, ...]`, `instruction: tuple[ComparisonResult, ...]`, `provenance: str`
  - `evaluate_unseen_users(records: Sequence[EvalRecord], *, holdout_fraction: float = 0.3, seed: int = 0, resamples: int = 2000, allowed_tools: frozenset[str] = frozenset({"search", "fetch"}), max_search_rounds: int = 5, provenance: str = "") -> UnseenUserReport`
  - `format_report(report: UnseenUserReport) -> str`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/eval/test_unseen_users.py`:

```python
"""Contracts for the split, the measurements, and the report."""

from __future__ import annotations

import pytest

from src.model.post_training.eval.cohort import CohortConfig, generate_cohort
from src.model.post_training.eval.unseen_users import (
    BEHAVIOR_COMPONENTS,
    evaluate_unseen_users,
    format_report,
    split_users,
)

from src.model.post_training.eval.instruction_following import CONSTRAINT_NAMES


def test_split_is_deterministic_for_a_fixed_seed():
    ids = [f"u{i}" for i in range(50)]

    assert split_users(ids, seed=4) == split_users(ids, seed=4)


def test_split_is_independent_of_input_order():
    ids = [f"u{i}" for i in range(50)]
    forward = split_users(ids, seed=4)
    backward = split_users(list(reversed(ids)), seed=4)

    assert forward == backward


def test_no_user_appears_on_both_sides():
    train, holdout = split_users([f"u{i}" for i in range(80)], seed=1)

    assert not (train & holdout)


def test_split_covers_every_user():
    ids = [f"u{i}" for i in range(80)]
    train, holdout = split_users(ids, seed=1)

    assert train | holdout == set(ids)


def test_holdout_fraction_is_approximately_respected():
    train, holdout = split_users(
        [f"u{i}" for i in range(400)], holdout_fraction=0.25, seed=2
    )

    assert 0.18 <= len(holdout) / 400 <= 0.32


def test_a_different_seed_moves_users_across_the_split():
    _, first = split_users([f"u{i}" for i in range(200)], seed=1)
    _, second = split_users([f"u{i}" for i in range(200)], seed=2)

    assert first != second


def test_report_measures_only_held_out_users():
    records = generate_cohort(CohortConfig(num_users=40, sessions_per_user=8, seed=3))
    _, holdout = split_users({r.user_id for r in records}, seed=3)

    report = evaluate_unseen_users(records, seed=3, resamples=200)

    assert report.n_holdout_users == len(holdout)


def test_alignment_is_detected_on_an_aligned_cohort():
    records = generate_cohort(
        CohortConfig(num_users=60, sessions_per_user=12, alignment=3.0, seed=10)
    )

    report = evaluate_unseen_users(records, seed=10, resamples=400)

    assert report.alignment.auc > 0.5
    assert report.alignment.ci_low > 0.5


def test_alignment_reports_how_many_users_it_dropped():
    records = generate_cohort(CohortConfig(num_users=40, sessions_per_user=10, seed=11))

    report = evaluate_unseen_users(records, seed=11, resamples=200)

    assert report.alignment.n_excluded >= 0
    assert report.alignment.n_users + report.alignment.n_excluded == (
        report.n_holdout_users
    )


def test_behavioral_separation_is_detected_when_planted():
    records = generate_cohort(
        CohortConfig(num_users=60, sessions_per_user=12, behavior_shift=2.5, seed=12)
    )

    report = evaluate_unseen_users(records, seed=12, resamples=400)
    rounds = next(r for r in report.behavior if r.name == "search_rounds")

    assert rounds.trained_mean < rounds.baseline_mean
    assert rounds.p_adjusted < 0.05
    assert rounds.effect < 0.0


def test_instruction_following_beats_the_baseline_when_planted():
    records = generate_cohort(
        CohortConfig(num_users=60, sessions_per_user=12, instruction_gap=0.35, seed=13)
    )

    report = evaluate_unseen_users(records, seed=13, resamples=400)
    tags = next(r for r in report.instruction if r.name == "answer_tag_present")

    assert tags.trained_mean > tags.baseline_mean
    assert tags.p_adjusted < 0.05


def test_every_component_and_constraint_is_reported():
    records = generate_cohort(CohortConfig(num_users=30, sessions_per_user=8, seed=14))

    report = evaluate_unseen_users(records, seed=14, resamples=200)

    assert {r.name for r in report.behavior} == set(BEHAVIOR_COMPONENTS)
    assert {r.name for r in report.instruction} == set(CONSTRAINT_NAMES)


def test_adjusted_p_values_are_never_below_raw():
    records = generate_cohort(CohortConfig(num_users=30, sessions_per_user=8, seed=15))

    report = evaluate_unseen_users(records, seed=15, resamples=200)

    for result in (*report.behavior, *report.instruction):
        assert result.p_adjusted >= result.p_value


def test_evaluation_is_deterministic():
    records = generate_cohort(CohortConfig(num_users=25, sessions_per_user=6, seed=16))

    first = evaluate_unseen_users(records, seed=16, resamples=200)
    second = evaluate_unseen_users(records, seed=16, resamples=200)

    assert first == second


def test_evaluation_rejects_an_empty_record_set():
    with pytest.raises(ValueError, match="records"):
        evaluate_unseen_users([], seed=0)


def test_formatted_report_states_its_provenance_and_user_count():
    records = generate_cohort(CohortConfig(num_users=30, sessions_per_user=8, seed=17))

    report = evaluate_unseen_users(
        records, seed=17, resamples=200, provenance="simulated cohort"
    )
    text = format_report(report)

    assert "simulated cohort" in text
    assert str(report.n_holdout_users) in text
    assert "held-out users" in text


def test_formatted_report_shows_effect_size_beside_every_p_value():
    records = generate_cohort(CohortConfig(num_users=30, sessions_per_user=8, seed=18))

    text = format_report(evaluate_unseen_users(records, seed=18, resamples=200))

    for name in (*BEHAVIOR_COMPONENTS, *CONSTRAINT_NAMES):
        assert name in text
    assert "effect" in text.lower()
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest -q tests/unit/eval/test_unseen_users.py`

Expected: FAIL with `ModuleNotFoundError` for `...eval.unseen_users`.

- [ ] **Step 3: Implement split, measurements, and report**

Create `src/model/post_training/eval/unseen_users.py`:

```python
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
from dataclasses import dataclass

from .cohort import EvalRecord
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
```

- [ ] **Step 4: Export the public surface**

Append to `src/model/post_training/eval/__init__.py`'s imports and `__all__`:

```python
from .cohort import CohortConfig, EvalRecord, generate_cohort, null_cohort_config
from .instruction_following import CONSTRAINT_NAMES, check_constraints
from .unseen_users import (
    AlignmentResult,
    ComparisonResult,
    UnseenUserReport,
    evaluate_unseen_users,
    format_report,
    split_users,
)
```

Add every one of those names to `__all__`, keeping it sorted.

- [ ] **Step 5: Run the tests and verify GREEN**

Run: `python -m pytest -q tests/unit/eval`

Expected: PASS, all tests across the four modules.

- [ ] **Step 6: Mutation-check that the split is actually honoured**

```bash
python - <<'EOF'
from pathlib import Path
p = Path("src/model/post_training/eval/unseen_users.py")
p.write_text(p.read_text().replace(
    "    frame = [record for record in records if record.user_id in holdout]",
    "    frame = list(records)"))
EOF
python -m pytest -q tests/unit/eval/test_unseen_users.py::test_report_measures_only_held_out_users
git checkout src/model/post_training/eval/unseen_users.py
```

Expected: FAIL under the mutation, then restore.

- [ ] **Step 7: Lint and commit**

```bash
ruff check src/model/post_training/eval tests/unit/eval --fix
ruff format src/model/post_training/eval tests/unit/eval
python -m pytest -q tests/unit/eval
git add src/model/post_training/eval tests/unit/eval
git commit -m "feat(eval): measure alignment, separation and instruction following on held-out users"
```

---

### Task 5: The Two Credibility Tests and Achieved Power

**Files:**
- Modify: `src/model/post_training/eval/unseen_users.py`
- Create: `tests/unit/eval/test_credibility.py`

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces:
  - `achieved_power(config: CohortConfig, *, replications: int = 200, alpha: float = 0.05, resamples: int = 200, seed: int = 0) -> dict[str, float]` — rejection rate per measurement name, plus key `"alignment"` for the fraction of replications whose alignment CI lower bound exceeded 0.5

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/eval/test_credibility.py`:

```python
"""The two tests that make every other number in this harness meaningful.

If a null cohort produces significance, the harness manufactures p-values and
nothing it prints can be trusted. If ignoring clustering does not inflate
significance, the design's central claim is unfounded and the complexity of
clustering buys nothing.
"""

from __future__ import annotations

from src.model.post_training.eval.cohort import (
    CohortConfig,
    generate_cohort,
    null_cohort_config,
)
from src.model.post_training.eval.stats import paired_permutation_p
from src.model.post_training.eval.unseen_users import (
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
        report = evaluate_unseen_users(records, seed=105 + index, resamples=100)
        rounds = next(r for r in report.behavior if r.name == "search_rounds")
        by_hand.append(int(rounds.p_adjusted < 0.05))

    power = achieved_power(config, replications=2, resamples=100, seed=105)

    assert power["search_rounds"] == sum(by_hand) / 2
    # And the two replications must not be the same cohort.
    assert generate_cohort(replace(config, seed=105)) != generate_cohort(
        replace(config, seed=106)
    )
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest -q tests/unit/eval/test_credibility.py`

Expected: FAIL with `ImportError: cannot import name 'achieved_power'`.

- [ ] **Step 3: Implement achieved_power**

Append to `src/model/post_training/eval/unseen_users.py`:

```python
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
        report = evaluate_unseen_users(
            records, seed=seed + index, resamples=resamples
        )
        counts["alignment"] += int(report.alignment.ci_low > 0.5)
        for result in (*report.behavior, *report.instruction):
            counts[result.name] += int(result.p_adjusted < alpha)

    return {name: value / replications for name, value in counts.items()}
```

Widen the module's two import lines so the new function's names resolve —
Task 4 deliberately imported only what it used, so `ruff` would have failed on
unused imports there:

```python
from dataclasses import dataclass, replace

from .cohort import CohortConfig, EvalRecord, generate_cohort
```

- [ ] **Step 4: Run the tests and verify GREEN**

Run: `python -m pytest -q tests/unit/eval/test_credibility.py`

Expected: PASS. This is the slowest test file in the suite; if it
exceeds ~60s, lower `replications` in the tests rather than in the default.

- [ ] **Step 5: Confirm the null test is load-bearing**

Break the correction so the harness over-rejects, and confirm the null test
catches it:

```bash
python - <<'EOF'
from pathlib import Path
p = Path("src/model/post_training/eval/unseen_users.py")
p.write_text(p.read_text().replace(
    "    adjusted = benjamini_hochberg([result.p_value for result in family])",
    "    adjusted = [0.0 for _ in family]"))
EOF
python -m pytest -q tests/unit/eval/test_credibility.py::test_a_null_cohort_does_not_produce_significance
git checkout src/model/post_training/eval/unseen_users.py
```

Expected: FAIL under the mutation, then restore.

- [ ] **Step 6: Lint and commit**

```bash
ruff check src/model/post_training/eval tests/unit/eval --fix
ruff format src/model/post_training/eval tests/unit/eval
python -m pytest -q tests/unit/eval
git add src/model/post_training/eval/unseen_users.py tests/unit/eval/test_credibility.py
git commit -m "test(eval): pin the null-cohort and clustering-inflation guarantees"
```

---

### Task 6: CLI, Documentation, and Verification

**Files:**
- Create: `examples/run_unseen_user_eval.py`
- Create: `tests/unit/eval/test_run_unseen_user_eval.py`
- Modify: `docs/training-and-evaluation.md`
- Modify: `.claude/CLAUDE.md`

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: `python -m examples.run_unseen_user_eval [--users N] [--sessions N] [--holdout F] [--seed N] [--resamples N] [--power-replications N] [--output PATH]`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/eval/test_run_unseen_user_eval.py`:

```python
"""Contracts for the unseen-user evaluation CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_cli_writes_a_report_stating_it_is_simulated(tmp_path: Path):
    out = tmp_path / "report.md"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.run_unseen_user_eval",
            "--users", "24",
            "--sessions", "6",
            "--resamples", "100",
            "--power-replications", "0",
            "--output", str(out),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr
    text = out.read_text()
    assert "simulated" in text.lower()
    assert "held-out users" in text
    assert "Conversion alignment" in text


def test_cli_includes_power_when_replications_are_requested(tmp_path: Path):
    out = tmp_path / "report.md"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.run_unseen_user_eval",
            "--users", "20",
            "--sessions", "5",
            "--resamples", "50",
            "--power-replications", "3",
            "--output", str(out),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr
    assert "Achieved power" in out.read_text()


def test_cli_fails_loudly_on_an_out_of_range_holdout():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.run_unseen_user_eval",
            "--users", "10",
            "--sessions", "4",
            "--resamples", "20",
            "--power-replications", "0",
            "--holdout", "1.5",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode != 0
    assert "holdout_fraction" in result.stderr


def test_cli_runs_without_torch():
    program = """
import sys

class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("No module named %r (blocked)" % name)
        return None

sys.meta_path.insert(0, _Blocker())

import examples.run_unseen_user_eval  # noqa: F401
from src.model.post_training.eval import evaluate_unseen_users  # noqa: F401

assert "torch" not in sys.modules
print("eval is torch-free")
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "eval is torch-free" in result.stdout
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest -q tests/unit/eval/test_run_unseen_user_eval.py`

Expected: FAIL — `examples.run_unseen_user_eval` does not exist.

- [ ] **Step 3: Implement the CLI**

Create `examples/run_unseen_user_eval.py`:

```python
"""Run the unseen-user evaluation against a simulated cohort.

    python -m examples.run_unseen_user_eval --output docs/benchmarks/unseen-users.md

The cohort is generated, not observed. This repository holds two users and zero
feedback rows, so the report below is evidence about the *pipeline* -- that it
detects an effect of the configured size, on held-out users, at the reported
power -- and not evidence that any model converts real users. The report says so
itself; do not quote a number from it without that sentence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.model.post_training.eval import (
    CohortConfig,
    evaluate_unseen_users,
    format_report,
    generate_cohort,
)
from src.model.post_training.eval.unseen_users import achieved_power


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=60)
    parser.add_argument("--sessions", type=int, default=12)
    parser.add_argument("--holdout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--power-replications", type=int, default=200)
    parser.add_argument("--output", help="Markdown file to write.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = CohortConfig(
        num_users=args.users, sessions_per_user=args.sessions, seed=args.seed
    )
    report = evaluate_unseen_users(
        generate_cohort(config),
        holdout_fraction=args.holdout,
        seed=args.seed,
        resamples=args.resamples,
        provenance=(
            f"simulated cohort ({args.users} users x {args.sessions} sessions, "
            f"seed {args.seed}) -- NOT real users"
        ),
    )
    text = format_report(report)

    if args.power_replications > 0:
        power = achieved_power(
            config,
            replications=args.power_replications,
            resamples=min(args.resamples, 200),
            seed=args.seed,
        )
        rows = "\n".join(
            f"| `{name}` | {rate:.2f} |" for name, rate in sorted(power.items())
        )
        text += (
            "\n## Achieved power\n\n"
            f"Rejection rate over {args.power_replications} freshly seeded cohorts.\n\n"
            "| measurement | power |\n| --- | ---: |\n" + rows + "\n"
        )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"Wrote {out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and verify GREEN**

Run: `python -m pytest -q tests/unit/eval/test_run_unseen_user_eval.py`

Expected: PASS.

- [ ] **Step 5: Generate the committed report**

```bash
python -m examples.run_unseen_user_eval \
  --users 60 --sessions 12 --resamples 2000 --power-replications 200 \
  --output docs/benchmarks/unseen-user-evaluation.md
```

Read the generated file. Confirm it states it is simulated, reports the
held-out user count, and shows an effect size beside every p-value.

- [ ] **Step 6: Update documentation**

In `docs/training-and-evaluation.md`, add to the reference table:

```markdown
| Unseen-user evaluation | `python3 -m examples.run_unseen_user_eval` |
| Unseen-user eval report | `docs/benchmarks/unseen-user-evaluation.md` |
```

In `.claude/CLAUDE.md`, extend the `eval/` bullet under **Post-training**:

```markdown
- `eval/` — Bamboogle and action-policy benchmark harnesses, plus the
  unseen-user harness (`cohort.py`, `stats.py`, `instruction_following.py`,
  `unseen_users.py`): held-out-user conversion alignment, behavioral separation
  and instruction following, every statistic clustered by user. Torch-free.
  Its cohort is simulated — the report is evidence about the pipeline, not
  about real users
```

- [ ] **Step 7: Full verification**

```bash
ruff check src tests examples
ruff format --check src tests examples
git diff --check
python -m pytest -q
```

Then confirm the torch-free gate still collects everything:

```bash
python -c "
import sys
class B:
    def find_spec(self, n, path=None, target=None):
        if n == 'torch' or n.startswith('torch.'):
            raise ImportError('blocked')
        return None
sys.meta_path.insert(0, B())
import pytest
raise SystemExit(pytest.main(['-q', '--co', '-q', 'tests/unit', '-p', 'no:cacheprovider']))
" 2>&1 | tail -3
```

Expected: all commands exit zero; the collection reports **more** tests than
before this branch and **zero** collection errors.

- [ ] **Step 8: Commit, push, and open the PR**

```bash
git add src tests examples docs .claude
git commit -m "feat(eval): add unseen-user evaluation CLI and report"
git push -u origin eval/unseen-user-conversion
gh pr create --base main \
  --title "feat(eval): unseen-user conversion alignment and instruction-following harness" \
  --body-file docs/superpowers/context-packs/unseen-user-evaluation-pr.md
```

Write the PR body first. It must state, in the summary and not only in a
footnote, that the cohort is simulated and what that does and does not license
anyone to claim. Include the generated report's tables and the achieved-power
table. Report the PR URL.

---

## Notes for the Implementer

**The one thing that must not be "simplified".** Every statistic collapses to
one value per user before any test, and every resample draws users. If a review
comment suggests analysing per session because "there's more data that way",
that is the bug this design exists to prevent — `test_ignoring_clustering_inflates_significance`
is the demonstration.

**If a null-cohort test fails, do not raise the threshold.** A null cohort
rejecting more than the nominal rate means a real defect somewhere upstream —
most likely a leaked effect in the generator or a test that is not actually
paired. Find it.

**Effect size travels with every p-value.** A significant, negligible effect
must be legible as negligible. Do not drop the Cliff's delta column to make a
table narrower.
