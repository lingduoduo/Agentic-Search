"""The tuning/test split that keeps hyperparameters off the reported number.

Three hyperparameters need values — top_k and the two abstention thresholds —
and choosing any of them on the queries used to report accuracy inflates that
accuracy. The split spends the cheapest data first: the legacy queries are
already contaminated (the canonical set was iterated against them during
curation), so they are worthless as a gate and free to tune on, which
preserves the clean queries as an untouched test set.

Pure Python: no numpy, no encoder. The split must be reproducible anywhere.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

LEGACY_PREFIX = "eval-"
DEFAULT_SLICE_SIZE = 40
DEFAULT_SEED = 17
# Fraction of the out-of-scope probes reserved for tuning. The rest denominate
# the reported separability and are never consulted by a sweep.
DEFAULT_PROBE_TUNING_SHARE = 0.5


@dataclass(frozen=True)
class EvalSplit:
    """Queries hyperparameters may see, and queries they may not."""

    tuning: tuple
    test: tuple


@dataclass(frozen=True)
class ProbeSplit:
    """Out-of-scope probes a sweep may see, and probes it may not."""

    tuning: tuple
    reporting: tuple


def split_out_of_scope_probes(
    probes: Sequence[tuple[str, str]],
    *,
    tuning_share: float = DEFAULT_PROBE_TUNING_SHARE,
    seed: int = DEFAULT_SEED,
) -> ProbeSplit:
    """Split probes into (tuning, reporting), stratified by category.

    Closes a caveat this repo has carried since #512: the *same* out-of-scope
    probes both tie-broke the threshold sweep and denominated the reported AUC
    and Cohen's d, so the headline separability figure was never fully held out
    from what selected the thresholds it was measured at.

    Stratified by the category encoded in the probe id (``oos-<category>-NN``),
    because the categories are not interchangeable — a reporting half that drew
    all the gibberish and none of the injections would measure something quite
    different from one that drew the reverse.

    Ids are sorted before sampling for the same reason ``split_eval_queries``
    sorts: the result must depend only on (probes, tuning_share, seed), never on
    the order the file happens to be written in.
    """
    if not 0.0 < tuning_share < 1.0:
        raise ValueError(f"tuning_share must be in (0, 1), got {tuning_share}")

    seen: dict[str, int] = {}
    for probe_id, _ in probes:
        seen[probe_id] = seen.get(probe_id, 0) + 1
    duplicates = sorted(pid for pid, count in seen.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate probe id(s) in input: {duplicates}")

    by_category: dict[str, list[tuple[str, str]]] = {}
    for probe in sorted(probes, key=lambda p: p[0]):
        by_category.setdefault(probe[0].rsplit("-", 1)[0], []).append(probe)

    rng = random.Random(seed)
    tuning: list[tuple[str, str]] = []
    reporting: list[tuple[str, str]] = []
    for _, group in sorted(by_category.items()):
        shuffled = rng.sample(group, len(group))
        # round() rather than int(): with an odd group the half that gets the
        # extra probe should alternate with the count, not always be reporting.
        take = round(len(shuffled) * tuning_share)
        take = min(max(take, 1), len(shuffled) - 1) if len(shuffled) > 1 else 0
        tuning.extend(shuffled[:take])
        reporting.extend(shuffled[take:])

    if not tuning or not reporting:
        raise ValueError(
            f"probe split left an empty half: {len(tuning)} tuning, "
            f"{len(reporting)} reporting, from {len(probes)} probes"
        )
    return ProbeSplit(tuning=tuple(tuning), reporting=tuple(reporting))


def split_eval_queries(
    queries: Sequence,
    *,
    slice_size: int = DEFAULT_SLICE_SIZE,
    seed: int = DEFAULT_SEED,
) -> EvalSplit:
    """Split into (tuning, test).

    Tuning is every legacy query plus a seeded, route-stratified *slice_size*
    sample of the clean ones. Test is the remaining clean queries.
    """
    if slice_size <= 0:
        raise ValueError(f"slice_size must be positive, got {slice_size}")

    seen_ids: dict[str, int] = {}
    for query in queries:
        seen_ids[query.id] = seen_ids.get(query.id, 0) + 1
    duplicates = sorted(qid for qid, count in seen_ids.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate query id(s) in input: {duplicates}")

    legacy = [q for q in queries if q.id.startswith(LEGACY_PREFIX)]
    # Sorted by id, not left in input order: the split must depend only on
    # (queries, slice_size, seed), and rng.sample's output depends on the
    # order of the sequence it draws from.
    clean = sorted(
        (q for q in queries if not q.id.startswith(LEGACY_PREFIX)), key=lambda q: q.id
    )
    if slice_size >= len(clean):
        raise ValueError(
            f"slice_size {slice_size} leaves no test queries: only "
            f"{len(clean)} clean queries available"
        )

    by_route: dict[str, list] = {}
    for query in clean:
        by_route.setdefault(query.label, []).append(query)

    rng = random.Random(seed)
    sampled: list = []
    # Round-robin across routes so the slice is stratified even when
    # slice_size is not divisible by the number of routes.
    pools = {
        route: rng.sample(group, len(group))
        for route, group in sorted(by_route.items())
    }
    while len(sampled) < slice_size:
        for route in sorted(pools):
            if len(sampled) == slice_size:
                break
            if pools[route]:
                sampled.append(pools[route].pop())

    chosen = {query.id for query in sampled}
    return EvalSplit(
        tuning=tuple(legacy + sampled),
        test=tuple(q for q in clean if q.id not in chosen),
    )
