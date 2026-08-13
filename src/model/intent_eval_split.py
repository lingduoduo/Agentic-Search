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


@dataclass(frozen=True)
class EvalSplit:
    """Queries hyperparameters may see, and queries they may not."""

    tuning: tuple
    test: tuple


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
    legacy = [q for q in queries if q.id.startswith(LEGACY_PREFIX)]
    clean = [q for q in queries if not q.id.startswith(LEGACY_PREFIX)]
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
