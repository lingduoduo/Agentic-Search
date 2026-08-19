"""Structural guarantees for the evaluation instrument itself.

These test the measuring device, not the model. A silently degraded eval set
would make every later number meaningless.
"""

from collections import Counter
from pathlib import Path

import pytest

from src.model.pre_training.intents.data import load_intent_eval_queries
from src.model.pre_training.intents.model import SEMANTIC_MODULES

DATA = Path(__file__).resolve().parents[2] / "data"
BULK = DATA / "intent_eval_queries.json"
HARD = DATA / "intent_eval_hard.json"


def _queries(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} not authored yet")
    return load_intent_eval_queries(path)


def test_bulk_set_is_large_enough_for_one_query_to_not_dominate():
    queries = _queries(BULK)

    assert len(queries) >= 170


def test_bulk_set_is_balanced_across_routes():
    counts = Counter(query.label for query in _queries(BULK))

    assert min(counts.values()) >= 0.25 * sum(counts.values())


def test_every_bulk_query_carries_at_least_one_module():
    missing = [query.id for query in _queries(BULK) if not query.modules]

    assert missing == []


def test_the_original_thirty_queries_survive_unchanged():
    """Continuity with the pinned 0.733 depends on these exact queries.

    Legacy ids are `eval-<route>-NN`; the new bulk queries use `bulk-NNN`, so
    the two are distinguishable by prefix and neither can swallow the other.
    """
    legacy = [q for q in _queries(BULK) if q.id.startswith("eval-")]

    assert len(legacy) == 30
    assert {q.id for q in legacy} == {
        f"eval-{route}-{n:02d}"
        for route in ("search", "chat", "tool")
        for n in range(1, 11)
    }


def test_new_bulk_queries_use_the_bulk_prefix():
    added = [q for q in _queries(BULK) if not q.id.startswith("eval-")]

    assert added
    assert all(q.id.startswith("bulk-") for q in added)


def test_hard_slice_exists_and_is_sized_for_triplets():
    queries = _queries(HARD)

    assert 34 <= len(queries) <= 46


def test_hard_slice_is_built_from_minimal_triplets():
    """Same entity across routes: it isolates the boundary from entity difficulty."""
    queries = _queries(HARD)
    groups = Counter(
        query.id.rsplit("-", 1)[0] for query in queries if query.id.startswith("hard-")
    )
    triplets = [group for group, count in groups.items() if count == 3]

    assert len(triplets) >= 10


def test_each_triplet_spans_more_than_one_route():
    queries = _queries(HARD)
    by_group: dict[str, set[str]] = {}
    for query in queries:
        if query.id.startswith("hard-"):
            by_group.setdefault(query.id.rsplit("-", 1)[0], set()).add(query.label)
    multi = [group for group, routes in by_group.items() if len(routes) > 1]

    assert len(multi) >= 10


def test_every_semantic_module_appears_in_the_evaluation_sets():
    """A module never evaluated is a module with no evidence behind it."""
    seen = {
        module
        for path in (BULK, HARD)
        for query in _queries(path)
        for module in query.modules
    }

    assert set(SEMANTIC_MODULES) - seen == set()


def test_no_query_text_is_repeated_across_the_two_sets():
    texts = [q.text.casefold().strip() for q in _queries(BULK)]
    texts += [q.text.casefold().strip() for q in _queries(HARD)]

    assert len(texts) == len(set(texts))
