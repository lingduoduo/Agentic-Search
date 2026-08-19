from collections import Counter

import pytest

from src.model.pre_training.intents.data import IntentEvalQuery
from src.model.pre_training.intents.evaluation import split_eval_queries


def _queries(n_clean: int = 151, n_legacy: int = 30):
    routes = ("chat", "search", "tool")
    legacy = [
        IntentEvalQuery(
            f"eval-{routes[i % 3]}-{i:02d}", f"legacy {i}", routes[i % 3], ()
        )
        for i in range(n_legacy)
    ]
    clean = [
        IntentEvalQuery(f"bulk-{i:03d}", f"clean {i}", routes[i % 3], ())
        for i in range(n_clean)
    ]
    return legacy + clean


def test_every_legacy_query_goes_to_tuning():
    """They are contaminated, so they are worthless as a gate and free to tune on."""
    split = split_eval_queries(_queries(n_legacy=30))

    assert len([q for q in split.tuning if q.id.startswith("eval-")]) == 30
    assert not any(q.id.startswith("eval-") for q in split.test)


def test_the_tuning_slice_takes_the_requested_number_of_clean_queries():
    split = split_eval_queries(_queries(), slice_size=40)

    clean_tuning = [q for q in split.tuning if q.id.startswith("bulk-")]
    assert len(clean_tuning) == 40
    assert len(split.test) == 151 - 40


def test_the_split_is_a_partition_with_no_overlap():
    split = split_eval_queries(_queries())

    ids_tuning = {q.id for q in split.tuning}
    ids_test = {q.id for q in split.test}
    assert not (ids_tuning & ids_test)
    assert len(ids_tuning) + len(ids_test) == 181


def test_the_clean_tuning_slice_is_route_stratified():
    split = split_eval_queries(_queries(), slice_size=39)

    counts = Counter(q.label for q in split.tuning if q.id.startswith("bulk-"))
    assert set(counts) == {"chat", "search", "tool"}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_the_split_is_deterministic_for_a_seed():
    a = split_eval_queries(_queries(), seed=17)
    b = split_eval_queries(_queries(), seed=17)

    assert [q.id for q in a.test] == [q.id for q in b.test]


def test_the_split_is_deterministic_regardless_of_input_order():
    """Determinism is keyed on (queries, slice_size, seed), not input order.

    rng.sample's output depends on the order of the sequence it draws from,
    so an unsorted pool would make the split depend on how the caller happens
    to have ordered its query list -- an implicit, unstated input.
    """
    forward = _queries()
    reversed_input = list(reversed(forward))

    a = split_eval_queries(forward, seed=17)
    b = split_eval_queries(reversed_input, seed=17)

    assert {q.id for q in a.tuning} == {q.id for q in b.tuning}
    assert [q.id for q in a.test] == [q.id for q in b.test]


def test_a_different_seed_gives_a_different_split():
    a = split_eval_queries(_queries(), seed=17)
    b = split_eval_queries(_queries(), seed=18)

    assert [q.id for q in a.test] != [q.id for q in b.test]


def test_a_slice_larger_than_the_clean_set_is_rejected():
    with pytest.raises(ValueError, match="slice_size"):
        split_eval_queries(_queries(n_clean=10), slice_size=40)


def test_a_non_positive_slice_size_is_rejected():
    with pytest.raises(ValueError, match="slice_size"):
        split_eval_queries(_queries(), slice_size=0)


def test_a_duplicate_query_id_is_rejected():
    """A duplicate id would silently drop a query from both sides of the split.

    ``test`` is filtered by id, not object identity, so if the sampled half
    of a duplicate-id pair lands in tuning, the id-based filter excludes the
    *other*, un-sampled copy from test too -- both copies vanish from one
    side or the other, breaking the partition invariant.
    """
    queries = list(_queries())
    duplicate = IntentEvalQuery(queries[-1].id, "a duplicate", queries[-1].label, ())
    queries.append(duplicate)

    with pytest.raises(ValueError, match="duplicate"):
        split_eval_queries(queries)


# --- out-of-scope probe split ---


def _probes(counts: dict[str, int]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (f"oos-{category}-{i:02d}", f"{category} probe {i}")
        for category, n in counts.items()
        for i in range(1, n + 1)
    )


def test_probe_split_is_stratified_across_categories():
    """Categories are not interchangeable, so neither half may monopolise one.

    A reporting half holding all the gibberish and none of the injections would
    measure something quite different from one holding the reverse, and the
    reported AUC would silently depend on which.
    """
    from src.model.pre_training.intents.evaluation import split_out_of_scope_probes

    split = split_out_of_scope_probes(_probes({"personal": 6, "injection": 6}))

    for half in (split.tuning, split.reporting):
        categories = {pid.rsplit("-", 1)[0] for pid, _ in half}
        assert categories == {"oos-personal", "oos-injection"}


def test_probe_split_never_empties_a_half_on_small_categories():
    """A category of 2 must contribute to both halves, not vanish from one."""
    from src.model.pre_training.intents.evaluation import split_out_of_scope_probes

    split = split_out_of_scope_probes(_probes({"personal": 2, "smalltalk": 2}))

    assert len(split.tuning) == 2
    assert len(split.reporting) == 2


def test_probe_split_rejects_duplicate_ids():
    from src.model.pre_training.intents.evaluation import split_out_of_scope_probes

    duplicated = (("oos-a-01", "one"), ("oos-a-01", "two"), ("oos-b-01", "three"))
    with pytest.raises(ValueError, match="duplicate probe id"):
        split_out_of_scope_probes(duplicated)


def test_probe_split_rejects_a_share_that_cannot_produce_two_halves():
    from src.model.pre_training.intents.evaluation import split_out_of_scope_probes

    for share in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="tuning_share"):
            split_out_of_scope_probes(_probes({"personal": 4}), tuning_share=share)
