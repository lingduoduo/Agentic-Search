"""Index behavior over hand-built orthogonal vectors.

Deliberately no encoder: the unit-test CI job installs neither torch nor
sentence-transformers, and all routing math must be exercisable without them.
Canonical vectors sit on the basis axes so every cosine is exact.
"""

from pathlib import Path

import numpy as np
import pytest

from src.model.intent_knn import (
    MIN_MODULE_SUPPORT,
    CanonicalExample,
    IntentIndex,
)

# search -> e0, chat -> e1, tool -> e2. Cosines are then exactly readable.
_AXIS = {"search": 0, "chat": 1, "tool": 2}
_MODULE = {"search": "lookup_fact", "chat": "explain", "tool": "schedule"}


def _unit(*components: float) -> np.ndarray:
    vector = np.array(components, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _index(per_route: int = 3, **kwargs) -> IntentIndex:
    examples, rows = [], []
    for route, axis in _AXIS.items():
        for position in range(per_route):
            examples.append(
                CanonicalExample(
                    id=f"{route}-{position}",
                    text=f"{route} example {position}",
                    route=route,
                    modules=(_MODULE[route],),
                )
            )
            rows.append(np.eye(3, dtype=np.float32)[axis])
    return IntentIndex(
        examples=examples,
        vectors=np.stack(rows),
        encoder="test-encoder",
        fingerprint="sha256:test",
        **kwargs,
    )


def _decide(index: IntentIndex, vector, **overrides):
    thresholds = {"min_confidence": 0.5, "min_margin": 0.05, "min_module_score": 0.4}
    return index.decide(vector, **{**thresholds, **overrides})


def test_query_on_a_route_axis_picks_that_route_with_full_confidence():
    decision = _decide(_index(), _unit(1, 0, 0))

    assert decision.route == "search"
    assert decision.confidence == pytest.approx(1.0)
    assert decision.margin == pytest.approx(1.0)
    assert decision.abstained is False


def test_route_score_is_the_mean_of_the_top_three_not_of_all_examples():
    """One near neighbor must not be diluted by distant same-route examples."""
    examples = [
        CanonicalExample(f"s-{i}", f"t{i}", "search", ("lookup_fact",))
        for i in range(5)
    ]
    examples += [CanonicalExample("c-0", "t", "chat", ("explain",))]
    vectors = np.stack([_unit(1, 0, 0)] * 3 + [_unit(0, 0, 1)] * 2 + [_unit(0, 1, 0)])
    index = IntentIndex(examples, vectors, "test-encoder", "sha256:test")

    scores = index.route_scores(_unit(1, 0, 0))

    # Top 3 of search are the three exact matches; the two distant ones do not
    # drag the mean down.
    assert scores["search"] == pytest.approx(1.0)


def test_fewer_than_three_examples_averages_what_exists():
    examples = [CanonicalExample("s-0", "t", "search", ("lookup_fact",))]
    examples += [CanonicalExample("c-0", "t", "chat", ("explain",))]
    examples += [CanonicalExample("t-0", "t", "tool", ("schedule",))]
    vectors = np.stack([_unit(1, 0, 0), _unit(0, 1, 0), _unit(0, 0, 1)])
    index = IntentIndex(examples, vectors, "test-encoder", "sha256:test")

    assert index.route_scores(_unit(1, 0, 0))["search"] == pytest.approx(1.0)


def test_equidistant_query_abstains_on_margin_not_on_confidence():
    """Ambiguity and out-of-scope are different failures with different causes."""
    decision = _decide(_index(), _unit(1, 1, 0), min_confidence=0.5)

    assert decision.confidence == pytest.approx(0.7071, abs=1e-3)
    assert decision.margin == pytest.approx(0.0, abs=1e-6)
    assert decision.abstained is True
    assert decision.abstain_reason == "margin_below_threshold"


def test_query_far_from_everything_abstains_on_confidence():
    """Cosine is unnormalized across routes, so 'far from all' is expressible."""
    decision = _decide(_index(), _unit(1, 1, 1), min_confidence=0.7)

    assert decision.confidence == pytest.approx(0.5774, abs=1e-3)
    assert decision.abstained is True
    assert decision.abstain_reason == "confidence_below_threshold"


def test_confidence_is_checked_before_margin():
    """A query far from everything is out of scope, not merely ambiguous."""
    decision = _decide(_index(), _unit(1, 1, 1), min_confidence=0.7, min_margin=0.9)

    assert decision.abstain_reason == "confidence_below_threshold"


def test_a_confident_decision_still_reports_its_route_and_modules():
    """Abstention defers routing; it never blanks the diagnostics.

    Note the asymmetric vector: exactly equal route scores tie-break by
    INTENT_LABELS order, which puts chat first.
    """
    decision = _decide(_index(), _unit(1, 0.98, 0), min_confidence=0.5)

    assert decision.abstained is True
    assert decision.route == "search"
    assert decision.modules == ("lookup_fact",)


def test_modules_are_multi_label_above_the_module_threshold():
    examples = [
        CanonicalExample("s-0", "t", "search", ("lookup_fact", "current_info")),
        CanonicalExample("s-1", "t", "search", ("lookup_document",)),
        CanonicalExample("c-0", "t", "chat", ("explain",)),
        CanonicalExample("t-0", "t", "tool", ("schedule",)),
    ]
    vectors = np.stack(
        [_unit(1, 0, 0), _unit(0, 0.2, 1), _unit(0, 1, 0), _unit(0, 0, 1)]
    )
    index = IntentIndex(examples, vectors, "test-encoder", "sha256:test")

    decision = _decide(index, _unit(1, 0, 0), min_module_score=0.5)

    assert set(decision.modules) == {"lookup_fact", "current_info"}


def test_modules_fall_back_to_the_single_best_when_none_clear_the_bar():
    decision = _decide(_index(), _unit(1, 0, 0), min_module_score=0.99999)

    assert decision.modules == ("lookup_fact",)


def test_modules_come_only_from_the_chosen_route():
    decision = _decide(_index(), _unit(1, 0, 0), min_module_score=0.0)

    assert decision.modules == ("lookup_fact",)


def test_module_threshold_cannot_change_the_route():
    """The module knob is diagnostic; it must never move routing."""
    routes = {
        _decide(_index(), _unit(1, 0, 0), min_module_score=score).route
        for score in (0.0, 0.5, 0.99)
    }

    assert routes == {"search"}


def test_composite_fires_when_a_close_runner_up_is_an_action():
    """The 'find the best place and book it' signature.

    Composite means "closer than the margin bar", so it always co-occurs with a
    margin abstention. That is correct: such a request needs two steps, and one
    route cannot serve it.
    """
    decision = _decide(_index(), _unit(1, 0, 0.98))

    assert decision.route == "search"
    assert decision.composite is True
    assert decision.abstain_reason == "margin_below_threshold"


def test_composite_does_not_fire_for_a_close_non_action_runner_up():
    decision = _decide(_index(), _unit(1, 0.98, 0))

    assert decision.composite is False


def test_composite_does_not_fire_when_the_runner_up_is_distant():
    decision = _decide(_index(), _unit(1, 0, 0), min_margin=0.05)

    assert decision.composite is False


def test_low_support_modules_are_reported():
    index = _index(per_route=3)

    assert set(index.low_support_modules()) >= {"lookup_fact", "explain", "schedule"}
    assert MIN_MODULE_SUPPORT == 10


def test_a_decision_always_carries_a_module_even_when_all_are_thin():
    """Support filtering must not leave a decision with no module at all."""
    index = _index(per_route=3)
    assert "lookup_fact" in index.low_support_modules()

    decision = _decide(index, _unit(1, 0, 0), min_module_score=0.0)

    assert decision.modules == ("lookup_fact",)


def test_a_thin_module_is_dropped_when_a_supported_one_exists():
    """Support filtering bites only when it has something better to offer."""
    examples, rows = [], []
    for position in range(12):
        examples.append(
            CanonicalExample(f"s-{position}", "t", "search", ("lookup_fact",))
        )
        rows.append(_unit(1, 0, 0))
    examples.append(CanonicalExample("s-thin", "t", "search", ("current_info",)))
    rows.append(_unit(1, 0, 0))
    examples.append(CanonicalExample("c-0", "t", "chat", ("explain",)))
    rows.append(_unit(0, 1, 0))
    examples.append(CanonicalExample("t-0", "t", "tool", ("schedule",)))
    rows.append(_unit(0, 0, 1))
    index = IntentIndex(examples, np.stack(rows), "test-encoder", "sha256:x")

    decision = _decide(index, _unit(1, 0, 0), min_module_score=0.0)

    assert decision.modules == ("lookup_fact",)
    assert "current_info" in index.low_support_modules()


def test_constructor_rejects_unnormalized_vectors():
    """An unnormalized row turns cosine into an arbitrary dot product."""
    examples = [CanonicalExample("s-0", "t", "search", ("lookup_fact",))]

    with pytest.raises(ValueError, match="normalized"):
        IntentIndex(
            examples,
            np.array([[2.0, 0.0, 0.0]], dtype=np.float32),
            "test-encoder",
            "sha256:test",
        )


def test_constructor_rejects_a_row_count_that_disagrees_with_the_examples():
    examples = [CanonicalExample("s-0", "t", "search", ("lookup_fact",))]

    with pytest.raises(ValueError, match="rows"):
        IntentIndex(examples, np.eye(3, dtype=np.float32), "test-encoder", "sha256:x")


def test_constructor_rejects_modules_outside_the_example_route():
    examples = [CanonicalExample("s-0", "t", "search", ("summarize",))]

    with pytest.raises(ValueError, match="summarize"):
        IntentIndex(
            examples,
            np.eye(3, dtype=np.float32)[:1],
            "test-encoder",
            "sha256:x",
        )


def test_round_trip_preserves_examples_vectors_and_provenance(tmp_path: Path):
    index = _index()
    path = tmp_path / "index.npz"
    index.save(path)

    reloaded = IntentIndex.load(path)

    assert reloaded.size == index.size
    assert reloaded.encoder == "test-encoder"
    assert reloaded.fingerprint == "sha256:test"
    before = _decide(index, _unit(1, 0, 0))
    after = _decide(reloaded, _unit(1, 0, 0))
    assert after.route == before.route
    assert after.confidence == pytest.approx(before.confidence)
    assert after.modules == before.modules


def test_load_reports_a_missing_index_by_name(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="index.npz"):
        IntentIndex.load(tmp_path / "index.npz")
