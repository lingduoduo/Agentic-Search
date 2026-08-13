"""Serving adapter behavior, exercised without an encoder."""

from pathlib import Path

import numpy as np
import pytest

from src.internal.configs import AppSettings
from src.internal.servers.web import ml_intent
from src.internal.servers.web.intent_routing import RouteStrategy
from src.model.intent_encoder import DEFAULT_ENCODER
from src.model.intent_knn import INDEX_FILENAME, CanonicalExample, IntentIndex

_AXIS = {"search": 0, "chat": 1, "tool": 2}
_MODULE = {"search": "lookup_fact", "chat": "explain", "tool": "schedule"}


def _write_index(tmp_path: Path) -> Path:
    examples, rows = [], []
    for route, axis in _AXIS.items():
        for position in range(12):
            examples.append(
                CanonicalExample(
                    f"{route}-{position}",
                    f"{route} {position}",
                    route,
                    (_MODULE[route],),
                )
            )
            rows.append(np.eye(3, dtype=np.float32)[axis])
    directory = tmp_path / "index"
    IntentIndex(examples, np.stack(rows), DEFAULT_ENCODER, "sha256:x").save(
        directory / INDEX_FILENAME
    )
    return directory


def _settings(tmp_path: Path, **overrides) -> AppSettings:
    defaults = {
        "intent_model_min_confidence": 0.5,
        "intent_min_route_margin": 0.05,
        "intent_min_module_score": 0.4,
    }
    return AppSettings(
        intent_index_path=_write_index(tmp_path), **{**defaults, **overrides}
    )


@pytest.fixture(autouse=True)
def _clear_cache():
    ml_intent._INTENT_INDEXES.clear()
    yield
    ml_intent._INTENT_INDEXES.clear()


def _on_axis(route: str, monkeypatch):
    vector = np.eye(3, dtype=np.float32)[_AXIS[route]][None, :]
    monkeypatch.setattr(ml_intent, "encode_texts", lambda texts: vector)


def test_confident_query_returns_its_route_and_modules(tmp_path, monkeypatch):
    _on_axis("search", monkeypatch)

    decision = ml_intent.predict_route("anything", settings=_settings(tmp_path))

    assert decision is not None
    assert decision.strategy is RouteStrategy.SEARCH
    assert decision.confidence == pytest.approx(1.0)
    assert decision.modules == ("lookup_fact",)
    assert decision.latency_ms >= 0.0


def test_predict_route_passes_settings_top_k_through_to_decide(tmp_path, monkeypatch):
    """decide()'s neighbor count must come from AppSettings, not the module default."""
    examples = [
        CanonicalExample("s-exact", "s", "search", ("lookup_fact",)),
    ]
    examples += [
        CanonicalExample(f"s-far-{i}", "s", "search", ("lookup_fact",))
        for i in range(3)
    ]
    for route, axis in (("chat", 1), ("tool", 2)):
        for position in range(12):
            examples.append(
                CanonicalExample(
                    f"{route}-{position}",
                    f"{route} {position}",
                    route,
                    (_MODULE[route],),
                )
            )
    rows = (
        [np.eye(3, dtype=np.float32)[0]]
        + [np.eye(3, dtype=np.float32)[2]] * 3
        + [np.eye(3, dtype=np.float32)[1]] * 12
        + [np.eye(3, dtype=np.float32)[2]] * 12
    )
    directory = tmp_path / "index"
    IntentIndex(examples, np.stack(rows), DEFAULT_ENCODER, "sha256:x").save(
        directory / INDEX_FILENAME
    )
    monkeypatch.setattr(
        ml_intent,
        "encode_texts",
        lambda texts: np.eye(3, dtype=np.float32)[0][None, :],
    )
    base = {
        "intent_index_path": directory,
        "intent_model_min_confidence": 0.1,
        "intent_min_route_margin": 0.0,
        "intent_min_module_score": 0.0,
    }

    decision_top_1 = ml_intent.predict_route(
        "anything", settings=AppSettings(**base, intent_top_k=1)
    )
    decision_top_4 = ml_intent.predict_route(
        "anything", settings=AppSettings(**base, intent_top_k=4)
    )

    assert decision_top_1 is not None and decision_top_4 is not None
    assert decision_top_1.confidence == pytest.approx(1.0)
    assert decision_top_4.confidence == pytest.approx(0.25)


def test_low_confidence_is_returned_for_route_request_to_abstain_on(
    tmp_path, monkeypatch
):
    """route_request already compares confidence to threshold; do not duplicate."""
    monkeypatch.setattr(
        ml_intent,
        "encode_texts",
        lambda texts: np.array([[0.577, 0.577, 0.577]], dtype=np.float32),
    )

    # A three-way tie has margin 0.0, which would also fail the margin gate at
    # the default threshold (0.05) — that's test_low_margin_defers_by_returning_none.
    # IntentIndex.decide() checks confidence first (elif margin), so overriding
    # the confidence bar above 0.577 isolates the confidence-only path this
    # test targets, without depending on the tied margin at all.
    decision = ml_intent.predict_route(
        "anything", settings=_settings(tmp_path, intent_model_min_confidence=0.9)
    )

    assert decision is not None
    assert decision.confidence < decision.threshold


def test_low_margin_defers_by_returning_none(tmp_path, monkeypatch):
    """route_request has no margin concept, so this abstention happens here."""
    monkeypatch.setattr(
        ml_intent,
        "encode_texts",
        lambda texts: np.array([[0.707, 0.707, 0.0]], dtype=np.float32),
    )

    assert ml_intent.predict_route("anything", settings=_settings(tmp_path)) is None


def test_missing_index_path_defers_without_raising(tmp_path):
    settings = AppSettings(intent_index_path=None)

    assert ml_intent.predict_route("anything", settings=settings) is None


def test_unreadable_index_defers_and_is_not_retried(tmp_path, monkeypatch):
    settings = AppSettings(intent_index_path=tmp_path / "absent")
    loads = {"count": 0}
    original = IntentIndex.load

    def _counting_load(path):
        loads["count"] += 1
        return original(path)

    monkeypatch.setattr(IntentIndex, "load", staticmethod(_counting_load))

    assert ml_intent.predict_route("anything", settings=settings) is None
    assert ml_intent.predict_route("anything", settings=settings) is None
    assert loads["count"] == 1


def test_encoder_mismatch_defers_and_is_cached_as_failure(tmp_path, monkeypatch):
    """A same- or different-dimension encoder mismatch must never silently
    score garbage dot products; the index records which encoder built it, and
    a mismatch must become one loud, cached failure instead."""
    examples = [CanonicalExample("search-0", "search 0", "search", ("lookup_fact",))]
    directory = tmp_path / "mismatched"
    IntentIndex(
        examples, np.eye(3, dtype=np.float32)[:1], "wrong-encoder", "sha256:x"
    ).save(directory / INDEX_FILENAME)
    settings = AppSettings(intent_index_path=directory)
    loads = {"count": 0}
    original = IntentIndex.load

    def _counting_load(path):
        loads["count"] += 1
        return original(path)

    monkeypatch.setattr(IntentIndex, "load", staticmethod(_counting_load))

    assert ml_intent.predict_route("anything", settings=settings) is None
    assert ml_intent.predict_route("anything", settings=settings) is None
    assert loads["count"] == 1


def test_encoder_failure_defers_rather_than_failing_the_request(tmp_path, monkeypatch):
    def _boom(texts):
        raise RuntimeError("no model")

    monkeypatch.setattr(ml_intent, "encode_texts", _boom)

    assert ml_intent.predict_route("anything", settings=_settings(tmp_path)) is None


def test_index_is_loaded_once_and_cached(tmp_path, monkeypatch):
    _on_axis("search", monkeypatch)
    settings = _settings(tmp_path)
    loads = {"count": 0}
    original = IntentIndex.load

    def _counting_load(path):
        loads["count"] += 1
        return original(path)

    monkeypatch.setattr(IntentIndex, "load", staticmethod(_counting_load))

    ml_intent.predict_route("a", settings=settings)
    ml_intent.predict_route("b", settings=settings)

    assert loads["count"] == 1


def test_composite_query_defers_and_is_recorded_as_composite(tmp_path, monkeypatch):
    """A composite request is by definition low-margin, so it defers.

    The flag therefore reaches the capture stage, never a returned decision.
    """
    monkeypatch.setattr(
        ml_intent,
        "encode_texts",
        lambda texts: np.array([[0.71, 0.0, 0.70]], dtype=np.float32),
    )
    recorded = []
    monkeypatch.setattr(
        ml_intent._capture,
        "record_stage",
        lambda stage, mechanism, detail: recorded.append((stage, detail)),
    )

    assert ml_intent.predict_route("anything", settings=_settings(tmp_path)) is None
    assert recorded[-1][0] == "intent_model"
    assert recorded[-1][1]["composite"] is True
    assert recorded[-1][1]["fallback_reason"] == "margin_below_threshold"


def test_an_index_built_with_the_previous_encoder_is_rejected(tmp_path, monkeypatch):
    """e5-small is also 384-d, so a stale index would otherwise score silently.

    The fixture must be genuinely 384-dimensional — the same width as e5's
    real output — not the 3-dim toy vectors this file uses elsewhere. With
    3-dim rows, index.decide() raises a numpy shape mismatch against the
    real-width query vector regardless of any encoder-name check, and
    predict_route's broad ``except Exception: return None`` swallows that —
    which would make this assertion hold even with the encoder-name guard
    removed. A structurally valid, same-width index, paired with a query
    vector that would score an unambiguous, confident decision if scoring
    were ever reached, leaves the encoder-name check in load_intent_index as
    the only thing that can make predict_route return None here.
    """
    import numpy as np

    from src.model.intent_knn import INDEX_FILENAME, CanonicalExample, IntentIndex

    basis = np.eye(384, dtype=np.float32)
    examples, rows = [], []
    for route, axis in _AXIS.items():
        for position in range(12):
            examples.append(
                CanonicalExample(
                    f"{route}-{position}",
                    f"{route} {position}",
                    route,
                    (_MODULE[route],),
                )
            )
            rows.append(basis[axis])
    directory = tmp_path / "stale"
    IntentIndex(
        examples,
        np.stack(rows),
        "sentence-transformers/all-MiniLM-L6-v2",
        "sha256:x",
    ).save(directory / INDEX_FILENAME)

    # An exact match on the "search" axis: if the guard did not short-circuit
    # before this vector is ever used, decide() would return a maximally
    # confident, unambiguous decision, well clear of every default threshold.
    monkeypatch.setattr(
        ml_intent, "encode_texts", lambda texts: basis[_AXIS["search"]][None, :]
    )

    settings = AppSettings(intent_index_path=directory)

    assert ml_intent.predict_route("anything", settings=settings) is None


def test_both_gates_trip_sets_composite_on_the_returned_decision(tmp_path, monkeypatch):
    """decide() checks confidence before margin, but composite is computed off
    the margin alone (see intent_knn._is_composite) — so a query that trips
    both gates gets abstain_reason="confidence_below_threshold" (not
    "margin_below_threshold") together with composite=True. That is not a
    margin abstention, so predict_route must still return a decision for
    route_request's own confidence check to act on, and that decision must
    carry composite=True rather than silently dropping it.

    predict_route itself must NOT record a capture stage here: it only
    returns a decision (not None) on this path, and route_request always
    records its own single "intent_model" stage whenever it gets a decision
    back. If predict_route also recorded one here, a request that trips both
    gates would produce two stages with conflicting payloads.
    """
    monkeypatch.setattr(
        ml_intent,
        "encode_texts",
        lambda texts: np.array([[0.71, 0.0, 0.70]], dtype=np.float32),
    )
    recorded = []
    monkeypatch.setattr(
        ml_intent._capture,
        "record_stage",
        lambda stage, mechanism, detail: recorded.append((stage, detail)),
    )

    decision = ml_intent.predict_route(
        "anything", settings=_settings(tmp_path, intent_model_min_confidence=0.8)
    )

    assert decision is not None
    assert decision.confidence < decision.threshold
    assert decision.composite is True
    assert recorded == []
