"""Serving adapter behavior, exercised without an encoder."""

from pathlib import Path

import numpy as np
import pytest

from src.internal.configs import AppSettings
from src.internal.servers.web import ml_intent
from src.internal.servers.web.intent_routing import RouteStrategy
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
    IntentIndex(examples, np.stack(rows), "test-encoder", "sha256:x").save(
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


def test_unreadable_index_defers_and_is_not_retried(tmp_path, monkeypatch, caplog):
    settings = AppSettings(intent_index_path=tmp_path / "absent")

    assert ml_intent.predict_route("anything", settings=settings) is None
    assert ml_intent.predict_route("anything", settings=settings) is None


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


def test_intent_model_decision_has_no_composite_field():
    """It could only ever be False: composite implies a margin abstention."""
    assert not hasattr(
        ml_intent.IntentModelDecision("chat", 1.0, 0.5, 1.0), "composite"
    )
