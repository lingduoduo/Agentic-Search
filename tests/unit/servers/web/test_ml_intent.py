from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.internal.servers.web.ml_intent as ml_intent
from src.internal.configs import AppSettings
from src.internal.servers.web.intent_routing import RouteStrategy


class _StubPipeline:
    def __init__(self, intent, confidence):
        self._pred = SimpleNamespace(intent=intent, confidence=confidence)

    def predict_text(self, _query):
        return self._pred


def test_no_configured_path_avoids_importing_model_dependency(monkeypatch):
    real_import = builtins.__import__

    def fail_if_model_dependency_is_imported(name, *args, **kwargs):
        if name in {"torch", "src.model.intent_classifier"}:
            raise AssertionError(f"unexpected import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_if_model_dependency_is_imported)

    assert ml_intent.load_intent_model(settings=AppSettings()) is None


def test_explicit_settings_choose_artifact_instead_of_process_environment(monkeypatch):
    configured_path = Path("/configured/intent.pt")
    loaded_paths: list[str] = []

    class _Pipeline:
        @classmethod
        def load(cls, path: str):
            loaded_paths.append(path)
            return _StubPipeline("search", 0.91)

    import src.model.intent_classifier as intent_classifier

    monkeypatch.setenv("AGENTIC_SEARCH_INTENT_MODEL_PATH", "/wrong/intent.pt")
    monkeypatch.setattr(intent_classifier, "IntentPipeline", _Pipeline)
    monkeypatch.setattr(ml_intent, "_INTENT_MODELS", {})

    result = ml_intent.load_intent_model(
        settings=AppSettings(intent_model_path=configured_path)
    )

    assert result is not None
    assert loaded_paths == [str(configured_path.resolve())]


def test_incompatible_checkpoint_returns_none_and_caches_failure(monkeypatch):
    configured_path = Path("/configured/bad-intent.pt")
    attempts: list[str] = []

    class _Pipeline:
        @classmethod
        def load(cls, path: str):
            attempts.append(path)
            raise ValueError("unsupported checkpoint")

    import src.model.intent_classifier as intent_classifier

    monkeypatch.setattr(intent_classifier, "IntentPipeline", _Pipeline)
    monkeypatch.setattr(ml_intent, "_INTENT_MODELS", {})
    settings = AppSettings(intent_model_path=configured_path)

    assert ml_intent.load_intent_model(settings=settings) is None
    assert ml_intent.load_intent_model(settings=settings) is None
    assert attempts == [str(configured_path.resolve())]


def test_predict_route_maps_label_and_confidence(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE", "0.05")
    monkeypatch.setattr(
        ml_intent,
        "load_intent_model",
        lambda settings: _StubPipeline("search", 0.91),
    )
    result = ml_intent.predict_route(
        "find FAISS", settings=AppSettings(intent_model_min_confidence=0.73)
    )
    assert result == ml_intent.IntentModelDecision(
        strategy=RouteStrategy.SEARCH,
        confidence=0.91,
        threshold=0.73,
        latency_ms=pytest.approx(result.latency_ms),
    )


def test_predict_route_unknown_label_returns_none(monkeypatch):
    monkeypatch.setattr(
        ml_intent,
        "load_intent_model",
        lambda settings: _StubPipeline("purchase", 0.99),
    )
    assert ml_intent.predict_route("buy a thing", settings=AppSettings()) is None


def test_predict_route_swallows_predict_errors(monkeypatch):
    class _Boom:
        def predict_text(self, _q):
            raise RuntimeError("boom")

    monkeypatch.setattr(ml_intent, "load_intent_model", lambda settings: _Boom())
    assert ml_intent.predict_route("anything", settings=AppSettings()) is None
