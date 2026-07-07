import src.internal.servers.web.ml_intent as ml_intent
from src.internal.servers.web.intent_routing import RouteStrategy
from src.model.intent_classifier import IntentPrediction


class _StubPipeline:
    def __init__(self, intent, confidence):
        self._pred = IntentPrediction(intent=intent, confidence=confidence)

    def predict_text(self, _query):
        return self._pred


def test_min_confidence_default_and_override(monkeypatch):
    monkeypatch.delenv("AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE", raising=False)
    assert ml_intent.intent_min_confidence() == 0.6
    monkeypatch.setenv("AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE", "0.75")
    assert ml_intent.intent_min_confidence() == 0.75


def test_no_model_path_returns_none(monkeypatch):
    monkeypatch.delenv("AGENTIC_SEARCH_INTENT_MODEL_PATH", raising=False)
    monkeypatch.setattr(ml_intent, "_INTENT_MODEL", None)
    assert ml_intent.load_intent_model() is None
    assert ml_intent.predict_route("find FAISS") is None


def test_predict_route_maps_label_and_confidence(monkeypatch):
    monkeypatch.setattr(
        ml_intent, "load_intent_model", lambda: _StubPipeline("search", 0.91)
    )
    result = ml_intent.predict_route("find FAISS")
    assert result == (RouteStrategy.SEARCH, 0.91)


def test_predict_route_unknown_label_returns_none(monkeypatch):
    monkeypatch.setattr(
        ml_intent, "load_intent_model", lambda: _StubPipeline("purchase", 0.99)
    )
    assert ml_intent.predict_route("buy a thing") is None


def test_predict_route_swallows_predict_errors(monkeypatch):
    class _Boom:
        def predict_text(self, _q):
            raise RuntimeError("boom")

    monkeypatch.setattr(ml_intent, "load_intent_model", lambda: _Boom())
    assert ml_intent.predict_route("anything") is None
