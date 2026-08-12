from __future__ import annotations

from src.context.models import ChatMessage
from src.internal.servers.web import request_capture as rc
from src.internal.servers.web import intent_routing as ir
from src.internal.servers.web.intent_routing import route_query
from src.internal.servers.web.ml_intent import IntentModelDecision


class _FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[ChatMessage], **_) -> str:
        self.calls += 1
        return "search"


def _intent_stages():
    return [s for s in rc.active().stages if s.stage == "intent"]


def test_route_query_emits_regex_intent_stage():
    token = rc.start_capture("r", "What is FAISS?")
    try:
        strategy = route_query(
            "What is FAISS?",
            llm=_FakeLLM(),
            explicit_source=False,
        )
        stages = _intent_stages()
        assert len(stages) == 1
        assert stages[0].label == "regex"
        assert stages[0].payload["mechanism"] == "regex"
        assert stages[0].payload["strategy"] == strategy.value  # "chat"
    finally:
        rc.reset_capture(token)


def test_route_query_emits_classifier_intent_stage_with_detail():
    # A phrase _regex_route defers on → classifier path, preserving prompt/raw_label.
    token = rc.start_capture("r", "the procurement approval flow")
    try:
        route_query(
            "the procurement approval flow",
            llm=_FakeLLM(),
            explicit_source=False,
        )
        stages = _intent_stages()
        assert len(stages) == 1
        assert stages[0].label == "classifier"
        assert stages[0].payload["mechanism"] == "classifier"
        assert stages[0].payload["raw_label"] == "search"
        assert "prompt" in stages[0].payload
    finally:
        rc.reset_capture(token)


def test_route_query_emits_explicit_source_intent_stage():
    token = rc.start_capture("r", "anything at all")
    try:
        route_query("anything at all", llm=None, explicit_source=True)
        stages = _intent_stages()
        assert len(stages) == 1
        assert stages[0].payload["mechanism"] == "explicit_source"
        assert stages[0].payload["strategy"] == "search"
    finally:
        rc.reset_capture(token)


def test_route_query_emits_rule_based_intent_stage_without_llm():
    token = rc.start_capture("r", "the procurement approval flow")
    try:
        route_query(
            "the procurement approval flow",
            llm=None,
            explicit_source=False,
        )
        stages = _intent_stages()
        assert len(stages) == 1
        assert stages[0].payload["mechanism"] == "rule_based"
    finally:
        rc.reset_capture(token)


def test_route_query_no_capture_does_not_raise():
    # With no active capture the emit is a silent no-op.
    route_query("q", llm=_FakeLLM(), explicit_source=False)


def test_confident_model_records_evaluation_and_final_intent(monkeypatch):
    monkeypatch.setattr(
        ir,
        "predict_route",
        lambda query, *, settings=None: IntentModelDecision(
            strategy=ir.RouteStrategy.SEARCH,
            confidence=0.91,
            threshold=0.73,
            latency_ms=2.5,
        ),
    )
    llm = _FakeLLM()
    token = rc.start_capture("r", "the vendor contract renewal terms")
    try:
        strategy = route_query(
            "the vendor contract renewal terms",
            llm=llm,
            explicit_source=False,
        )
        model_stages = [s for s in rc.active().stages if s.stage == "intent_model"]
        intent_stages = _intent_stages()

        assert strategy is ir.RouteStrategy.SEARCH
        assert len(model_stages) == 1
        assert model_stages[0].label == "evaluation"
        assert model_stages[0].payload == {
            "predicted_intent": "search",
            "confidence": 0.91,
            "threshold": 0.73,
            "abstained": False,
            "fallback_reason": None,
            "latency_ms": 2.5,
        }
        assert len(intent_stages) == 1
        assert intent_stages[0].label == "model"
        assert intent_stages[0].payload["mechanism"] == "model"
        assert intent_stages[0].payload["predicted_intent"] == "search"
        assert intent_stages[0].payload["confidence"] == 0.91
        assert intent_stages[0].payload["threshold"] == 0.73
        assert intent_stages[0].payload["latency_ms"] == 2.5
        assert llm.calls == 0
    finally:
        rc.reset_capture(token)


def test_abstaining_model_records_evaluation_and_classifier_fallback(monkeypatch):
    monkeypatch.setattr(
        ir,
        "predict_route",
        lambda query, *, settings=None: IntentModelDecision(
            strategy=ir.RouteStrategy.TOOL,
            confidence=0.42,
            threshold=0.8,
            latency_ms=3.25,
        ),
    )
    token = rc.start_capture("r", "the vendor contract renewal terms")
    try:
        strategy = route_query(
            "the vendor contract renewal terms",
            llm=_FakeLLM(),
            explicit_source=False,
        )
        model_stages = [s for s in rc.active().stages if s.stage == "intent_model"]
        intent_stages = _intent_stages()

        assert strategy is ir.RouteStrategy.SEARCH
        assert model_stages[0].payload["predicted_intent"] == "tool"
        assert model_stages[0].payload["abstained"] is True
        assert model_stages[0].payload["fallback_reason"] == "model_below_threshold"
        assert intent_stages[0].label == "classifier"
        assert intent_stages[0].payload["fallback_reason"] == "model_below_threshold"
    finally:
        rc.reset_capture(token)


def test_missing_model_does_not_fabricate_prediction_detail(monkeypatch):
    monkeypatch.setattr(ir, "predict_route", lambda query, *, settings=None: None)
    token = rc.start_capture("r", "the vendor contract renewal terms")
    try:
        route_query(
            "the vendor contract renewal terms",
            llm=_FakeLLM(),
            explicit_source=False,
        )

        assert not [s for s in rc.active().stages if s.stage == "intent_model"]
        payload = _intent_stages()[0].payload
        assert "predicted_intent" not in payload
        assert "confidence" not in payload
        assert "fallback_reason" not in payload
    finally:
        rc.reset_capture(token)
