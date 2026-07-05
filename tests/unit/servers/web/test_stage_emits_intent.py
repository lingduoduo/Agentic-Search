from __future__ import annotations

from src.context.models import ChatMessage
from src.internal.servers.web import request_capture as rc
from src.internal.servers.web.intent_routing import classify_route


class _FakeLLM:
    def complete(self, messages: list[ChatMessage], **_) -> str:
        return "search"


def test_classify_route_emits_intent_stage_when_capturing():
    token = rc.start_capture("r", "vector database benchmarks")
    try:
        classify_route("vector database benchmarks", _FakeLLM())
        cap = rc.active()
        intent = [s for s in cap.stages if s.stage == "intent"]
        assert intent, "expected an intent stage"
        assert intent[0].payload["raw_label"] == "search"
        assert "prompt" in intent[0].payload
    finally:
        rc.reset_capture(token)


def test_classify_route_no_capture_does_not_raise():
    # With no active capture the emit is a silent no-op.
    classify_route("q", _FakeLLM())
