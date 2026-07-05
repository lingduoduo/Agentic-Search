from __future__ import annotations

from src.context.models import ChatMessage
from src.internal.servers.web import request_capture as rc
from src.internal.servers.web.intent_routing import route_query


class _FakeLLM:
    def complete(self, messages: list[ChatMessage], **_) -> str:
        return "search"


def _intent_stages():
    return [s for s in rc.active().stages if s.stage == "intent"]


def test_route_query_emits_regex_intent_stage():
    token = rc.start_capture("r", "What is FAISS?")
    try:
        strategy = route_query(
            "What is FAISS?",
            llm=_FakeLLM(),
            has_local_model=True,
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
            has_local_model=True,
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
        route_query(
            "anything at all", llm=None, has_local_model=False, explicit_source=True
        )
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
            has_local_model=False,
            explicit_source=False,
        )
        stages = _intent_stages()
        assert len(stages) == 1
        assert stages[0].payload["mechanism"] == "rule_based"
    finally:
        rc.reset_capture(token)


def test_route_query_no_capture_does_not_raise():
    # With no active capture the emit is a silent no-op.
    route_query("q", llm=_FakeLLM(), has_local_model=True, explicit_source=False)
