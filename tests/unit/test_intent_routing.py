# tests/unit/test_intent_routing.py
import asyncio
import json
import json as _json
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from src.agents.core.base import AgentLoopOutput
from src.internal.configs import AppSettings
from src.internal.servers.web import ml_intent
from src.internal.servers.web import request_capture as rc
from src.internal.servers.web.intent_routing import (
    RouteStrategy,
    _infer_intent_from_output,
    route_request,
)
from src.internal.tools.routing_tools import (
    build_rag_routing_tool,
    build_search_routing_tool,
)
from src.internal.tools import ToolEffect
from src.model.intent_encoder import DEFAULT_ENCODER
from src.model.intent_knn import INDEX_FILENAME, CanonicalExample, IntentIndex


def _make_output(
    action_trace: str | None = None, final_answer: str | None = None
) -> AgentLoopOutput:
    return AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        action_trace=action_trace,
        final_answer=final_answer,
    )


# --- _infer_intent_from_output ---


def _trace(tool_name: str) -> str:
    return json.dumps(
        {
            "tool_name": tool_name,
            "status": "completed",
            "result": "ok",
            "performance": {},
            "error_code": None,
            "error_message": None,
            "optimization_suggestions": [],
            "retry_count": 0,
        }
    )


def test_infer_intent_corpus_search_tool():
    output = _make_output(action_trace=_trace("search"))
    assert _infer_intent_from_output(output) == "search"


def test_infer_intent_rag_routing_tool():
    output = _make_output(action_trace=_trace("rag_routing_tool"))
    assert _infer_intent_from_output(output) == "chat"


def test_infer_intent_mcp_tool():
    output = _make_output(action_trace=_trace("custom_api"))
    assert _infer_intent_from_output(output) == "tool"


def test_infer_intent_no_trace_defaults_to_chat():
    output = _make_output(action_trace=None)
    assert _infer_intent_from_output(output) == "chat"


def test_infer_intent_malformed_trace_defaults_to_chat():
    output = _make_output(action_trace="not json")
    assert _infer_intent_from_output(output) == "chat"


# --- routing tool builder tests ---


def test_build_search_routing_tool_schema():
    tool = build_search_routing_tool(
        search_url="http://localhost:8001/retrieve", top_k=3
    )
    assert tool.schema.name == "search"
    assert tool.effect is ToolEffect.READ_ONLY
    assert "query" in tool.schema.parameters.get("properties", {})


def test_build_rag_routing_tool_schema():
    tool = build_rag_routing_tool(
        llm=None, search_url="http://localhost:8001/retrieve", top_k=3
    )
    assert tool.schema.name == "rag_routing_tool"
    assert tool.effect is ToolEffect.READ_ONLY
    assert "query" in tool.schema.parameters.get("properties", {})


def test_search_routing_tool_returns_json():
    from src.internal.tools.search import SearchPage

    fake_pages = [SearchPage(title="Doc A", summary="Content A", url="http://a.com")]

    async def run():
        with patch(
            "src.internal.tools.routing_tools.search_tool",
            new=AsyncMock(return_value=fake_pages),
        ):
            tool = build_search_routing_tool(
                search_url="http://localhost:8001/retrieve", top_k=3
            )
            response_text, raw, _meta = await tool.execute(
                "default", {"query": "FAISS"}
            )
        return response_text

    result = asyncio.run(run())
    data = _json.loads(result)
    assert isinstance(data, list)
    assert data[0]["title"] == "Doc A"


def test_rag_routing_tool_returns_answer():
    from src.context.models import (
        AnswerGenerationResult,
        PromptBundle,
        SearchContextBundle,
    )

    fake_context = SearchContextBundle(query="q", documents=[])
    fake_prompt = PromptBundle(system="", user="", messages=[])
    fake_result = AnswerGenerationResult(
        answer="42",
        citations=["[D1]"],
        context=fake_context,
        prompt=fake_prompt,
    )

    async def run():
        with patch(
            "src.context.answer_with_retrieval", new=AsyncMock(return_value=fake_result)
        ):
            tool = build_rag_routing_tool(
                llm=None, search_url="http://localhost:8001/retrieve", top_k=3
            )
            response_text, raw, _meta = await tool.execute(
                "default", {"query": "What is the answer?"}
            )
        return response_text

    result = asyncio.run(run())
    data = _json.loads(result)
    assert data["answer"] == "42"


def test_route_request_is_unchanged_by_a_none_returning_model(monkeypatch):
    """A margin abstention must look exactly like having no model at all."""
    from src.internal.servers.web import intent_routing

    monkeypatch.setattr(intent_routing, "predict_route", lambda q, settings=None: None)
    calls = []

    class _LLM:
        def complete(self, messages, temperature=0.0):
            calls.append(messages)
            return "search"

    decision = intent_routing.route_request(
        "where does the reranker timeout live",
        llm=_LLM(),
        explicit_source=False,
    )

    assert decision.strategy is intent_routing.RouteStrategy.SEARCH
    assert calls, "the LLM classifier must still be consulted"


# --- route_request end-to-end: exactly one intent_model capture stage ---

_AXIS = {"search": 0, "chat": 1, "tool": 2}
_MODULE = {"search": "lookup_fact", "chat": "explain", "tool": "schedule"}

# Deliberately not "?" and more than 3 words with no tool/search/chat cue, so
# _regex_route defers and the query actually reaches predict_route instead of
# being decided deterministically at cascade step 2.
_MODEL_STAGE_QUERY = "the vendor contract renewal terms"


def _write_routing_index(tmp_path):
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


class _ChatLLM:
    def complete(self, messages, temperature=0.0):
        return "chat"


@pytest.mark.parametrize(
    "vector, min_confidence, expect_composite",
    [
        # Confidence-only abstention: clear margin, low absolute confidence.
        pytest.param([0.9950, 0.1005, 0.0], 1.0, False, id="confidence_only"),
        # Margin abstention: confidence clears the (low) bar, but the top two
        # routes tie, failing the margin gate. predict_route returns the
        # decision with abstain_reason set, and route_request records the one
        # stage -- previously predict_route returned None and recorded its own,
        # which is what kept this deferral out of production telemetry.
        pytest.param([0.707, 0.707, 0.0], 0.5, False, id="margin_only"),
        # Both gates trip together: confidence 0.71 < 0.8, and margin 0.01
        # against the "tool" runner-up (whose only module is an action) is
        # composite.
        pytest.param([0.71, 0.0, 0.70], 0.8, True, id="confidence_and_composite"),
    ],
)
def test_route_request_records_exactly_one_intent_model_stage(
    tmp_path, monkeypatch, vector, min_confidence, expect_composite
):
    ml_intent._INTENT_INDEXES.clear()
    monkeypatch.setattr(
        ml_intent, "encode_texts", lambda texts: np.array([vector], dtype=np.float32)
    )
    settings = AppSettings(
        intent_index_path=_write_routing_index(tmp_path),
        intent_model_min_confidence=min_confidence,
        intent_min_route_margin=0.05,
        intent_min_module_score=0.4,
    )
    token = rc.start_capture("r", _MODEL_STAGE_QUERY)
    try:
        route_request(
            _MODEL_STAGE_QUERY,
            llm=_ChatLLM(),
            explicit_source=False,
            settings=settings,
        )
        model_stages = [s for s in rc.active().stages if s.stage == "intent_model"]

        assert len(model_stages) == 1
        assert model_stages[0].payload["composite"] is expect_composite
    finally:
        rc.reset_capture(token)
        ml_intent._INTENT_INDEXES.clear()


def _route_with_telemetry(tmp_path, monkeypatch, vector, **overrides):
    """Route one query with **no capture active** — production conditions.

    Starting no capture is the point: ``request_capture`` only records under
    the debug panels, so anything asserted here is reaching the telemetry dict
    that is actually persisted with the session.
    """
    ml_intent._INTENT_INDEXES.clear()
    monkeypatch.setattr(
        ml_intent, "encode_texts", lambda texts: np.array([vector], dtype=np.float32)
    )
    settings = AppSettings(
        intent_index_path=_write_routing_index(tmp_path),
        intent_min_route_margin=0.05,
        intent_min_module_score=0.4,
        **overrides,
    )
    telemetry: dict = {}
    try:
        decision = route_request(
            _MODEL_STAGE_QUERY,
            llm=_ChatLLM(),
            explicit_source=False,
            settings=settings,
            telemetry=telemetry,
        )
    finally:
        ml_intent._INTENT_INDEXES.clear()
    return decision, telemetry


def test_margin_abstention_is_distinguishable_in_production_telemetry(
    tmp_path, monkeypatch
):
    """The gate that does all the abstaining under e5 must be countable.

    The confidence floor cannot fire under this encoder (in-scope scores sit
    far above it), so every deferral the router makes is a margin abstention.
    It previously reached ``None`` inside predict_route, indistinguishable from
    "no index configured", and the only record of it was a capture stage that
    runs solely under the debug panels. "How often does the router defer, and
    why" was therefore unanswerable from production data.
    """
    # Confidence clears the default floor; the top two routes tie on margin.
    _, telemetry = _route_with_telemetry(
        tmp_path, monkeypatch, [0.707, 0.707, 0.0], intent_model_min_confidence=0.5
    )

    assert telemetry["route_abstained"] is True
    assert telemetry["route_fallback_reason"] == "margin_below_threshold"
    # Not the confidence label: these are different failures and conflating
    # them would make the counts useless.
    assert telemetry["route_fallback_reason"] != "model_below_threshold"


def test_confidence_abstention_keeps_its_existing_production_label(
    tmp_path, monkeypatch
):
    """The confidence path must be untouched by the margin path's plumbing.

    ``decide()`` labels this one "confidence_below_threshold" internally, but
    only the margin reason is propagated onto the decision, so route_request
    still derives and reports ``model_below_threshold`` exactly as before.
    """
    _, telemetry = _route_with_telemetry(
        tmp_path, monkeypatch, [0.9950, 0.1005, 0.0], intent_model_min_confidence=1.0
    )

    assert telemetry["route_abstained"] is True
    assert telemetry["route_fallback_reason"] == "model_below_threshold"


def test_modules_and_composite_reach_production_telemetry(tmp_path, monkeypatch):
    """Both were recorded only in the dev-only capture stage.

    The composite flag exists solely so a future plan-aware router can be
    designed against measured data; observable only under a debug panel, it
    gathered none.
    """
    _, telemetry = _route_with_telemetry(
        tmp_path, monkeypatch, [1.0, 0.0, 0.0], intent_model_min_confidence=0.1
    )

    assert telemetry["route_abstained"] is False
    assert telemetry["route_modules"] == list(telemetry["route_modules"])
    assert telemetry["route_modules"], "a served route must carry its modules"
    assert telemetry["route_composite"] is False


def test_margin_abstention_still_defers_to_the_classifier(tmp_path, monkeypatch):
    """Observability only. The routing behavior must not have moved.

    ``_ChatLLM`` answers "chat", so a decision of CHAT proves the abstention
    fell through to the classifier rather than being served as the model's own
    (search) answer.
    """
    decision, telemetry = _route_with_telemetry(
        tmp_path, monkeypatch, [0.707, 0.707, 0.0], intent_model_min_confidence=0.5
    )

    assert decision.strategy is RouteStrategy.CHAT
    assert telemetry["route_mechanism"] == "classifier"
