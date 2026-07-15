"""Tests for observability tracer and pipeline instrumentation."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.internal.observability.tracer import (
    NoOpTracer,
    OtelTracer,
    get_tracer,
    set_tracer,
)


# ---------------------------------------------------------------------------
# NoOpTracer
# ---------------------------------------------------------------------------


def test_no_op_tracer_span_yields():
    tracer = NoOpTracer()
    reached = False
    with tracer.span("some.span", key="value"):
        reached = True
    assert reached


def test_no_op_tracer_span_nested():
    tracer = NoOpTracer()
    order: list[str] = []
    with tracer.span("outer"):
        order.append("in_outer")
        with tracer.span("inner"):
            order.append("in_inner")
    assert order == ["in_outer", "in_inner"]


# ---------------------------------------------------------------------------
# get_tracer / set_tracer
# ---------------------------------------------------------------------------


def test_get_tracer_default_is_noop():
    import src.internal.observability.tracer as mod

    original = mod._tracer
    try:
        mod._tracer = NoOpTracer()
        assert isinstance(get_tracer(), NoOpTracer)
    finally:
        mod._tracer = original


def test_set_tracer_replaces_instance():
    import src.internal.observability.tracer as mod

    original = mod._tracer
    try:
        replacement = NoOpTracer()
        set_tracer(replacement)
        assert get_tracer() is replacement
    finally:
        mod._tracer = original


# ---------------------------------------------------------------------------
# OtelTracer — happy path (opentelemetry mocked)
# ---------------------------------------------------------------------------


def _make_otel_mocks() -> tuple[MagicMock, MagicMock]:
    """Return (mock_otel_module, mock_span)."""
    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)

    mock_tracer_obj = MagicMock()
    mock_tracer_obj.start_as_current_span.return_value = mock_span

    mock_trace_module = MagicMock()
    mock_trace_module.get_tracer.return_value = mock_tracer_obj

    mock_otel = MagicMock()
    mock_otel.trace = mock_trace_module

    return mock_otel, mock_span


def test_otel_tracer_creates_span_with_name():
    mock_otel, mock_span = _make_otel_mocks()
    with patch.dict(
        sys.modules,
        {"opentelemetry": mock_otel, "opentelemetry.trace": mock_otel.trace},
    ):
        tracer = OtelTracer()
        with tracer.span("rag.retrieve", top_k=5):
            pass

    mock_otel.trace.get_tracer.return_value.start_as_current_span.assert_called_once_with(
        "rag.retrieve"
    )


def test_otel_tracer_sets_string_attribute():
    mock_otel, mock_span = _make_otel_mocks()
    with patch.dict(
        sys.modules,
        {"opentelemetry": mock_otel, "opentelemetry.trace": mock_otel.trace},
    ):
        tracer = OtelTracer()
        with tracer.span("rag.query", query="What is FAISS?"):
            pass

    mock_span.set_attribute.assert_any_call("query", "What is FAISS?")


def test_otel_tracer_sets_int_attribute():
    mock_otel, mock_span = _make_otel_mocks()
    with patch.dict(
        sys.modules,
        {"opentelemetry": mock_otel, "opentelemetry.trace": mock_otel.trace},
    ):
        tracer = OtelTracer()
        with tracer.span("rag.query", top_k=10):
            pass

    mock_span.set_attribute.assert_any_call("top_k", 10)


def test_otel_tracer_skips_none_attributes():
    mock_otel, mock_span = _make_otel_mocks()
    with patch.dict(
        sys.modules,
        {"opentelemetry": mock_otel, "opentelemetry.trace": mock_otel.trace},
    ):
        tracer = OtelTracer()
        with tracer.span("rag.query", query=None, top_k=5):
            pass

    called_keys = [c.args[0] for c in mock_span.set_attribute.call_args_list]
    assert "query" not in called_keys
    assert "top_k" in called_keys


def test_otel_tracer_raises_import_error_without_opentelemetry():
    with patch.dict(sys.modules, {"opentelemetry": None}):
        with pytest.raises((ImportError, ModuleNotFoundError)):
            OtelTracer()


# ---------------------------------------------------------------------------
# setup.py — ImportError paths
# ---------------------------------------------------------------------------


def test_setup_phoenix_raises_without_opentelemetry():
    from src.internal.observability.setup import setup_phoenix

    with patch.dict(sys.modules, {"opentelemetry": None}):
        with pytest.raises((ImportError, ModuleNotFoundError)):
            setup_phoenix()


def test_setup_langfuse_raises_without_opentelemetry():
    from src.internal.observability.setup import setup_langfuse

    with patch.dict(sys.modules, {"opentelemetry": None}):
        with pytest.raises((ImportError, ModuleNotFoundError)):
            setup_langfuse(public_key="pk", secret_key="sk")


# ---------------------------------------------------------------------------
# pipeline.py instrumentation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_answer_with_retrieval_calls_tracer_spans():
    """answer_with_retrieval must open rag.query > rag.retrieve > rag.generate spans."""
    from src.context.pipeline import answer_with_retrieval

    span_names: list[str] = []

    class RecordingTracer:
        from contextlib import contextmanager
        from typing import Generator

        @contextmanager
        def span(self, name: str, **attrs: object) -> Generator[None, None, None]:
            span_names.append(name)
            yield

    mock_context = MagicMock()
    mock_context.documents = []

    mock_result = MagicMock()

    with (
        patch(
            "src.context.pipeline.retrieve_context",
            AsyncMock(return_value=mock_context),
        ),
        patch("src.context.pipeline.generate_answer", return_value=mock_result),
        patch(
            "src.internal.observability.tracer.get_tracer",
            return_value=RecordingTracer(),
        ),
    ):
        result = await answer_with_retrieval("What is FAISS?")

    assert result is mock_result
    assert "rag.query" in span_names
    assert "rag.retrieve" in span_names
    assert "rag.generate" in span_names


@pytest.mark.asyncio
async def test_answer_with_retrieval_omits_query_text_from_span():
    """The rag.query span records bounded metadata, never prompt text."""
    from src.context.pipeline import answer_with_retrieval

    recorded_attrs: dict[str, object] = {}

    class AttrRecordingTracer:
        from contextlib import contextmanager
        from typing import Generator

        @contextmanager
        def span(self, name: str, **attrs: object) -> Generator[None, None, None]:
            if name == "rag.query":
                recorded_attrs.update(attrs)
            yield

    mock_context = MagicMock()
    mock_context.documents = []

    with (
        patch(
            "src.context.pipeline.retrieve_context",
            AsyncMock(return_value=mock_context),
        ),
        patch("src.context.pipeline.generate_answer", return_value=MagicMock()),
        patch(
            "src.internal.observability.tracer.get_tracer",
            return_value=AttrRecordingTracer(),
        ),
    ):
        await answer_with_retrieval("What is FAISS?", top_k=7)

    assert "query" not in recorded_attrs
    assert recorded_attrs.get("top_k") == 7
