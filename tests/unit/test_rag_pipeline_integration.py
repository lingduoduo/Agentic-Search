"""Integration coverage for retrieval, bounded tools, generation, and tracing."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from src.context import GroundedGenerationConfig
from src.context import LLMResponse
from src.context import ToolDescriptor
from src.context import ToolRequest
from src.context import ToolSafety
from src.context import build_context_bundle
from src.context.pipeline import answer_with_retrieval
from src.context.search import SearchResult


class Registry:
    def __init__(self, descriptors, results):
        self.descriptors = descriptors
        self.results = results
        self.calls = []

    def list_tools(self):
        return self.descriptors

    async def invoke(self, request):
        self.calls.append(request)
        value = self.results[request.tool_name]
        if isinstance(value, Exception):
            raise value
        return value


class Selector:
    def __init__(self, requests):
        self.requests = requests

    def select(self, query, tools):
        self.visible = tools
        return self.requests


class StructuredLLM:
    def complete(self, messages, **kwargs):
        return LLMResponse(
            '{"claims":[{"text":"FAISS is available","evidence_ids":["D1"]},'
            '{"text":"The status is healthy","evidence_ids":["T1"]}],'
            '"missing_information":[],"abstain":false}'
        )


def _context():
    return build_context_bundle(
        "FAISS status",
        [
            SearchResult(
                title="FAISS",
                contents="FAISS search is currently available.",
                score=1.0,
            )
        ],
    )


@pytest.mark.asyncio
async def test_pipeline_combines_retrieval_and_selected_read_only_tool(monkeypatch):
    registry = Registry(
        [ToolDescriptor("health", safety=ToolSafety.READ_ONLY)],
        {"health": {"status": "healthy"}},
    )
    selector = Selector([ToolRequest("health", {"token": "do-not-trace"})])
    monkeypatch.setattr(
        "src.context.pipeline.retrieve_context", lambda *a, **k: _async(_context())
    )

    result = await answer_with_retrieval(
        "FAISS status",
        llm=StructuredLLM(),
        tool_registry=registry,
        tool_selector=selector,
    )

    assert [item.id for item in result.tool_evidence] == ["T1"]
    assert result.tool_evidence[0].tool_name == "health"
    assert result.answer == "FAISS is available [D1] The status is healthy [T1]"
    assert result.verification_status.value == "verified"


@pytest.mark.asyncio
async def test_pipeline_rejects_unsafe_selection_and_falls_back_on_tool_failure(
    monkeypatch,
):
    registry = Registry(
        [
            ToolDescriptor("delete", safety=ToolSafety.SIDE_EFFECTING),
            ToolDescriptor("broken", safety=ToolSafety.READ_ONLY),
        ],
        {"delete": "unsafe", "broken": RuntimeError("raw-secret-output")},
    )
    selector = Selector([ToolRequest("delete"), ToolRequest("broken")])
    monkeypatch.setattr(
        "src.context.pipeline.retrieve_context", lambda *a, **k: _async(_context())
    )

    result = await answer_with_retrieval(
        "FAISS status",
        tool_registry=registry,
        tool_selector=selector,
    )

    assert [tool.name for tool in selector.visible] == ["broken"]
    assert [request.tool_name for request in registry.calls] == ["broken"]
    assert result.tool_evidence == []
    assert result.answer == "FAISS search is currently available. [D1]"


@pytest.mark.asyncio
async def test_pipeline_trace_contains_only_safe_summary_metadata(monkeypatch):
    spans = []

    class Tracer:
        @contextmanager
        def span(self, name, **attrs):
            spans.append((name, attrs))
            yield

    registry = Registry(
        [ToolDescriptor("health", safety=ToolSafety.READ_ONLY)],
        {"health": {"secret_body": "raw-secret-output"}},
    )
    selector = Selector([ToolRequest("health", {"token": "argument-secret"})])
    monkeypatch.setattr(
        "src.context.pipeline.retrieve_context", lambda *a, **k: _async(_context())
    )
    monkeypatch.setattr(
        "src.internal.observability.tracer.get_tracer", lambda: Tracer()
    )

    await answer_with_retrieval(
        "prompt-secret",
        tool_registry=registry,
        tool_selector=selector,
        grounded_generation=GroundedGenerationConfig(enabled=False),
    )

    rendered = repr(spans)
    assert "prompt-secret" not in rendered
    assert "raw-secret-output" not in rendered
    assert "argument-secret" not in rendered
    summary = dict(spans)["rag.summary"]
    assert summary["evidence_count"] == 2
    assert summary["evidence_types"] == "retrieval,tool"
    assert summary["tool_names"] == "health"
    assert summary["tool_statuses"] == "succeeded"
    assert "verification_status" in summary
    assert "confidence" in summary
    assert "abstained" in summary
    assert "retry_count" in summary


async def _async(value):
    return value
