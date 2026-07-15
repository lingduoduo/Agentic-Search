import asyncio
import json

import pytest

from src.context import (
    ToolDescriptor,
    ToolRequest,
    ToolSafety,
    collect_tool_evidence,
)


class Registry:
    def __init__(self, descriptors, results):
        self.descriptors = descriptors
        self.results = results
        self.calls = []

    def list_tools(self):
        return self.descriptors

    async def invoke(self, request):
        self.calls.append(request)
        result = self.results[request.tool_name]
        if isinstance(result, Exception):
            raise result
        if callable(result):
            return await result()
        return result


class Selector:
    def __init__(self, requests):
        self.requests = requests
        self.visible_tools = None

    def select(self, query, tools):
        self.visible_tools = tools
        return self.requests


@pytest.mark.asyncio
async def test_read_only_tools_supply_stable_normalized_evidence():
    registry = Registry(
        [ToolDescriptor("weather", "Current weather", ToolSafety.READ_ONLY)],
        {"weather": {"temperature": 72, "conditions": ["clear", "dry"]}},
    )
    selector = Selector([ToolRequest("weather", {"city": "Boston"})])

    evidence = await collect_tool_evidence("weather now", registry, selector)

    assert evidence[0].id == "T1"
    assert evidence[0].title == "Tool: weather"
    assert evidence[0].provenance == "tool"
    assert evidence[0].tool_name == "weather"
    assert evidence[0].text == json.dumps(
        {"conditions": ["clear", "dry"], "temperature": 72},
        sort_keys=True,
        separators=(",", ":"),
    )
    assert evidence[0].metadata == {"arguments": {"city": "Boston"}}


@pytest.mark.asyncio
async def test_unsafe_and_unknown_tools_are_not_visible_or_invoked():
    registry = Registry(
        [
            ToolDescriptor("read", safety=ToolSafety.READ_ONLY),
            ToolDescriptor("write", safety=ToolSafety.SIDE_EFFECTING),
            ToolDescriptor("mystery", safety=ToolSafety.UNSPECIFIED),
        ],
        {"read": "ok", "write": "bad", "mystery": "bad"},
    )
    selector = Selector(
        [ToolRequest("write"), ToolRequest("mystery"), ToolRequest("unknown")]
    )

    assert await collect_tool_evidence("query", registry, selector) == []
    assert [tool.name for tool in selector.visible_tools] == ["read"]
    assert registry.calls == []


@pytest.mark.asyncio
async def test_calls_are_bounded_before_invocation():
    descriptors = [
        ToolDescriptor(name, safety=ToolSafety.READ_ONLY) for name in ("a", "b", "c")
    ]
    registry = Registry(descriptors, {"a": 1, "b": 2, "c": 3})
    selector = Selector([ToolRequest(name) for name in ("a", "b", "c")])

    evidence = await collect_tool_evidence("query", registry, selector)

    assert [item.id for item in evidence] == ["T1", "T2"]
    assert [request.tool_name for request in registry.calls] == ["a", "b"]


@pytest.mark.asyncio
async def test_errors_and_timeouts_degrade_to_available_evidence():
    async def slow():
        await asyncio.sleep(0.05)
        return "late"

    descriptors = [
        ToolDescriptor(name, safety=ToolSafety.READ_ONLY)
        for name in ("broken", "slow", "good")
    ]
    registry = Registry(
        descriptors, {"broken": RuntimeError("boom"), "slow": slow, "good": "usable"}
    )
    selector = Selector([ToolRequest(name) for name in ("broken", "slow", "good")])

    evidence = await collect_tool_evidence(
        "query", registry, selector, max_calls=3, timeout_seconds=0.001
    )

    assert [(item.id, item.text) for item in evidence] == [("T1", '"usable"')]


@pytest.mark.asyncio
async def test_invalid_limits_reject_without_selecting_or_invoking():
    registry = Registry([], {})
    selector = Selector([])

    with pytest.raises(ValueError, match="max_calls"):
        await collect_tool_evidence("query", registry, selector, max_calls=-1)
    with pytest.raises(ValueError, match="timeout_seconds"):
        await collect_tool_evidence("query", registry, selector, timeout_seconds=0)
