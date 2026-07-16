import asyncio
import json
from collections.abc import Mapping

import pytest

from src.context import (
    ToolDescriptor,
    ToolRequest,
    ToolSafety,
    collect_tool_evidence,
)
from src.context.tool_evidence import _encode_bounded


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
@pytest.mark.parametrize(
    "descriptors", [RuntimeError("discovery failed"), None, [object()]]
)
async def test_discovery_failures_degrade_to_no_tool_evidence(descriptors):
    class DiscoveryRegistry:
        def list_tools(self):
            if isinstance(descriptors, Exception):
                raise descriptors
            return descriptors

        async def invoke(self, request):
            raise AssertionError("malformed discovery must never invoke a tool")

    selector = Selector([ToolRequest("read")])

    assert await collect_tool_evidence("query", DiscoveryRegistry(), selector) == []
    assert selector.visible_tools is None


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
    assert evidence[0].metadata == {}


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
async def test_conflicting_duplicate_registration_is_rejected():
    registry = Registry(
        [
            ToolDescriptor("shared", safety=ToolSafety.READ_ONLY),
            ToolDescriptor("shared", safety=ToolSafety.SIDE_EFFECTING),
        ],
        {"shared": "must not run"},
    )
    selector = Selector([ToolRequest("shared")])

    assert await collect_tool_evidence("query", registry, selector) == []
    assert selector.visible_tools == []
    assert registry.calls == []


@pytest.mark.asyncio
async def test_same_safety_duplicate_registration_is_rejected():
    registry = Registry(
        [
            ToolDescriptor("duplicate", safety=ToolSafety.READ_ONLY),
            ToolDescriptor("duplicate", safety=ToolSafety.READ_ONLY),
        ],
        {"duplicate": "must not run"},
    )
    selector = Selector([ToolRequest("duplicate")])

    assert await collect_tool_evidence("query", registry, selector) == []
    assert selector.visible_tools == []
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
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_selector_exceptions_degrade_to_no_tool_evidence(asynchronous):
    registry = Registry(
        [ToolDescriptor("read", safety=ToolSafety.READ_ONLY)], {"read": "unused"}
    )

    class BrokenSelector:
        if asynchronous:

            async def select(self, query, tools):
                raise RuntimeError("selector secret")

        else:

            def select(self, query, tools):
                raise RuntimeError("selector secret")

    assert await collect_tool_evidence("query", registry, BrokenSelector()) == []
    assert registry.calls == []


@pytest.mark.asyncio
async def test_hanging_selector_times_out_to_no_tool_evidence():
    registry = Registry(
        [ToolDescriptor("read", safety=ToolSafety.READ_ONLY)], {"read": "unused"}
    )

    class HangingSelector:
        async def select(self, query, tools):
            await asyncio.sleep(1)
            return [ToolRequest("read")]

    evidence = await collect_tool_evidence(
        "query", registry, HangingSelector(), timeout_seconds=0.001
    )

    assert evidence == []
    assert registry.calls == []


@pytest.mark.asyncio
async def test_selection_iteration_is_bounded_before_invocation():
    registry = Registry(
        [ToolDescriptor("read", safety=ToolSafety.READ_ONLY)], {"read": "unused"}
    )
    traversed = 0

    def overlong_selection():
        nonlocal traversed
        while True:
            traversed += 1
            yield ToolRequest("unknown")

    selector = Selector(overlong_selection())

    assert await collect_tool_evidence("query", registry, selector, max_calls=2) == []
    assert traversed <= 8
    assert registry.calls == []


@pytest.mark.asyncio
async def test_selection_iteration_failure_degrades_to_no_tool_evidence():
    registry = Registry(
        [ToolDescriptor("read", safety=ToolSafety.READ_ONLY)], {"read": "unused"}
    )

    def broken_selection():
        yield ToolRequest("unknown")
        raise RuntimeError("iteration secret")

    selector = Selector(broken_selection())

    assert await collect_tool_evidence("query", registry, selector) == []
    assert registry.calls == []


@pytest.mark.asyncio
async def test_tool_arguments_are_not_copied_into_evidence_metadata():
    registry = Registry(
        [ToolDescriptor("lookup", safety=ToolSafety.READ_ONLY)], {"lookup": "safe"}
    )
    selector = Selector([ToolRequest("lookup", {"api_key": "secret-value"})])

    evidence = await collect_tool_evidence("query", registry, selector)

    assert evidence[0].metadata == {}
    assert "secret-value" not in repr(evidence[0])
    assert isinstance(registry.calls[0].arguments, Mapping)
    assert registry.calls[0].arguments == {"api_key": "secret-value"}


def test_tool_request_deeply_copies_arguments_into_immutable_mappings():
    original = {"filters": {"tenant": "first"}, "items": [1, {"key": "value"}]}

    request = ToolRequest("lookup", original)
    original["filters"]["tenant"] = "changed"
    original["items"][1]["key"] = "changed"

    assert isinstance(request.arguments, Mapping)
    assert request.arguments["filters"]["tenant"] == "first"
    assert request.arguments["items"][1]["key"] == "value"
    with pytest.raises(TypeError):
        request.arguments["new"] = "value"


@pytest.mark.asyncio
async def test_invalid_limits_reject_without_selecting_or_invoking():
    registry = Registry([], {})
    selector = Selector([])

    with pytest.raises(ValueError, match="max_calls"):
        await collect_tool_evidence("query", registry, selector, max_calls=-1)
    with pytest.raises(ValueError, match="timeout_seconds"):
        await collect_tool_evidence("query", registry, selector, timeout_seconds=0)
    with pytest.raises(ValueError, match="max_result_chars"):
        await collect_tool_evidence("query", registry, selector, max_result_chars=0)
    with pytest.raises(ValueError, match="max_result_chars"):
        await collect_tool_evidence("query", registry, selector, max_result_chars=-1)


@pytest.mark.asyncio
async def test_oversized_tool_result_is_rejected_and_reported_failed():
    registry = Registry(
        [ToolDescriptor("bulk", "Bulk data", ToolSafety.READ_ONLY)],
        {"bulk": {"blob": "x" * 5000}},
    )
    selector = Selector([ToolRequest("bulk")])
    statuses: list[tuple[str, str]] = []

    evidence = await collect_tool_evidence(
        "query",
        registry,
        selector,
        max_result_chars=1000,
        status_callback=lambda name, status: statuses.append((name, status)),
    )

    assert evidence == []
    assert statuses == [("bulk", "failed")]


@pytest.mark.asyncio
async def test_result_at_exactly_the_cap_is_accepted_and_one_over_is_rejected():
    result = {"value": "x" * 10}
    exact = json.dumps(
        result, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    registry = Registry(
        [ToolDescriptor("probe", "Probe", ToolSafety.READ_ONLY)], {"probe": result}
    )
    selector = Selector([ToolRequest("probe")])

    accepted = await collect_tool_evidence(
        "query", registry, selector, max_result_chars=len(exact)
    )
    assert [item.text for item in accepted] == [exact]

    rejected = await collect_tool_evidence(
        "query", registry, selector, max_result_chars=len(exact) - 1
    )
    assert rejected == []


@pytest.mark.asyncio
async def test_under_cap_encoding_is_byte_identical_to_json_dumps():
    result = {"z": 1.5, "a": "café", "n": [1, 2, {"k": None}]}
    registry = Registry(
        [ToolDescriptor("probe", "Probe", ToolSafety.READ_ONLY)], {"probe": result}
    )
    selector = Selector([ToolRequest("probe")])

    evidence = await collect_tool_evidence("query", registry, selector)

    assert evidence[0].text == json.dumps(
        result, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def test_bounded_encoding_aborts_before_walking_the_whole_structure():
    # Observing early abort requires calling the encoder directly: through
    # collect_tool_evidence both outcomes collapse to the same "failed" status.
    # A sentinel json cannot serialize sits past the cap. Aborting early raises
    # ValueError without ever reaching it; encoding the whole structure would
    # instead raise TypeError. This test fails if _one_shot=True is introduced.
    class Unserializable:
        pass

    result = ["y" * 1000 for _ in range(50)] + [Unserializable()]

    with pytest.raises(ValueError, match="max_result_chars"):
        _encode_bounded(result, 8192)


@pytest.mark.asyncio
async def test_oversized_result_does_not_prevent_other_tool_evidence():
    registry = Registry(
        [
            ToolDescriptor("bulk", "Bulk data", ToolSafety.READ_ONLY),
            ToolDescriptor("small", "Small data", ToolSafety.READ_ONLY),
        ],
        {"bulk": {"blob": "x" * 5000}, "small": {"ok": True}},
    )
    selector = Selector([ToolRequest("bulk"), ToolRequest("small")])
    statuses: list[tuple[str, str]] = []

    evidence = await collect_tool_evidence(
        "query",
        registry,
        selector,
        max_result_chars=1000,
        status_callback=lambda name, status: statuses.append((name, status)),
    )

    assert [item.tool_name for item in evidence] == ["small"]
    assert evidence[0].id == "T1"
    assert statuses == [("bulk", "failed"), ("small", "succeeded")]
