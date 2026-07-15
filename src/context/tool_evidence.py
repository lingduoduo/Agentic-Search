"""Bounded, read-only tool execution for normalized RAG evidence."""

from __future__ import annotations

import asyncio
import inspect
import itertools
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Awaitable, Callable, Protocol

from .models import EvidenceSource


class ToolSafety(str, Enum):
    """Declared side-effect classification for a registered tool."""

    READ_ONLY = "read_only"
    SIDE_EFFECTING = "side_effecting"
    UNSPECIFIED = "unspecified"


@dataclass(frozen=True)
class ToolDescriptor:
    name: str
    description: str = ""
    safety: ToolSafety = ToolSafety.UNSPECIFIED


@dataclass(frozen=True)
class ToolRequest:
    tool_name: str
    arguments: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", _freeze_mapping(self.arguments))


class ToolRegistry(Protocol):
    def list_tools(self) -> list[ToolDescriptor]:
        """Return registered tool descriptors."""

    def invoke(self, request: ToolRequest) -> Awaitable[object]:
        """Invoke a registered tool."""


class ToolSelector(Protocol):
    def select(
        self, query: str, tools: list[ToolDescriptor]
    ) -> Iterable[ToolRequest] | Awaitable[Iterable[ToolRequest]]:
        """Select requests using only the supplied eligible tools."""


async def collect_tool_evidence(
    query: str,
    registry: ToolRegistry,
    selector: ToolSelector,
    *,
    max_calls: int = 2,
    timeout_seconds: float = 5.0,
    status_callback: Callable[[str, str], None] | None = None,
) -> list[EvidenceSource]:
    """Collect normalized evidence from explicitly read-only tools.

    Invalid selections, invocation failures, timeouts, and results that cannot be
    represented as JSON are ignored so retrieval-based answering can continue.
    """
    if max_calls < 0:
        raise ValueError("max_calls must be non-negative")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    try:
        descriptors = registry.list_tools()
        if not isinstance(descriptors, list) or not all(
            isinstance(descriptor, ToolDescriptor)
            and isinstance(descriptor.name, str)
            and bool(descriptor.name)
            and isinstance(descriptor.safety, ToolSafety)
            for descriptor in descriptors
        ):
            return []
        registered_names = {descriptor.name for descriptor in descriptors}
        name_counts = Counter(descriptor.name for descriptor in descriptors)
        eligible = [
            descriptor
            for descriptor in descriptors
            if descriptor.safety is ToolSafety.READ_ONLY
            and name_counts[descriptor.name] == 1
        ]
        eligible_names = {descriptor.name for descriptor in eligible}
    except Exception:
        return []
    try:
        selected = await asyncio.wait_for(
            asyncio.to_thread(selector.select, query, eligible),
            timeout=timeout_seconds,
        )
        if inspect.isawaitable(selected):
            selected = await asyncio.wait_for(selected, timeout=timeout_seconds)
    except Exception:
        return []

    evidence: list[EvidenceSource] = []
    attempted = 0
    selection_scan_limit = max_calls * 4
    try:
        selected_requests = await asyncio.wait_for(
            asyncio.to_thread(
                lambda: list(itertools.islice(selected, selection_scan_limit))
            ),
            timeout=timeout_seconds,
        )
    except Exception:
        return []
    for request in selected_requests:
        if not isinstance(request, ToolRequest):
            continue
        if request.tool_name not in eligible_names:
            if status_callback is not None and request.tool_name in registered_names:
                status_callback(request.tool_name, "rejected")
            continue
        if attempted >= max_calls:
            break
        attempted += 1
        try:
            result = await asyncio.wait_for(
                registry.invoke(request), timeout=timeout_seconds
            )
            text = json.dumps(
                result, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
        except Exception:
            if status_callback is not None:
                status_callback(request.tool_name, "failed")
            continue
        if status_callback is not None:
            status_callback(request.tool_name, "succeeded")
        evidence.append(
            EvidenceSource(
                id=f"T{len(evidence) + 1}",
                text=text,
                title=f"Tool: {request.tool_name}",
                provenance="tool",
                tool_name=request.tool_name,
            )
        )
    return evidence


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({key: _freeze(item) for key, item in value.items()})


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze(item) for item in value)
    return value
