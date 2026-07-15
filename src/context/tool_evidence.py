"""Bounded, read-only tool execution for normalized RAG evidence."""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Protocol

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
    arguments: dict[str, object] = field(default_factory=dict)


class ToolRegistry(Protocol):
    def list_tools(self) -> list[ToolDescriptor]:
        """Return registered tool descriptors."""

    def invoke(self, request: ToolRequest) -> Awaitable[object]:
        """Invoke a registered tool."""


class ToolSelector(Protocol):
    def select(
        self, query: str, tools: list[ToolDescriptor]
    ) -> list[ToolRequest] | Awaitable[list[ToolRequest]]:
        """Select requests using only the supplied eligible tools."""


async def collect_tool_evidence(
    query: str,
    registry: ToolRegistry,
    selector: ToolSelector,
    *,
    max_calls: int = 2,
    timeout_seconds: float = 5.0,
) -> list[EvidenceSource]:
    """Collect normalized evidence from explicitly read-only tools.

    Invalid selections, invocation failures, timeouts, and results that cannot be
    represented as JSON are ignored so retrieval-based answering can continue.
    """
    if max_calls < 0:
        raise ValueError("max_calls must be non-negative")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    eligible = [
        descriptor
        for descriptor in registry.list_tools()
        if descriptor.safety is ToolSafety.READ_ONLY
    ]
    eligible_names = {descriptor.name for descriptor in eligible}
    selected = selector.select(query, eligible)
    if inspect.isawaitable(selected):
        selected = await selected

    evidence: list[EvidenceSource] = []
    attempted = 0
    for request in selected:
        if request.tool_name not in eligible_names:
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
            continue
        evidence.append(
            EvidenceSource(
                id=f"T{len(evidence) + 1}",
                text=text,
                title=f"Tool: {request.tool_name}",
                provenance="tool",
                tool_name=request.tool_name,
                metadata={"arguments": request.arguments},
            )
        )
    return evidence
