"""Tool abstraction for the ToolAgentLoop."""

from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolSchema:
    """JSON Schema description of one tool, following the OpenAI function-calling spec."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Tool(ABC):
    """Abstract base for a callable tool used by ToolAgentLoop.

    Lifecycle per tool call:
        instance_id = await tool.create()
        response, raw, meta = await tool.execute(instance_id, arguments)
        await tool.release(instance_id)
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def schema(self) -> ToolSchema: ...

    async def create(self) -> str:
        """Allocate a tool instance. Returns an opaque instance_id."""
        return "default"

    @abstractmethod
    async def execute(
        self, instance_id: str, arguments: dict[str, Any]
    ) -> tuple[str, Any, Any]:
        """Execute the tool. Returns (response_text, raw_output, metadata)."""
        ...

    async def release(self, instance_id: str) -> None:
        """Free any resources held by *instance_id*."""


class FunctionTool(Tool):
    """Wrap a plain Python callable (sync or async) as a Tool.

    Example::

        @FunctionTool.from_fn(
            description="Search Wikipedia",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        async def search_wiki(query: str) -> str:
            ...
    """

    def __init__(
        self,
        fn: Callable,
        name: str | None = None,
        description: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self._fn = fn
        self._name = name or fn.__name__
        self._schema = ToolSchema(
            name=self._name,
            description=description or (fn.__doc__ or "").strip(),
            parameters=parameters or {},
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    async def execute(
        self, instance_id: str, arguments: dict[str, Any]
    ) -> tuple[str, Any, Any]:
        if inspect.iscoroutinefunction(self._fn):
            result = await self._fn(**arguments)
        else:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: self._fn(**arguments))
        return str(result), result, {}

    @classmethod
    def from_fn(
        cls,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> Callable:
        """Decorator factory that wraps a function as a FunctionTool."""

        def decorator(fn: Callable) -> "FunctionTool":
            return cls(fn, name=name, description=description, parameters=parameters)

        return decorator
