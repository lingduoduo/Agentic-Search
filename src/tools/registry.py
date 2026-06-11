"""Global tool registry for Agentic Search.

Provides a singleton registry that:
  - Wraps Python callables as tools via ``@tool`` (schema auto-inferred from type hints)
  - Registers OpenAPI operations as tools via ``register_from_openapi()``
  - Validates arguments against the declared JSON schema before execution
  - Acts as the single source of truth for both the REST management API and
    the MCP server bridge

Usage::

    from src.tools.registry import tool, tool_registry

    @tool(description="Add two numbers")
    def add(a: int, b: int) -> int:
        return a + b

    result = await tool_registry.invoke("add", {"a": 1, "b": 2})

    # Register an OpenAPI spec (all operations become tools)
    ids = tool_registry.register_from_openapi(
        openapi_json_str, name="my_api", headers={"Authorization": "Bearer ..."}
    )
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints
from uuid import UUID

from .api import ApiToolRegistry, ApiToolNotFoundError
from .base import FunctionTool, Tool

logger = logging.getLogger(__name__)

_PY_TO_JSON_TYPE: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    bytes: "string",
}


def _params_from_signature(fn: Callable) -> dict[str, Any]:
    """Auto-build a JSON Schema ``parameters`` object from a function's type hints."""
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        raw_hint = hints.get(pname)

        # Unwrap Optional[X] → X
        origin = get_origin(raw_hint)
        if origin is Union:
            inner = [a for a in get_args(raw_hint) if a is not type(None)]
            raw_hint = inner[0] if inner else None

        json_type = _PY_TO_JSON_TYPE.get(raw_hint, "string")  # type: ignore[arg-type]
        prop: dict[str, Any] = {"type": json_type}

        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(pname)

        properties[pname] = prop

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _check_json_type(value: Any, json_type: str) -> bool:
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "array":
        return isinstance(value, list)
    if json_type == "object":
        return isinstance(value, dict)
    if json_type == "null":
        return value is None
    return True  # unknown type — don't reject


def _validate_arguments(
    parameters: dict[str, Any], arguments: dict[str, Any]
) -> list[str]:
    """Return a list of validation error strings (empty = valid)."""
    errors: list[str] = []
    required = parameters.get("required", [])
    props: dict[str, Any] = parameters.get("properties", {})

    for req in required:
        if req not in arguments:
            errors.append(f"Missing required argument: {req!r}")

    for key, value in arguments.items():
        if key not in props:
            continue
        expected = props[key].get("type")
        if expected and not _check_json_type(value, expected):
            errors.append(
                f"Argument {key!r}: expected {expected!r}, got {type(value).__name__!r}"
            )

    return errors


@dataclass
class ToolEntry:
    """Metadata stored alongside each registered tool."""

    tool: Tool
    source: str  # "function" | "openapi"
    provider_id: str | None  # set for OpenAPI-registered tools


class ToolRegistry:
    """Singleton registry for all tools exposed via MCP and the REST API."""

    def __init__(self) -> None:
        self._entries: dict[str, ToolEntry] = {}
        self._api_registry = ApiToolRegistry()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self, tool: Tool, *, source: str = "function", provider_id: str | None = None
    ) -> None:
        """Add a tool to the registry (replaces any existing tool with the same name)."""
        self._entries[tool.name] = ToolEntry(
            tool=tool, source=source, provider_id=provider_id
        )
        logger.debug("Tool registered: %s (source=%s)", tool.name, source)

    def tool(
        self,
        fn: Callable | None = None,
        *,
        name: str | None = None,
        description: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> Any:
        """Decorator that registers a Python function as a tool.

        Schema is auto-inferred from type hints when *parameters* is omitted.

        Can be used with or without arguments::

            @tool_registry.tool
            def greet(name: str) -> str: ...

            @tool_registry.tool(description="Greet someone")
            def greet(name: str) -> str: ...
        """

        def _decorate(f: Callable) -> Callable:
            resolved_params = parameters or _params_from_signature(f)
            resolved_desc = description or (f.__doc__ or "").strip()
            t = FunctionTool(
                f,
                name=name or f.__name__,
                description=resolved_desc,
                parameters=resolved_params,
            )
            self.register(t, source="function")
            return (
                f  # return original fn, not FunctionTool, so callers can still call it
            )

        if fn is not None:
            return _decorate(fn)
        return _decorate

    def register_from_openapi(
        self,
        openapi_json: str,
        *,
        name: str,
        headers: dict[str, str] | None = None,
        icon: str | None = None,
    ) -> list[str]:
        """Parse an OpenAPI JSON string and register all operations as tools.

        Returns the list of tool names that were registered.
        """
        provider = self._api_registry.create_provider(
            name=name,
            openapi_schema=openapi_json,
            headers=headers,
            icon=icon,
        )
        tool_names: list[str] = []
        for api_tool in self._api_registry.build_tools(provider.id):
            self.register(api_tool, source="openapi", provider_id=str(provider.id))
            tool_names.append(api_tool.name)
        logger.info(
            "Registered %d tool(s) from OpenAPI provider %r (id=%s)",
            len(tool_names),
            name,
            provider.id,
        )
        return tool_names

    def unregister(self, name: str) -> bool:
        """Remove a tool by name. Returns True if it existed."""
        entry = self._entries.pop(name, None)
        return entry is not None

    def unregister_provider(self, provider_id: str) -> list[str]:
        """Remove all tools from an OpenAPI provider. Returns removed tool names."""
        removed = [n for n, e in self._entries.items() if e.provider_id == provider_id]
        for n in removed:
            del self._entries[n]
        try:
            self._api_registry.delete_provider(UUID(provider_id))
        except (ApiToolNotFoundError, ValueError):
            pass
        return removed

    # ------------------------------------------------------------------
    # Lookup & invocation
    # ------------------------------------------------------------------

    def get(self, name: str) -> Tool | None:
        entry = self._entries.get(name)
        return entry.tool if entry else None

    def list(self) -> list[ToolEntry]:
        return list(self._entries.values())

    def list_tools(self) -> list[Tool]:
        return [e.tool for e in self._entries.values()]

    async def invoke(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        validate: bool = True,
    ) -> tuple[str, Any, list[str]]:
        """Execute a tool by name.

        Returns ``(response_text, raw_output, validation_errors)``.
        *validate=True* (default) checks arguments against the declared schema
        and returns errors without executing if invalid.
        """
        entry = self._entries.get(name)
        if entry is None:
            return "", None, [f"Tool {name!r} not found."]

        tool = entry.tool
        errors: list[str] = []
        if validate and tool.schema.parameters:
            errors = _validate_arguments(tool.schema.parameters, arguments)
            if errors:
                return "", None, errors

        instance_id = await tool.create()
        try:
            response, raw, _meta = await tool.execute(instance_id, arguments)
        finally:
            await tool.release(instance_id)

        return response, raw, []

    # ------------------------------------------------------------------
    # Summary for the REST API
    # ------------------------------------------------------------------

    def tool_summary(self, name: str) -> dict[str, Any] | None:
        entry = self._entries.get(name)
        if not entry:
            return None
        t = entry.tool
        return {
            "name": t.name,
            "description": t.schema.description,
            "parameters": t.schema.parameters,
            "source": entry.source,
            "provider_id": entry.provider_id,
        }

    def all_summaries(self) -> list[dict[str, Any]]:
        return [
            {
                "name": e.tool.name,
                "description": e.tool.schema.description,
                "parameters": e.tool.schema.parameters,
                "source": e.source,
                "provider_id": e.provider_id,
            }
            for e in self._entries.values()
        ]


# ---------------------------------------------------------------------------
# Module-level singleton + convenience decorator
# ---------------------------------------------------------------------------

tool_registry = ToolRegistry()


def tool(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    description: str = "",
    parameters: dict[str, Any] | None = None,
) -> Any:
    """Module-level shorthand for ``tool_registry.tool(...)``."""
    return tool_registry.tool(
        fn, name=name, description=description, parameters=parameters
    )


__all__ = [
    "ToolRegistry",
    "ToolEntry",
    "tool_registry",
    "tool",
]
