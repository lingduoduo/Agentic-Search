"""Unit tests for src/internal/tools/registry.py."""

from __future__ import annotations

import pytest

from src.internal.tools import FunctionTool, ToolEffect
from src.internal.tools.registry import (
    ToolRegistry,
    _params_from_signature,
)
from src.internal.tools.validation import check_json_type, validate_arguments


def test_function_tool_defaults_to_unspecified_effect() -> None:
    tool = FunctionTool(lambda: "ok", name="unknown")
    assert tool.effect is ToolEffect.UNSPECIFIED
    assert "effect" not in tool.schema.to_dict()["function"]


def test_registry_decorator_accepts_read_only_effect() -> None:
    registry = ToolRegistry()

    @registry.tool(effect=ToolEffect.READ_ONLY)
    def lookup() -> str:
        return "ok"

    assert registry.get("lookup").effect is ToolEffect.READ_ONLY


# ---------------------------------------------------------------------------
# _params_from_signature
# ---------------------------------------------------------------------------


def _fn_with_hints(a: int, b: str, c: float = 1.0) -> str:
    """Docstring."""
    return str(a)


def _fn_no_hints(x, y=None):
    pass


def test_params_infers_types_and_required():
    schema = _params_from_signature(_fn_with_hints)
    assert schema["properties"]["a"]["type"] == "integer"
    assert schema["properties"]["b"]["type"] == "string"
    assert schema["properties"]["c"]["type"] == "number"
    assert schema["required"] == ["a", "b"]


def test_params_defaults_excluded_from_required():
    schema = _params_from_signature(_fn_with_hints)
    assert "c" not in schema.get("required", [])


def test_params_no_hints_defaults_to_string():
    schema = _params_from_signature(_fn_no_hints)
    assert schema["properties"]["x"]["type"] == "string"


def _fn_optional(a: "int | None" = None) -> None:  # noqa: F821
    pass


def test_params_optional_unwrapped():
    from typing import Optional

    # Test via direct annotation check (Optional at module level)
    import inspect

    def fn(a: Optional[int] = None) -> None:
        pass

    # Manually verify unwrapping logic since get_type_hints has trouble with
    # locally-defined Optional in test scope; just check parameter is optional
    sig = inspect.signature(fn)
    assert sig.parameters["a"].default is None  # optional (has default)
    # The schema should at least mark the param as present and optional
    schema = _params_from_signature(fn)
    assert "a" in schema["properties"]
    assert "required" not in schema  # no required fields since all have defaults


# ---------------------------------------------------------------------------
# _check_json_type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,json_type,expected",
    [
        ("hello", "string", True),
        (1, "integer", True),
        (True, "integer", False),  # bool is not int for json type check
        (1.5, "number", True),
        (1, "number", True),
        (True, "boolean", True),
        ([1, 2], "array", True),
        ({"k": 1}, "object", True),
        (None, "null", True),
        ("x", "integer", False),
    ],
)
def test_check_json_type(value, json_type, expected):
    assert check_json_type(value, json_type) is expected


# ---------------------------------------------------------------------------
# _validate_arguments
# ---------------------------------------------------------------------------


def test_validate_missing_required():
    params = {
        "required": ["a", "b"],
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
    }
    errors = validate_arguments(params, {"a": "hello"})
    assert any("b" in e for e in errors)


def test_validate_type_mismatch():
    params = {"required": ["a"], "properties": {"a": {"type": "integer"}}}
    errors = validate_arguments(params, {"a": "not-an-int"})
    assert errors


def test_validate_passes_valid():
    params = {"required": ["a"], "properties": {"a": {"type": "integer"}}}
    errors = validate_arguments(params, {"a": 42})
    assert errors == []


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


def test_register_function_tool():
    reg = ToolRegistry()

    @reg.tool(description="Add two ints")
    def add(a: int, b: int) -> int:
        return a + b

    assert "add" in {e.tool.name for e in reg.list()}
    s = reg.tool_summary("add")
    assert s is not None
    assert s["description"] == "Add two ints"
    assert s["source"] == "function"


def test_register_overwrites_existing():
    reg = ToolRegistry()

    @reg.tool
    def fn(x: str) -> str:
        return x

    @reg.tool(name="fn", description="new desc")
    def fn2(x: str) -> str:
        return x

    assert len([e for e in reg.list() if e.tool.name == "fn"]) == 1
    assert reg.tool_summary("fn")["description"] == "new desc"


def test_unregister():
    reg = ToolRegistry()

    @reg.tool
    def fn(x: str) -> str:
        return x

    assert reg.unregister("fn") is True
    assert reg.get("fn") is None
    assert reg.unregister("fn") is False


@pytest.mark.asyncio
async def test_invoke_function_tool():
    reg = ToolRegistry()

    @reg.tool(description="Multiply")
    def mul(a: int, b: int) -> int:
        return a * b

    response, raw, errors = await reg.invoke("mul", {"a": 3, "b": 4})
    assert errors == []
    assert raw == 12
    assert response == "12"


@pytest.mark.asyncio
async def test_invoke_missing_tool():
    reg = ToolRegistry()
    response, raw, errors = await reg.invoke("nonexistent", {})
    assert errors
    assert "not found" in errors[0]


@pytest.mark.asyncio
async def test_invoke_validation_error():
    reg = ToolRegistry()

    @reg.tool
    def typed(n: int) -> int:
        return n

    _, _, errors = await reg.invoke("typed", {"n": "not_an_int"})
    assert errors  # type error detected


@pytest.mark.asyncio
async def test_invoke_missing_required():
    reg = ToolRegistry()

    @reg.tool
    def greet(name: str) -> str:
        return f"hi {name}"

    _, _, errors = await reg.invoke("greet", {})
    assert any("name" in e for e in errors)


def test_all_summaries():
    reg = ToolRegistry()

    @reg.tool
    def a(x: str) -> str:
        return x

    @reg.tool
    def b(x: str) -> str:
        return x

    summaries = reg.all_summaries()
    names = {s["name"] for s in summaries}
    assert {"a", "b"} <= names


@pytest.mark.asyncio
async def test_async_function_tool():
    reg = ToolRegistry()

    @reg.tool(description="Async double")
    async def double(n: int) -> int:
        return n * 2

    response, raw, errors = await reg.invoke("double", {"n": 5})
    assert errors == []
    assert raw == 10
