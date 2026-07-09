"""Tool-argument validation in ToolAgentLoop._call_tool."""

from __future__ import annotations

import pytest

from src.agents.core.state import TaskStatus
from src.tools import FunctionTool, ToolEffect
from src.tools.validation import validate_arguments
from tests.unit.test_tool_approval import _loop, _trace

_INT_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "integer"}},
    "required": ["value"],
}


def test_validate_arguments_missing_and_wrong_type():
    assert validate_arguments(_INT_SCHEMA, {}) == ["Missing required argument: 'value'"]
    assert validate_arguments(_INT_SCHEMA, {"value": "x"}) != []
    assert validate_arguments(_INT_SCHEMA, {"value": 3}) == []
    assert validate_arguments({}, {"anything": 1}) == []  # schemaless → no errors


@pytest.mark.asyncio
async def test_missing_required_argument_not_executed():
    executions = []

    @FunctionTool.from_fn(effect=ToolEffect.READ_ONLY, parameters=_INT_SCHEMA)
    def needs_int(value: int):
        executions.append(value)
        return value

    loop, _ = _loop([needs_int], ['{"name":"needs_int","arguments":{}}', "done"])
    output = await loop.run([{"role": "user", "content": "go"}], {})
    result = _trace(output)[0]
    assert result["status"] == str(TaskStatus.FAILED)
    assert result["error_code"] == "invalid_arguments"
    assert executions == []  # the tool body never ran


@pytest.mark.asyncio
async def test_wrong_type_argument_not_executed():
    executions = []

    @FunctionTool.from_fn(effect=ToolEffect.READ_ONLY, parameters=_INT_SCHEMA)
    def needs_int(value: int):
        executions.append(value)
        return value

    loop, _ = _loop(
        [needs_int], ['{"name":"needs_int","arguments":{"value":"x"}}', "done"]
    )
    output = await loop.run([{"role": "user", "content": "go"}], {})
    result = _trace(output)[0]
    assert result["error_code"] == "invalid_arguments"
    assert executions == []
