# Generated Context Pack

# Validate Tool Arguments

## Sources

- [Specification: 2026-07-09-validate-tool-arguments-design.md](../specs/2026-07-09-validate-tool-arguments-design.md)
- [Plan: 2026-07-09-validate-tool-arguments.md](../plans/2026-07-09-validate-tool-arguments.md)

## Specification Context

### Non-goals

- No change to `ToolRegistry.invoke()` behavior (validation there is unchanged).
- No new validation rules — reuse the existing `_validate_arguments` logic.
- No dependency on PR #395 (they compose; see below).

### Testing (no model)

- `validate_arguments` (via the new module): missing required → error; wrong type →
  error; valid → `[]`; empty schema → `[]`.
- Loop-level: a `FunctionTool` with a required `int` arg, called with a missing arg
  and with a wrong-typed arg — assert the result is FAILED with
  `error_code="invalid_arguments"` and that the tool's body did **not** run (a
  side-effect list stays empty).
- `test_tool_registry.py` stays green (re-exports intact).

### Risks

- Low. Additive validation on a path that currently fails anyway; the only behavior
  change is *when* and *how cleanly* an invalid call fails (before execution, with a
  clear error) — strictly safer.

## Implementation Plan Context

### Global Constraints

- Branch off `main` (never commit to `main`); branch `feat/validate-tool-arguments`.
- `ToolRegistry.invoke()` behavior and `test_tool_registry.py` must be unchanged (back-compat re-exports keep `_check_json_type`/`_validate_arguments` importable from `registry`).
- Schemaless tools (`parameters == {}`) behave exactly as before.
- Match repo ruff formatting.

---

### Task 1: Extract validators to `src/tools/validation.py`

**Files:**
- Create: `src/tools/validation.py`
- Modify: `src/tools/registry.py`
- Test: `tests/unit/test_tool_registry.py` (unchanged — verify still green)

**Interfaces:**
- Produces: `validate_arguments(parameters: dict, arguments: dict) -> list[str]`, `check_json_type(value, json_type: str) -> bool` in `src.tools.validation`.

- [ ] **Step 1: Create the module (move verbatim)**

Create `src/tools/validation.py`:

```python
"""Validate call arguments against a tool's JSON-schema parameters."""

from __future__ import annotations

from typing import Any


def check_json_type(value: Any, json_type: str) -> bool:
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


def validate_arguments(
    parameters: dict[str, Any], arguments: dict[str, Any]
) -> list[str]:
    """Return a list of validation error strings (empty = valid)."""
    errors: list[str] = []
    required = parameters.get("required", [])
    props: dict[str, Any] = parameters.get("properties", {})

    for req in required:
        if req not in arguments:

_[Section compacted.]_

### Task 2: Validate in `_call_tool` + tests

**Files:**
- Modify: `src/agents/tool/tool_calling.py`
- Test: `tests/unit/test_tool_arg_validation.py`

**Interfaces:**
- Consumes: `validate_arguments` (Task 1).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_tool_arg_validation.py`:

```python
"""Tool-argument validation in ToolAgentLoop._call_tool."""

from __future__ import annotations

import pytest

from src.agents.core.state import TaskStatus
from src.tools import FunctionTool, ToolEffect
from src.tools.validation import validate_arguments
from tests.unit.test_tool_approval import _loop, _trace


def test_validate_arguments_missing_and_wrong_type():
    params = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    }
    assert validate_arguments(params, {}) == ["Missing required argument: 'value'"]
    assert validate_arguments(params, {"value": "x"}) != []
    assert validate_arguments(params, {"value": 3}) == []
    assert validate_arguments({}, {"anything": 1}) == []  # schemaless → no errors


@pytest.mark.asyncio
async def test_missing_required_argument_not_executed():
    executions = []

    @FunctionTool.from_fn(effect=ToolEffect.SIDE_EFFECTING)
    def needs_int(value: int):
        executions.append(value)
        return value

    loop, _ = _loop([needs_int], ['{"name":"needs_int","arguments":{}}', "done"])
    output = await loop.run([{"role": "user", "content": "go"}], {})
    result = _trace(output)[0]
    assert result["status"] == str(TaskStatus.FAILED)
    assert result["error_code"] == "invalid_arguments"

_[Section compacted.]_

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
