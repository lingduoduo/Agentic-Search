# Validate Tool Arguments Implementation Plan

> Use superpowers:test-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Validate tool-call arguments in `ToolAgentLoop._call_tool` before executing, reusing the registry's validation logic via a shared module.

**Architecture:** Extract `check_json_type`/`validate_arguments` into `src/tools/validation.py`; `registry.py` re-imports them (back-compat); `_call_tool` validates before execute.

**Tech Stack:** Python.

## Global Constraints

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
            errors.append(f"Missing required argument: {req!r}")

    for key, value in arguments.items():
        if key not in props:
            continue
        expected = props[key].get("type")
        if expected and not check_json_type(value, expected):
            errors.append(
                f"Argument {key!r}: expected {expected!r}, got {type(value).__name__!r}"
            )

    return errors
```

- [ ] **Step 2: Re-import in registry.py (delete the local defs)**

In `src/tools/registry.py`, delete the local `_check_json_type` and
`_validate_arguments` definitions (registry.py:99-138) and add near the other
imports:

```python
from .validation import check_json_type as _check_json_type
from .validation import validate_arguments as _validate_arguments
```

(`registry.invoke` at :316 keeps calling `_validate_arguments`; `test_tool_registry.py`
keeps importing `_check_json_type`/`_validate_arguments` from `registry` — both
resolve via the re-import.)

- [ ] **Step 3: Verify registry unchanged**

Run: `python3 -m pytest tests/unit/test_tool_registry.py -q`
Expected: PASS (behavior identical; validators just moved).

- [ ] **Step 4: Commit**

```bash
git add src/tools/validation.py src/tools/registry.py
git commit -m "refactor(tools): extract arg validation into src/tools/validation.py

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

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
    assert executions == []  # the tool body never ran


@pytest.mark.asyncio
async def test_wrong_type_argument_not_executed():
    executions = []

    @FunctionTool.from_fn(effect=ToolEffect.SIDE_EFFECTING)
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/unit/test_tool_arg_validation.py -q`
Expected: FAIL — the loop currently executes `needs_int` (raising TypeError →
`error_code` is `"TypeError"`, not `"invalid_arguments"`; `executions` non-empty
for the wrong-type case is possible depending on the tool).

- [ ] **Step 3: Validate in `_call_tool`**

In `src/agents/tool/tool_calling.py`, add the import:
```python
from src.tools.validation import validate_arguments
```
Rewrite the body of `_call_tool` inside the `try`:
```python
        try:
            tool = self.tools[tool_call.name]
            args = tool_call.parsed_arguments()
            errors = validate_arguments(tool.schema.parameters, args)
            if errors:
                error_code = "invalid_arguments"
                error_message = "; ".join(errors)
            else:
                instance_id = await tool.create()
                result, _, _ = await tool.execute(instance_id, args)
                elapsed = time.perf_counter() - start
                status = TaskStatus.COMPLETED
                self._record_tool_stage(tool_call.name, args, result)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.exception("Error executing tool %r: %s", tool_call.name, exc)
            error_code = type(exc).__name__
            error_message = str(exc)
        finally:
            if tool is not None and instance_id is not None:
                await tool.release(instance_id)
```
And update the `ToolExecutionResult(... arguments=...)` to use `args` (compute
`args` once; keep a safe default `args = {}` before the `try` so the final
`return` can reference it even if `parsed_arguments()` was never reached).

- [ ] **Step 4: Run new tests + regression**

Run: `python3 -m pytest tests/unit/test_tool_arg_validation.py tests/unit/test_tool_approval.py tests/unit/test_tool_registry.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agents/tool/tool_calling.py tests/unit/test_tool_arg_validation.py
git commit -m "feat(tool): validate arguments in _call_tool before executing

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

- Extract validators to `validation.py`, registry re-imports (spec §Approach 1) → Task 1. ✓
- Registry + its tests unchanged → Task 1 Step 3. ✓
- Validate before execute in `_call_tool` (spec §Approach 2) → Task 2 Step 3. ✓
- Missing/wrong-type → FAILED `invalid_arguments`, not executed (spec success criteria) → Task 2 tests. ✓
- Schemaless unaffected → `test_validate_arguments_missing_and_wrong_type` (`{}` case) + unchanged execute path. ✓
- Types consistent: `validate_arguments(dict, dict) -> list[str]` across def, registry re-import, and `_call_tool` call. ✓
