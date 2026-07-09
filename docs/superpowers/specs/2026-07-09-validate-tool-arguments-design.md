# Validate Tool Arguments in the Agent Loop — Design

Date: 2026-07-09
Status: Approved (brainstorming)
Branch/PR: feat/validate-tool-arguments
Related: [[project_chat_orchestration]] (tool sub-item), composes with PR #395 (tool-error feedback)

## Problem

Two execution paths share the same `Tool` objects with different safety:
- `ToolRegistry.invoke()` (REST/MCP path, `src/tools/registry.py:296`) validates
  arguments against the tool's JSON schema (`_validate_arguments`) and returns
  errors **without executing** when invalid.
- `ToolAgentLoop._call_tool()` (RL/agent path, `src/agents/tool/tool_calling.py:186`)
  calls `tool.execute()` **directly, with no validation** — so a malformed call
  (missing required arg / wrong type) reaches the tool body and fails with a raw
  Python exception (e.g. `TypeError`), and a **side-effecting tool runs anyway**
  before failing.

Goal: validate arguments in `_call_tool` before executing, mirroring
`ToolRegistry.invoke()`.

## Non-goals

- No change to `ToolRegistry.invoke()` behavior (validation there is unchanged).
- No new validation rules — reuse the existing `_validate_arguments` logic.
- No dependency on PR #395 (they compose; see below).

## Approach

### 1. Extract the validators to a dependency-free module

`_check_json_type` and `_validate_arguments` are pure schema utilities that happen
to live in `registry.py`. Importing them into the token-level agent loop would drag
the whole REST/OpenAPI registry (`ApiToolRegistry`, the `tool_registry` singleton)
into the loop's import chain — a layering smell.

Create `src/tools/validation.py` (deps: `typing` only):
- `check_json_type(value, json_type) -> bool` (moved verbatim from `_check_json_type`).
- `validate_arguments(parameters, arguments) -> list[str]` (moved verbatim from
  `_validate_arguments`, calling `check_json_type`).

`registry.py` re-imports them with the old names for back-compat, so its logic and
the existing `test_tool_registry.py` imports are unchanged:
```python
from .validation import check_json_type as _check_json_type
from .validation import validate_arguments as _validate_arguments
```

### 2. Validate in `_call_tool`

In `ToolAgentLoop._call_tool` (`tool_calling.py`), compute `args` once and validate
before `create()`/`execute()`:
```python
    args = tool_call.parsed_arguments()
    errors = validate_arguments(tool.schema.parameters, args)
    if errors:
        error_code = "invalid_arguments"
        error_message = "; ".join(errors)
    else:
        instance_id = await tool.create()
        result, _, _ = await tool.execute(instance_id, args)
        status = TaskStatus.COMPLETED
        self._record_tool_stage(tool_call.name, args, result)
```
Empty schema (`parameters == {}`) → `validate_arguments` returns `[]` → executes
normally (schemaless tools unaffected), matching `registry.invoke`.

## Why it matters (two wins)

1. **Safety (independent of #395):** a `SIDE_EFFECTING` tool is no longer executed
   with malformed args — validation runs before `create()`/`execute()`.
2. **Self-correction (composes with #395):** the failure now carries a clear
   `error_code="invalid_arguments"` + `"Missing required argument: 'query'"` instead
   of a raw `TypeError`. With #395 merged (feeds FAILED results back to the model),
   the model can fix its own call. This PR is off `main`, independent of #395.

## Success criteria

- A tool call with a missing required argument or wrong-typed argument yields a
  FAILED result with `error_code="invalid_arguments"` and the validation errors in
  `error_message`, and the tool body is **not executed**.
- Valid calls and schemaless tools behave exactly as before.
- `ToolRegistry.invoke()` and `test_tool_registry.py` are unchanged (back-compat
  re-exports).

## Testing (no model)

- `validate_arguments` (via the new module): missing required → error; wrong type →
  error; valid → `[]`; empty schema → `[]`.
- Loop-level: a `FunctionTool` with a required `int` arg, called with a missing arg
  and with a wrong-typed arg — assert the result is FAILED with
  `error_code="invalid_arguments"` and that the tool's body did **not** run (a
  side-effect list stays empty).
- `test_tool_registry.py` stays green (re-exports intact).

## Risks

- Low. Additive validation on a path that currently fails anyway; the only behavior
  change is *when* and *how cleanly* an invalid call fails (before execution, with a
  clear error) — strictly safer.
