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

- [ ] **Step 2: Re-import in registry.py (delete the local defs)**

In `src/tools/registry.py`, delete the local `_check_json_type` and
`_validate_arguments` definitions (registry.py:99-138) and add near the other
imports:

…

### Task 2: Validate in `_call_tool` + tests

**Files:**
- Modify: `src/agents/tool/tool_calling.py`
- Test: `tests/unit/test_tool_arg_validation.py`

**Interfaces:**
- Consumes: `validate_arguments` (Task 1).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_tool_arg_validation.py`:

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/unit/test_tool_arg_validation.py -q`
Expected: FAIL — the loop currently executes `needs_int` (raising TypeError →
`error_code` is `"TypeError"`, not `"invalid_arguments"`; `executions` non-empty
for the wrong-type case is possible depending on the tool).

- [ ] **Step 3: Validate in `_call_tool`**

In `src/agents/tool/tool_calling.py`, add the import:

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
