# Generated Context Pack

# Validate Tool Arguments

## Sources

- [Specification: 2026-07-09-validate-tool-arguments-design.md](../archive/specs/2026-07-09-validate-tool-arguments-design.md)
- [Plan: 2026-07-09-validate-tool-arguments.md](../archive/plans/2026-07-09-validate-tool-arguments.md)

## Specification Context

### Overview

Date: 2026-07-09
Status: Approved (brainstorming)
Branch/PR: feat/validate-tool-arguments
Related: [[project_chat_orchestration]] (tool sub-item), composes with PR #395 (tool-error feedback)

## Implementation Plan Context

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
