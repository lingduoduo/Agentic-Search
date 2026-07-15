# Generated Context Pack

# Consolidate Tool Execution

## Sources

- [Specification: 2026-07-09-consolidate-tool-execution-design.md](../specs/2026-07-09-consolidate-tool-execution-design.md)
- [Plan: 2026-07-09-consolidate-tool-execution.md](../plans/2026-07-09-consolidate-tool-execution.md)

## Specification Context

### Decision & caveats (from brainstorming)

- **Per-loop registry, NOT the global singleton.** The loop's tools come from its
  constructor, not `tool_registry`; routing through the global singleton would
  return "Tool not found" for the loop's own tools. So the loop builds its own
  `ToolRegistry` instance.
- **Accepted tradeoff:** the token-level loop now imports `ToolRegistry` (pulling
  the REST/OpenAPI registry into its import chain). Explicitly chosen for a single
  execution path.
- This supersedes #397's *inline* validation in `_call_tool` (invoke validates);
  the extracted `src/tools/validation.py` stays (invoke uses it).

## Implementation Plan Context

### Task 1: Per-loop registry + reroute `_call_tool` + tests

**Files:**
- Modify: `src/agents/tool/tool_calling.py`
- Test: `tests/unit/test_tool_arg_validation.py` (add unknown-tool case)

- [ ] **Step 1: Write the failing test (unknown tool → tool_not_found)**

Append to `tests/unit/test_tool_arg_validation.py`:

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/unit/test_tool_arg_validation.py::test_unknown_tool_reports_not_found -v`
Expected: FAIL — current `_call_tool` raises `KeyError` → `error_code == "KeyError"`.

- [ ] **Step 3: Add the ToolRegistry import**

In `src/agents/tool/tool_calling.py`, add:
- [ ] **Step 4: Build a per-loop registry in `__init__`**

Replace:
with:

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
