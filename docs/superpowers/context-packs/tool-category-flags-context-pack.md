# Generated Context Pack

# Tool Category Flags

## Sources

- [Specification: 2026-07-09-tool-category-flags-design.md](../specs/2026-07-09-tool-category-flags-design.md)
- [Plan: 2026-07-09-tool-category-flags.md](../plans/2026-07-09-tool-category-flags.md)

## Specification Context

### Overview

Date: 2026-07-09
Status: Approved (brainstorming)
Branch/PR: feat/tool-category-flags
Related: [[project_chat_orchestration]]; first step of reconciling `src/internal/tools/` (stub layer) onto the canonical `src/tools/` registry

## Implementation Plan Context

### Task 1: Category flags on `Tool` / `FunctionTool` / `@tool` + real tools + tests

**Files:**
- Modify: `src/tools/base.py`, `src/tools/registry.py`, `src/tools/search.py`
- Test: `tests/unit/test_tool_categories.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_tool_categories.py`:

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/unit/test_tool_categories.py -v`
Expected: FAIL — `FunctionTool` has no `citeable`/`stopping`; `__init__` rejects the kwargs.

- [ ] **Step 3: Add properties to the `Tool` ABC**

In `src/tools/base.py`, in `class Tool(ABC)`, next to the existing `effect`
property, add:
- [ ] **Step 4: Add constructor params + property overrides to `FunctionTool`**

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
