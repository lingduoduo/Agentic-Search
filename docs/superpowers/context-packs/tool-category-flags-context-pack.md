# Generated Context Pack

# Tool Category Flags

## Sources

- [Specification: 2026-07-09-tool-category-flags-design.md](../specs/2026-07-09-tool-category-flags-design.md)
- [Plan: 2026-07-09-tool-category-flags.md](../plans/2026-07-09-tool-category-flags.md)

## Specification Context

### This PR (scoped, additive)

Teach the canonical `Tool` those two facts, and have the real tools declare them.
**Nothing consumes the flags yet** — this is the safe foundation. (The natural
consumer, `admin_surface.py` deriving the sets from the registry, is deferred:
today the built-in tools are NOT registered in the global `tool_registry`, so a
registry-derived count would be wrong. That migration is a separate, larger step.)

### Non-goals

- No change to `admin_surface.py` / `tool_call_args_streaming.py` (deferred).
- No removal of `built_in_tools.py` name-sets yet (still the source of truth for
  admin until the migration).
- Flags are **not** added to `ToolSchema.to_dict()` — they are tool metadata, not
  part of the JSON function-definition the model sees (like `effect`).

### Testing (no model)

- `FunctionTool(..., citeable=True, stopping=True)` → the properties reflect it;
  a default `FunctionTool` → both `False`.
- `@tool_registry.tool(citeable=True)` registers a tool whose `.citeable is True`.
- `build_search_tool(...).citeable is True`; a `MultiQueryWebSearchTool().citeable
  is True`.
- `ToolSchema.to_dict()` has no `citeable`/`stopping` keys.

### Risks

- Minimal — purely additive metadata with `False` defaults. The only "value" is
  that the canonical `search`/`web_search` tools now self-describe their category,
  ready for the later admin migration.

## Implementation Plan Context

### Global Constraints

- Branch off `main` (never commit to `main`); branch `feat/tool-category-flags`.
- Purely additive: default `False`; no behavior change; `ToolSchema.to_dict()` unchanged (flags are tool metadata, not JSON schema).
- Do NOT touch `admin_surface.py` / `tool_call_args_streaming.py` / `built_in_tools.py` (deferred migration).
- Match repo ruff formatting.

---

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
