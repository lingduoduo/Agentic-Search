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

```python
"""Category flags (citeable / stopping) on the canonical Tool."""

from __future__ import annotations

from src.tools.base import FunctionTool, ToolEffect
from src.tools.registry import ToolRegistry
from src.tools.search import MultiQueryWebSearchTool, build_search_tool


def test_functiontool_defaults_false():
    t = FunctionTool(lambda: "ok", name="x")
    assert t.citeable is False
    assert t.stopping is False


def test_functiontool_flags_settable():
    t = FunctionTool(lambda: "ok", name="x", citeable=True, stopping=True)
    assert t.citeable is True
    assert t.stopping is True


def test_flags_not_leaked_into_schema():
    t = FunctionTool(lambda: "ok", name="x", citeable=True)
    fn = t.schema.to_dict()["function"]
    assert "citeable" not in fn
    assert "stopping" not in fn


def test_tool_decorator_threads_flags():
    reg = ToolRegistry()

    @reg.tool(description="d", citeable=True)
    def search_ish(q: str) -> str:
        return q

    assert reg.get("search_ish").citeable is True
    assert reg.get("search_ish").stopping is False


def test_real_search_tools_are_citeable():
    assert build_search_tool().citeable is True
    assert MultiQueryWebSearchTool().citeable is True
    # web_search doesn't stop the loop
    assert MultiQueryWebSearchTool().stopping is False
```

- [ ] **Step 2: Run to verify it fails**

_[Section compacted.]_

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
