# Tool Category Flags Implementation Plan

> Use superpowers:test-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add `citeable`/`stopping` category flags to the canonical `Tool`, and have the real `search`/`web_search` tools declare `citeable=True`.

**Architecture:** Additive metadata in `src/tools/` — properties on `Tool`, constructor params on `FunctionTool`, decorator params on `@tool_registry.tool`, and declarations on the two real search tools. No consumer yet; not serialized to the model.

**Tech Stack:** Python.

## Global Constraints

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

Run: `python3 -m pytest tests/unit/test_tool_categories.py -v`
Expected: FAIL — `FunctionTool` has no `citeable`/`stopping`; `__init__` rejects the kwargs.

- [ ] **Step 3: Add properties to the `Tool` ABC**

In `src/tools/base.py`, in `class Tool(ABC)`, next to the existing `effect`
property, add:
```python
    @property
    def citeable(self) -> bool:
        """True if this tool produces citable documents."""
        return False

    @property
    def stopping(self) -> bool:
        """True if the loop should stop after this tool runs."""
        return False
```

- [ ] **Step 4: Add constructor params + property overrides to `FunctionTool`**

In `FunctionTool.__init__`, add params and store them:
```python
    def __init__(
        self,
        fn: Callable,
        name: str | None = None,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        effect: ToolEffect = ToolEffect.UNSPECIFIED,
        citeable: bool = False,
        stopping: bool = False,
    ) -> None:
        ...
        self._effect = effect
        self._citeable = citeable
        self._stopping = stopping
        ...
```
And add property overrides beside the existing `effect` property:
```python
    @property
    def citeable(self) -> bool:
        return self._citeable

    @property
    def stopping(self) -> bool:
        return self._stopping
```
Also thread `citeable`/`stopping` through `FunctionTool.from_fn` (add the two
params, default `False`, pass to the constructor).

- [ ] **Step 5: Thread flags through the `@tool` decorator**

In `src/tools/registry.py`, `ToolRegistry.tool(...)`, add `citeable: bool = False`,
`stopping: bool = False` params and pass them into the `FunctionTool(...)` it builds.

- [ ] **Step 6: Declare on the real tools**

- `src/tools/search.py`: in `MultiQueryWebSearchTool`, add:
  ```python
      @property
      def citeable(self) -> bool:
          return True
  ```
- `build_search_tool`: add `citeable=True` to the `FunctionTool(...)` call.

- [ ] **Step 7: Run new tests + regression**

Run: `python3 -m pytest tests/unit/test_tool_categories.py tests/unit/test_tool_registry.py tests/unit/test_tool_approval.py -q`
Expected: PASS — flags work, decorator threads them, real search tools citeable,
`ToolSchema.to_dict()` unchanged, existing tool tests green.

- [ ] **Step 8: Commit**

```bash
git add src/tools/base.py src/tools/registry.py src/tools/search.py tests/unit/test_tool_categories.py
git commit -m "feat(tools): add citeable/stopping category flags to canonical Tool

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

- Properties on `Tool` (spec §Approach 1) → Step 3 + `test_functiontool_defaults_false`. ✓
- `FunctionTool` params + `from_fn` (spec §Approach 2) → Step 4 + `test_functiontool_flags_settable`. ✓
- `@tool` decorator threads flags (spec §Approach 3) → Step 5 + `test_tool_decorator_threads_flags`. ✓
- Real search/web_search citeable (spec §Approach 4) → Step 6 + `test_real_search_tools_are_citeable`. ✓
- Not leaked to `ToolSchema.to_dict()` (spec non-goal) → `test_flags_not_leaked_into_schema`. ✓
- admin/streaming/built_in_tools untouched → only base/registry/search edited. ✓
