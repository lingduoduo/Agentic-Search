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

### Testing

- Update `test_tool_arg_validation.py` if needed (should stay green — validation
  now via `invoke`, same `error_code="invalid_arguments"` and not-executed).
- Add: unknown tool → `error_code="tool_not_found"`, not executed.
- `test_tool_approval.py`, `test_tool_registry.py`, `test_on_turn_callback.py`
  green (approval + execution unchanged for schemaless tools).

### Risks

- New registry coupling in the agent loop (accepted).
- `invoke` returns validation errors instead of raising on unknown tool — a minor
  error-code shift, covered by a test.

## Implementation Plan Context

### Global Constraints

- Continue on branch `feat/validate-tool-arguments` (extends PR #397); never commit to `main`.
- Single execution path: no direct `tool.create()/execute()/release()` in the loop.
- Preserve validation behavior (invalid args → FAILED `invalid_arguments`, not executed) and the approval flow.
- Match repo ruff formatting.

---

### Task 1: Per-loop registry + reroute `_call_tool` + tests

**Files:**
- Modify: `src/agents/tool/tool_calling.py`
- Test: `tests/unit/test_tool_arg_validation.py` (add unknown-tool case)

- [ ] **Step 1: Write the failing test (unknown tool → tool_not_found)**

Append to `tests/unit/test_tool_arg_validation.py`:

```python
@pytest.mark.asyncio
async def test_unknown_tool_reports_not_found():
    @FunctionTool.from_fn(effect=ToolEffect.READ_ONLY, parameters=_INT_SCHEMA)
    def needs_int(value: int):
        return value

    # Model calls a tool that isn't registered in the loop.
    loop, _ = _loop([needs_int], ['{"name":"ghost","arguments":{}}', "done"])
    output = await loop.run([{"role": "user", "content": "go"}], {})
    result = _trace(output)[0]
    assert result["status"] == str(TaskStatus.FAILED)
    assert result["error_code"] == "tool_not_found"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/unit/test_tool_arg_validation.py::test_unknown_tool_reports_not_found -v`
Expected: FAIL — current `_call_tool` raises `KeyError` → `error_code == "KeyError"`.

- [ ] **Step 3: Add the ToolRegistry import**

In `src/agents/tool/tool_calling.py`, add:
```python
from src.tools.registry import ToolRegistry
```

- [ ] **Step 4: Build a per-loop registry in `__init__`**

Replace:
```python
        _tools = list(tools or [])
        self.tools: dict[str, Tool] = {t.name: t for t in _tools}
        self.tool_schemas: list[dict[str, Any]] = [t.schema.to_dict() for t in _tools]
```
with:
```python
        _tools = list(tools or [])
        self._registry = ToolRegistry()
        for _t in _tools:

_[Section compacted.]_

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
