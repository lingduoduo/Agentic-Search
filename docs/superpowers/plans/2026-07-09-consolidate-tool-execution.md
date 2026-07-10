# Consolidate Tool Execution Implementation Plan

> Use superpowers:test-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Route `ToolAgentLoop` execution through a per-loop `ToolRegistry.invoke`, making it the single tool-execution path.

**Architecture:** In `src/agents/tool/tool_calling.py`: hold a per-loop `ToolRegistry` instead of a `self.tools` dict; `_call_tool` adapts `invoke`'s tuple into `ToolExecutionResult`.

**Tech Stack:** Python.

## Global Constraints

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
            self._registry.register(_t)
        self.tool_schemas: list[dict[str, Any]] = [
            t.schema.to_dict() for t in self._registry.list_tools()
        ]
```

- [ ] **Step 5: Point approval at the registry**

In `_request_approval`, replace `tool = self.tools.get(tool_call.name)` with
`tool = self._registry.get(tool_call.name)`.

- [ ] **Step 6: Reroute `_call_tool` through `invoke`**

Replace the body of `_call_tool` with:
```python
    async def _call_tool(self, tool_call: FunctionCall) -> ToolExecutionResult:
        """Execute one tool call via the per-loop registry; return a structured result."""
        start = time.perf_counter()
        args = tool_call.parsed_arguments()
        status = TaskStatus.FAILED
        result: Any = None
        error_code: str | None = None
        error_message: str | None = None
        try:
            response, _raw, errors = await self._registry.invoke(tool_call.name, args)
            if errors:
                error_code = (
                    "tool_not_found"
                    if self._registry.get(tool_call.name) is None
                    else "invalid_arguments"
                )
                error_message = "; ".join(errors)
            else:
                result = response
                status = TaskStatus.COMPLETED
                self._record_tool_stage(tool_call.name, args, result)
        except Exception as exc:
            logger.exception("Error executing tool %r: %s", tool_call.name, exc)
            error_code = type(exc).__name__
            error_message = str(exc)
        elapsed = time.perf_counter() - start
        return ToolExecutionResult(
            tool_name=tool_call.name,
            status=status,
            result=result,
            arguments=args,
            performance=PerformanceMetrics(
                execution_time=elapsed,
                success_rate=1.0 if status is TaskStatus.COMPLETED else 0.0,
            ),
            error_code=error_code,
            error_message=error_message,
        )
```

Remove any now-unused imports (e.g. `Tool` if no longer referenced; keep
`validate_arguments` only if still used — after this change it is not, so drop that
import).

- [ ] **Step 7: Run new + regression tests**

Run: `python3 -m pytest tests/unit/test_tool_arg_validation.py tests/unit/test_tool_approval.py tests/unit/test_tool_registry.py tests/unit/test_on_turn_callback.py tests/unit/test_intent_routing.py -q`
Expected: PASS — unknown-tool → `tool_not_found`; validation → `invalid_arguments`
not executed; approval + completion unchanged.

- [ ] **Step 8: Grep-verify single execution path**

Run: `grep -n "self.tools\|tool.create()\|tool.execute(\|await tool.release" src/agents/tool/tool_calling.py`
Expected: no `self.tools`; no direct `tool.create()/execute()/release()` in `_call_tool`
(those now live only in `ToolRegistry.invoke`).

- [ ] **Step 9: Commit**

```bash
git add src/agents/tool/tool_calling.py tests/unit/test_tool_arg_validation.py
git commit -m "refactor(tool): route _call_tool through a per-loop ToolRegistry

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

- Per-loop registry replaces `self.tools` (spec §Approach 1) → Step 4. ✓
- Approval via registry (spec §Approach 2) → Step 5. ✓
- `_call_tool` routes through `invoke`, no direct lifecycle (spec §Approach 3) → Step 6 + Step 8 grep. ✓
- Validation preserved; unknown-tool → `tool_not_found` (spec §Behavior) → Step 1 test + existing validation tests. ✓
- Regression (approval/registry/callbacks) → Step 7. ✓
