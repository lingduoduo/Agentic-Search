# Consolidate Tool Execution via a Per-Loop Registry — Design

Date: 2026-07-09
Status: Approved (brainstorming)
Branch/PR: feat/validate-tool-arguments (extends PR #397)
Related: [[project_chat_orchestration]] (tool sub-item), builds on the #397 validation work

## Problem

`ToolAgentLoop` and `ToolRegistry.invoke()` are two parallel implementations of
"look up a tool → validate args → create/execute/release". After #397 they share
`validate_arguments`, but the loop still keeps its own `self.tools` dict and its own
execution lifecycle in `_call_tool`. Two code paths run tools on the same `Tool`
objects.

Goal: make `ToolRegistry.invoke()` the single execution path — the loop holds a
**per-loop `ToolRegistry`** (built from the tools passed to its constructor) and
routes execution through `invoke()`. `_call_tool` becomes a thin adapter that
shapes `invoke`'s tuple into the `ToolExecutionResult` the loop needs.

## Decision & caveats (from brainstorming)

- **Per-loop registry, NOT the global singleton.** The loop's tools come from its
  constructor, not `tool_registry`; routing through the global singleton would
  return "Tool not found" for the loop's own tools. So the loop builds its own
  `ToolRegistry` instance.
- **Accepted tradeoff:** the token-level loop now imports `ToolRegistry` (pulling
  the REST/OpenAPI registry into its import chain). Explicitly chosen for a single
  execution path.
- This supersedes #397's *inline* validation in `_call_tool` (invoke validates);
  the extracted `src/tools/validation.py` stays (invoke uses it).

## Approach (all in `src/agents/tool/tool_calling.py`)

1. **Init** — replace `self.tools: dict[str, Tool]` with a per-loop registry:
   ```python
   self._registry = ToolRegistry()
   for t in _tools:
       self._registry.register(t)
   self.tool_schemas = [t.schema.to_dict() for t in self._registry.list_tools()]
   ```

2. **Approval** — `_request_approval` uses `self._registry.get(tool_call.name)`
   instead of `self.tools.get(...)`.

3. **`_call_tool`** — route through `invoke` and adapt the result:
   ```python
   start = time.perf_counter()
   args = tool_call.parsed_arguments()
   status = TaskStatus.FAILED
   result = None
   error_code = error_message = None
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
       tool_name=tool_call.name, status=status, result=result, arguments=args,
       performance=PerformanceMetrics(
           execution_time=elapsed,
           success_rate=1.0 if status is TaskStatus.COMPLETED else 0.0,
       ),
       error_code=error_code, error_message=error_message,
   )
   ```
   `invoke` handles create/execute/release internally; the loop no longer manages
   `instance_id`. Validation happens inside `invoke` (schemaless tools skip it).

## Behavior notes

- **Validation preserved:** invalid args → `invoke` returns errors → FAILED
  `error_code="invalid_arguments"`, tool not executed (as #397).
- **Unknown tool:** was `KeyError`; now `invoke` returns "not found" →
  `error_code="tool_not_found"` (clearer than before).
- **Execute exceptions:** `invoke` does not catch them; `_call_tool`'s `try/except`
  still maps them to FAILED with the exception name (unchanged).
- COMPLETED `result` is `invoke`'s `response` text (same value as before).

## Success criteria

- One execution path: `_call_tool` calls `self._registry.invoke(...)`; no direct
  `tool.create()/execute()/release()` in the loop.
- Existing tool tests stay green: approval flow, validation (`invalid_arguments`,
  not executed), and completion all behave as before (modulo unknown-tool →
  `tool_not_found`).

## Testing

- Update `test_tool_arg_validation.py` if needed (should stay green — validation
  now via `invoke`, same `error_code="invalid_arguments"` and not-executed).
- Add: unknown tool → `error_code="tool_not_found"`, not executed.
- `test_tool_approval.py`, `test_tool_registry.py`, `test_on_turn_callback.py`
  green (approval + execution unchanged for schemaless tools).

## Risks

- New registry coupling in the agent loop (accepted).
- `invoke` returns validation errors instead of raising on unknown tool — a minor
  error-code shift, covered by a test.
