# Generated Context Pack

# Tool Error Feedback

## Sources

- [Specification: 2026-07-09-tool-error-feedback-design.md](../specs/2026-07-09-tool-error-feedback-design.md)
- [Plan: 2026-07-09-tool-error-feedback.md](../plans/2026-07-09-tool-error-feedback.md)

## Specification Context

### Non-goals

- No new failure-specific cap (`max_tool_failures`) — the loop is already bounded
  by `max_user_turns`/`max_assistant_turns` (both 10), and the model can end early
  by emitting a final answer. Deferred (YAGNI).
- No change to the SKIPPED tool-response format (kept byte-identical).
- No change to `action_trace` (it already records FAILED results).

### Testing

- `_tool_message_content` (pure/static): COMPLETED → `str(result)`; SKIPPED →
  `{"status":"skipped","error_code":...}` (unchanged); FAILED →
  `{"status":"failed","error_code":...,"error_message":...}`.
- Loop-level (update `test_failed_tool_retains_stop_behavior` →
  `test_failed_tool_feeds_error_back_and_continues`): a broken tool then a
  recovery response — assert the trace records FAILED, the loop continued
  (`len(manager.prompts) == 2`), and the recovery answer is the final answer.
- Existing `test_tool_approval` / `test_on_turn_callback` stay green (SKIPPED path
  unchanged).

### Risks

- A model that keeps calling a broken tool burns up to `max_user_turns` turns
  before stopping — bounded, acceptable; a dedicated failure cap is the deferred
  follow-up if it proves noisy.

## Implementation Plan Context

### Global Constraints

- Branch off `main` (never commit to `main`); branch `feat/tool-error-feedback`.
- COMPLETED and SKIPPED tool-response outputs must be byte-identical to today.
- No new config; the loop stays bounded by existing `max_user_turns`/`max_assistant_turns`.
- Match repo ruff formatting.

---

### Task 1: `_tool_message_content` helper + remove abort + tests

**Files:**
- Modify: `src/agents/tool/tool_calling.py`
- Test: `tests/unit/test_tool_error_feedback.py` (new pure-helper tests) and update `tests/unit/test_tool_approval.py::test_failed_tool_retains_stop_behavior`.

**Interfaces:**
- Produces: `@staticmethod ToolAgentLoop._tool_message_content(result: ToolExecutionResult) -> str`.

- [ ] **Step 1: Write the failing pure-helper test**

Create `tests/unit/test_tool_error_feedback.py`:

```python
"""Unit tests for tool-result → tool-message formatting (error feedback)."""

from __future__ import annotations

import json

from src.agents.core.state import (
    PerformanceMetrics,
    TaskStatus,
    ToolExecutionResult,
)
from src.agents.tool.tool_calling import ToolAgentLoop


def _result(status, *, result=None, error_code=None, error_message=None):
    return ToolExecutionResult(
        tool_name="t",
        status=status,
        result=result,
        arguments={},
        performance=PerformanceMetrics(execution_time=0.0, success_rate=0.0),
        error_code=error_code,
        error_message=error_message,
    )


def test_completed_returns_raw_result():
    r = _result(TaskStatus.COMPLETED, result={"a": 1})
    assert ToolAgentLoop._tool_message_content(r) == str({"a": 1})


def test_skipped_format_unchanged():
    r = _result(TaskStatus.SKIPPED, error_code="approval_denied",
                error_message="skipped msg")
    assert ToolAgentLoop._tool_message_content(r) == json.dumps(
        {"status": "skipped", "error_code": "approval_denied"}
    )


def test_failed_includes_error_message():

_[Section compacted.]_

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
