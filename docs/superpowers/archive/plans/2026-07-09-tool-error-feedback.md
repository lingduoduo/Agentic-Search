# Tool-Error Feedback Implementation Plan

> Use superpowers:test-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Feed a failed tool call back to the model as a `role:"tool"` message and continue the loop, instead of aborting the whole run.

**Architecture:** In `src/agents/tool/tool_calling.py`: a pure `_tool_message_content` helper, remove the break-on-FAILED, and route `tool_responses` through the helper. Update the one test that locked the old abort.

**Tech Stack:** Python, ToolAgentLoop.

## Global Constraints

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
    r = _result(TaskStatus.FAILED, error_code="ValueError", error_message="nope")
    assert json.loads(ToolAgentLoop._tool_message_content(r)) == {
        "status": "failed",
        "error_code": "ValueError",
        "error_message": "nope",
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/unit/test_tool_error_feedback.py -v`
Expected: FAIL — `AttributeError` for `_tool_message_content`.
(`PerformanceMetrics`/`ToolExecutionResult`/`TaskStatus` live in `src.agents.core.state`.)

- [ ] **Step 3: Add the helper**

In `ToolAgentLoop`, add (near `_skipped_tool_result` / `_truncate_tool_response`):

```python
    @staticmethod
    def _tool_message_content(result: ToolExecutionResult) -> str:
        """Serialize a tool result into the content of a role:"tool" message."""
        if result.status is TaskStatus.COMPLETED:
            return str(result.result)
        if result.status is TaskStatus.SKIPPED:
            return json.dumps(
                {"status": "skipped", "error_code": result.error_code}
            )
        payload = {"status": "failed", "error_code": result.error_code}
        if result.error_message:
            payload["error_message"] = result.error_message
        return json.dumps(payload)
```

- [ ] **Step 4: Remove the abort and route through the helper**

Delete:
```python
            if any(r.status is TaskStatus.FAILED for r in tool_execution_results):
                break
```
Replace the `tool_responses` comprehension with:
```python
            tool_responses = [
                {
                    "role": "tool",
                    "content": self._truncate_tool_response(
                        self._tool_message_content(result)
                    ),
                }
                for result in tool_execution_results
            ]
```

- [ ] **Step 5: Update the loop test that locked the old abort**

In `tests/unit/test_tool_approval.py`, replace `test_failed_tool_retains_stop_behavior`
with:

```python
async def test_failed_tool_feeds_error_back_and_continues():
    @FunctionTool.from_fn(effect=ToolEffect.READ_ONLY)
    def broken():
        raise ValueError("nope")

    loop, manager = _loop(
        [broken], ['{"name":"broken","arguments":{}}', "recovered"]
    )
    output = await loop.run([{"role": "user", "content": "go"}], {})
    # Failure is recorded...
    assert _trace(output)[0]["status"] == str(TaskStatus.FAILED)
    # ...fed back into the next prompt...
    assert "nope" in manager.prompts[-1]
    # ...the loop continued (prompted again) and produced the recovery answer.
    assert len(manager.prompts) == 2
    assert output.final_answer == "recovered"
```

(Keep the `@pytest.mark.asyncio` / async-runner decorator that the sibling async
tests in the file use — match the existing pattern above the original test.)

- [ ] **Step 6: Run new + updated + regression tests**

Run: `python3 -m pytest tests/unit/test_tool_error_feedback.py tests/unit/test_tool_approval.py tests/unit/test_on_turn_callback.py tests/unit/test_intent_routing.py -q`
Expected: PASS — pure-helper tests, the rewritten loop test, and the SKIPPED-path
tests all green.

- [ ] **Step 7: Commit**

```bash
git add src/agents/tool/tool_calling.py tests/unit/test_tool_error_feedback.py tests/unit/test_tool_approval.py
git commit -m "feat(tool): feed tool errors back to the model instead of aborting

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

- `_tool_message_content` (spec §Approach 2) → Step 3 + 3 pure tests. ✓
- Abort removed, comprehension routed through helper (spec §Approach 1,3) → Step 4. ✓
- SKIPPED/COMPLETED byte-identical (spec success criteria) → `test_skipped_format_unchanged`, `test_completed_returns_raw_result`. ✓
- FAILED fed back + loop continues (spec success criteria) → updated loop test (Step 5): trace FAILED, `"nope"` in next prompt, `prompts == 2`, recovery answer. ✓
- Bounded by turn caps (no new config) → unchanged loop caps; noted in spec. ✓
- Types consistent: `_tool_message_content(ToolExecutionResult) -> str` across def, tests, and the comprehension call. ✓
