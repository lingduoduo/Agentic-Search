# Tool-Error Feedback (not abort) — Design

Date: 2026-07-09
Status: Approved (brainstorming)
Branch/PR: feat/tool-error-feedback
Related: [[project_chat_orchestration]] (gap #4)

## Problem

`ToolAgentLoop.run` aborts the whole trajectory on the first failed tool call:
`if any(r.status is TaskStatus.FAILED ...): break` (`src/agents/tool/tool_calling.py:384`).
The failure is **never appended** as a `role:"tool"` message, so the model can't
see the error or recover — unusual for a function-calling loop, where the standard
pattern is to feed the error back for self-correction. (SKIPPED results *are* fed
back as `{"status":"skipped","error_code":...}`; only FAILED aborts.)

Goal: feed the failure back to the model and continue the loop, bounded by the
existing turn caps.

## Non-goals

- No new failure-specific cap (`max_tool_failures`) — the loop is already bounded
  by `max_user_turns`/`max_assistant_turns` (both 10), and the model can end early
  by emitting a final answer. Deferred (YAGNI).
- No change to the SKIPPED tool-response format (kept byte-identical).
- No change to `action_trace` (it already records FAILED results).

## Approach (minimal; all in `src/agents/tool/tool_calling.py`)

1. **Delete the abort** (`:384-385`): `if any(...FAILED...): break`.

2. **Serialize a FAILED result into a tool message with the error detail** so the
   model can correct. Extract a pure helper:

   ```python
   @staticmethod
   def _tool_message_content(result: ToolExecutionResult) -> str:
       if result.status is TaskStatus.COMPLETED:
           return str(result.result)
       if result.status is TaskStatus.SKIPPED:
           return json.dumps({"status": "skipped", "error_code": result.error_code})
       payload = {"status": "failed", "error_code": result.error_code}
       if result.error_message:
           payload["error_message"] = result.error_message
       return json.dumps(payload)
   ```

   The `tool_responses` comprehension calls
   `self._truncate_tool_response(self._tool_message_content(result))`. COMPLETED
   and SKIPPED outputs are byte-identical to today; FAILED now produces
   `{"status":"failed","error_code":...,"error_message":...}` instead of aborting.

3. The loop continues naturally: the failed tool message is appended (role
   `"tool"`), re-tokenized, and the model generates again. It can retry, call a
   different tool, or answer (no tool call → break).

## Stop condition

Unchanged and already sufficient: the loop breaks at `max_user_turns` /
`max_assistant_turns` (10), or when the model emits a final answer (no tool call).
A persistently-failing tool is bounded to ≤10 turns.

## Behavior change (flagged)

For a failing-tool run, the trajectory now **continues** (the model gets a chance
to recover) instead of ending immediately. The existing test
`test_failed_tool_retains_stop_behavior` asserts the old abort
(`len(manager.prompts) == 1`); it is updated to assert the new
feed-back-and-continue behavior (this is the intended spec change, not
test-gaming).

## Success criteria

- A FAILED result is appended as a `role:"tool"` message with
  `status:"failed"`, `error_code`, and `error_message`.
- After a failure the loop continues (the model is prompted again); it can then
  produce a final answer.
- An all-failing tool run still terminates via the turn cap.
- COMPLETED and SKIPPED tool-response formats are byte-identical to before.

## Testing

- `_tool_message_content` (pure/static): COMPLETED → `str(result)`; SKIPPED →
  `{"status":"skipped","error_code":...}` (unchanged); FAILED →
  `{"status":"failed","error_code":...,"error_message":...}`.
- Loop-level (update `test_failed_tool_retains_stop_behavior` →
  `test_failed_tool_feeds_error_back_and_continues`): a broken tool then a
  recovery response — assert the trace records FAILED, the loop continued
  (`len(manager.prompts) == 2`), and the recovery answer is the final answer.
- Existing `test_tool_approval` / `test_on_turn_callback` stay green (SKIPPED path
  unchanged).

## Risks

- A model that keeps calling a broken tool burns up to `max_user_turns` turns
  before stopping — bounded, acceptable; a dedicated failure cap is the deferred
  follow-up if it proves noisy.
