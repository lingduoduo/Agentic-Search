# Spec: Agent-loop final answer + queue event delivery + RAG follow-up offload

Date: 2026-07-11

## Problem

Three verified, independent bugs across the agent-loop / streaming stack cause
dropped model output, silent stream termination, and event-loop blocking.

### A1 — Final answer dropped on turn/length-cap exit
`ToolAgentLoop.run()` (`src/agents/tool/tool_calling.py`) evaluated its three
stopping conditions (`response_length`, `max_assistant_turns`, `max_user_turns`)
*before* parsing the freshly generated response into `assistant_content`. When a
cap fired on the same turn the model produced its answer, the parse +
`working_messages.append(...)` + `final_answer = assistant_content` block never
ran, so `AgentLoopOutput.final_answer` returned `None` even though the model had
answered.

### A2 — STOP / TIMEOUT events never delivered to the consumer
`AgentQueueManager.listen()` (`src/internal/chat/queue_manager.py`), on a
self-detected timeout or stop, called `self.publish(task_id, event)` then
`break`. `publish()` enqueues the event onto the *same* queue this generator
reads, but the generator breaks immediately and never re-reads it. The SSE
consumer (`for thought in listen(...)`) saw the stream end with no STOP/TIMEOUT
reason. `publish()` also has terminal-event side effects (a `stop_listen`
sentinel) meant for the worker→queue path, redundant here.

### A3 — Blocking follow-up LLM call on the event loop, no timeout
`AgenticRAGLoop.run()` (`src/agents/search/agentic_rag.py`) awaited nothing when
calling the synchronous `_generate_followup`, which invoked `self.llm.complete`
directly — blocking the asyncio event loop with no timeout. Its sibling
`_is_sufficient` already offloads via
`asyncio.wait_for(asyncio.to_thread(...), timeout=self.config.sufficiency_timeout_s)`.

## Fix

- **A1**: Move the parse + `working_messages.append` + `final_answer` assignment
  to run immediately after `assistant_turns += 1`, before the three stopping
  checks. The stopping breaks now fire *after* the answer is recorded. The
  `if not tool_calls: break` stays after the stopping checks; tools execute only
  when no break fired. Flow: generate → record tokens → parse → record assistant
  msg + final_answer → (stopping? break) → (no tool_calls? break) → execute tools.
- **A2**: Replace `publish(...) + break` with `yield self._event(...) + break`
  for both TIMEOUT and STOP. Do not also publish. The PING branch (publishes,
  does not break) is unchanged.
- **A3**: Make `_generate_followup` async and offload with the same pattern as
  `_is_sufficient`: `asyncio.wait_for(asyncio.to_thread(self.llm.complete, ...),
  timeout=self.config.sufficiency_timeout_s)`. On timeout/exception log a warning
  and return `[]` (fail-open, matching current behavior). Update the call site to
  `await self._generate_followup(...)`. Reuse the existing `sufficiency_timeout_s`
  config field; no new field.

## Non-goals / invariants

- No behavior change to the happy path (tool calls present, no caps hit).
- Token-level truncation from `response_length` is unchanged; A1 only guarantees
  the (possibly truncated) answer is recorded rather than lost.
- No new config fields.

## Tests

- A1: caps (`max_assistant_turns=1`, `response_length`) fire on the first turn →
  `final_answer` equals the generated content and `trajectory_messages` ends with
  the assistant message.
- A2: `_is_stopped` monkeypatched True → consumer receives a STOP event before the
  generator ends; `_LISTEN_TIMEOUT_SECONDS` monkeypatched to 0 → TIMEOUT event.
- A3: a `complete` that sleeps past a tiny `sufficiency_timeout_s` → returns `[]`
  without hanging; a normal response → parsed to queries.
