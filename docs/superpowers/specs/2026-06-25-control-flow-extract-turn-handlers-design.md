# Control-flow extract turn handlers — design

**Date:** 2026-06-25
**Status:** Approved (design); implementation plan pending.
**Stacks on:** PR #338 (`feat/control-flow-decompose-run`) — uses the
already-decomposed `run()`.
**Scope:** Phase-2 "decompose part 2": extract the two control-flow-dense blocks
(`no-action`/format-error handler, answer-gate) out of `run()` into helpers that
return a sentinel directive `run()` acts on. **Behavior-preserving.** The explicit
state machine, flag→state-object consolidation, and component unification remain
deferred.

## Goal

`run()` keeps two dense decision blocks whose every path ends in `break`/`continue`
and which mutate the same shared counters. Extracting them clarifies the loop and
is the concrete stepping stone to a state machine (the directives become
transitions). Both blocks are extracted as helpers returning a small
control-flow **directive** the caller applies.

## Hard constraint: behavior-preserving

- Full `pytest` green before and after — the existing suite is the safety net.
- The `metrics` dict consumed by `reward.py` / `action_eval.py` stays
  **byte-identical**. Verbatim logic, relocated.
- `AgentLoopOutput` fields unchanged.
- No behavior change: the guard conditions (`if not actions:`, the answer-`if`),
  every branch's effects, and the order of operations are preserved exactly.

## Design

### Shared sentinel

```python
class TurnControl(Enum):
    CONTINUE = "continue"   # caller does `continue`
    BREAK = "break"         # caller sets exit_status from the directive, then `break`
```

### Helper 1 — `_handle_no_action(...)`

Extracts the no-action/format-error body (`search.py:1155-1215`), called inside the
existing `if not actions:` guard. Performs the forced-answer hook, the rejection
feedback append, and the metric bumps internally; returns:

```python
@dataclass(frozen=True)
class _NoActionDirective:
    control: TurnControl
    exit_status: str | None
    consecutive_format_errors: int
    consecutive_rejections: int
    forced_answer_attempted: bool
    final_answer: str | None
    num_turns: int
```

Signature (keyword-only inputs it reads/needs):
`async _handle_no_action(self, *, working_messages, agent_ctx, request_id,
sampling_params, metrics, latest_evaluation, task_statuses, active_tasks,
rounds_used, consecutive_format_errors, consecutive_rejections,
forced_answer_attempted, final_answer, num_turns) -> _NoActionDirective`.

### Helper 2 — `_apply_answer_gate(...)`

Extracts the gate body (`search.py:1278-1330`), called inside the existing answer
guard. Returns:

```python
@dataclass(frozen=True)
class _GateDirective:
    control: TurnControl
    exit_status: str | None
    final_answer: str | None
    consecutive_rejections: int
```

Signature:
`async _apply_answer_gate(self, *, on_turn, num_turns, rounds_used, active_tasks,
task_statuses, latest_evaluation, latest_search_decision, consecutive_rejections,
final_answer, metrics, working_messages) -> _GateDirective`.

The direct-answer short-circuit (`allow_internal_knowledge_answer …`), the
`LoopController.final_answer_decision` call, and the ACCEPT/FORCE/REJECT branches
all move inside the helper.

### `run()` after extraction

`run()` keeps both guard conditions and applies the directive:

```python
if not actions:
    d = await self._handle_no_action(...)
    consecutive_format_errors = d.consecutive_format_errors
    consecutive_rejections = d.consecutive_rejections
    forced_answer_attempted = d.forced_answer_attempted
    final_answer = d.final_answer
    num_turns = d.num_turns
    if d.control is TurnControl.BREAK:
        exit_status = d.exit_status
        break
    continue
...
if (<answer guard>):
    d = await self._apply_answer_gate(...)
    final_answer = d.final_answer
    consecutive_rejections = d.consecutive_rejections
    if d.control is TurnControl.BREAK:
        exit_status = d.exit_status
        break
    continue
```

`metrics` and `working_messages` are mutated in place by the helpers (passed by
reference); the per-counter scalars travel back in the directive. Where a helper
does not touch a counter, that counter is not in its directive and `run()` does not
reassign it.

## Testing

- **Primary gate (behavior-preserving proof):** full unit suite passes
  **unchanged**. No existing test modified. Metrics keys/values byte-identical.
- **Added coverage:** focused unit tests for each helper driving the real method
  with a stub server_manager:
  - `_apply_answer_gate`: REJECT path (insufficient evidence, below cap) returns
    `CONTINUE`, `final_answer=None`, `consecutive_rejections` incremented; ACCEPT
    path (sufficient) returns `BREAK`, `exit_status="answered"`.
  - `_handle_no_action`: format-error-limit path returns `BREAK`,
    `exit_status="format_error_limit"`; below-limit re-prompt path returns
    `CONTINUE` with `consecutive_format_errors` incremented.

## Files touched

- **Modify:** `src/agents/search.py` — add `TurnControl`, the two directive
  dataclasses, the two helpers; replace the two inline blocks in `run()` with
  calls.
- **Test:** `tests/unit/test_agent_loop.py` — add the helper unit tests; all
  existing tests pass unchanged.

## Non-goals (deferred)

- The explicit state machine (`DECIDE→SEARCH→EVALUATE→ANSWER→STOP`). These
  directives are its precursor, not the machine.
- Consolidating the counters into a shared mutable state object (the directive
  echoes scalars instead — the deliberate consequence of the sentinel choice).
- Extracting the observation-assembly block (entangled with search-round
  execution — a later increment).
- Component unification (`Planner`/`SearchTool`/…).

## Relationship to other specs

- Second slice of `2026-06-25-control-flow-phase2-refactor-sketch.md`; stacks on the
  first slice (`2026-06-25-control-flow-decompose-run-design.md`, PR #338).
- Reuses `self._loop_controller` and `_force_final_answer` (from #333).
- Held to `2026-06-25-agent-framework-design-invariants.md`.
