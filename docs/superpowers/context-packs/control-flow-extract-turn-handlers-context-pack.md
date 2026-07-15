# Generated Context Pack

# Control Flow Extract Turn Handlers

## Sources

- [Specification: 2026-06-25-control-flow-extract-turn-handlers-design.md](../specs/2026-06-25-control-flow-extract-turn-handlers-design.md)
- [Plan: 2026-06-25-control-flow-extract-turn-handlers.md](../plans/2026-06-25-control-flow-extract-turn-handlers.md)

## Specification Context

### Goal

`run()` keeps two dense decision blocks whose every path ends in `break`/`continue`
and which mutate the same shared counters. Extracting them clarifies the loop and
is the concrete stepping stone to a state machine (the directives become
transitions). Both blocks are extracted as helpers returning a small
control-flow **directive** the caller applies.

### Hard constraint: behavior-preserving

- Full `pytest` green before and after — the existing suite is the safety net.
- The `metrics` dict consumed by `reward.py` / `action_eval.py` stays
  **byte-identical**. Verbatim logic, relocated.
- `AgentLoopOutput` fields unchanged.
- No behavior change: the guard conditions (`if not actions:`, the answer-`if`),
  every branch's effects, and the order of operations are preserved exactly.

## Implementation Plan Context

### Task 1: `TurnControl` + `_apply_answer_gate`

**Files:**
- Modify: `src/agents/search.py` (`run()` answer-gate block `:1272-1330`; add enum/dataclass/method)
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Produces: `TurnControl` enum (`CONTINUE`, `BREAK`); `_GateDirective(control, exit_status, final_answer, consecutive_rejections)`; `async _apply_answer_gate(self, *, on_turn, num_turns, rounds_used, active_tasks, task_statuses, latest_evaluation, latest_search_decision, consecutive_rejections, final_answer, metrics, working_messages) -> _GateDirective`.

- [ ] **Step 1: Write the failing tests**

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_agent_loop.py -k apply_answer_gate -v`

…

### Task 2: `_handle_no_action`

**Files:**
- Modify: `src/agents/search.py` (`run()` no-action block `:1154-1215`; add dataclass/method)
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Consumes: `TurnControl` (Task 1), `self._force_final_answer`, `self._build_decision_feedback`, `self._has_sufficient_evidence`.
- Produces: `_NoActionDirective(control, exit_status, consecutive_format_errors, consecutive_rejections, forced_answer_attempted, final_answer, num_turns)`; `async _handle_no_action(self, *, working_messages, agent_ctx, request_id, sampling_params, metrics, latest_evaluation, task_statuses, active_tasks, rounds_used, consecutive_format_errors, consecutive_rejections, forced_answer_attempted,

…

### Task 3: Full-suite + metrics-contract verification

**Files:**
- Test: whole unit suite

- [ ] **Step 1: Full unit suite (behavior-preserving proof)**

Run: `pytest tests/unit -q`
Expected: PASS — no behavior change; the only count increase is the 4 new helper tests.

- [ ] **Step 2: Metrics-contract check**

Run: `grep -oE 'metrics\.get\("[a-z_]+"' src/training/reward.py src/training/eval/action_eval.py | sort -u`
Confirm every key listed is still produced in `src/agents/search.py` (read `_finalize_run_metrics`, `_initial_metrics`, and the new helpers). Report any missing key.

- [ ] **Step 3: Lint**

Run: `ruff check . --fix && ruff format .` then re-run `pytest tests/unit -q`.

- [ ] **Step 4: Commit (only if lint changed anything)**

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
