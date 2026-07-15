# Generated Context Pack

# Simplify Search Control Flow

## Sources

- [Specification: 2026-07-03-simplify-search-control-flow-design.md](../specs/2026-07-03-simplify-search-control-flow-design.md)
- [Plan: 2026-07-03-simplify-search-control-flow.md](../plans/2026-07-03-simplify-search-control-flow.md)

## Specification Context

### Goal

After the phase-2 extractions, `run()` still carried two large inline blocks:

1. **No recognised tag** (`if not actions:`) — branches between the deterministic
   deadend auto-search (inject a search on the user's question when the model has
   never searched and we've hit the format-error limit) and the ordinary
   re-prompt via `_handle_no_action`. ~45 lines of nested conditionals.
2. **Parallel search round** (`if search_tool_call.has_new_queries:`) — per-task
   follow-up accounting, the `_execute_search_round` call, the `LoopSnapshot`
   plateau check, and the observation/`on_turn` assembly. ~80 lines.

Both entangle counter mutation with control flow, obscuring the loop's shape.
Extracting them continues the directive pattern already used for
`_handle_no_action` / `_apply_answer_gate` and leaves `run()` reading as a flat
sequence of guarded stages.

### Hard constraint: behavior-preserving

- Relevant unit suites green before and after — verbatim logic, relocated.
- The `metrics` dict consumed by `reward.py` / `action_eval.py` stays
  byte-identical (`research_followup_queries`, `plateau_early_stop`,
  `decision_prompts` bumps happen in the same order, same conditions).
- `AgentLoopOutput` fields unchanged.
- Guard conditions, branch effects, and order of operations preserved exactly.

### Testing

- **Primary gate (behavior-preserving proof):** the search/agent unit suites pass
  unchanged — `test_agent_loop`, `test_loop_controller`, `test_components`,
  `test_on_turn_callback`, `test_run_agentic_search`, `test_reward`, `test_sft`,
  `test_search_tools`, `test_search_query` (287 tests). No existing test modified.
- The full `pytest tests/unit` sweep is gated by unrelated web/model-load slow
  tests (see `project_web_test_model_load`); it is not part of this change's gate.

### Non-goals (deferred)

- The explicit state machine (`DECIDE→SEARCH→EVALUATE→ANSWER→STOP`).
- Consolidating the loop counters into a shared mutable state object — the
  directives still echo scalars back, by design.
- Any change to retrieval, plateau, or answer-gate policy.

## Implementation Plan Context

### Global Constraints

- **Behavior-preserving.** No existing test changes. Logic moves verbatim; `break`→return a BREAK directive, `continue`→return a CONTINUE directive, the auto-search fall-through→return an `injected_actions` directive.
- **`metrics` dict byte-identical.** `research_followup_queries`, `plateau_early_stop`, `decision_prompts` bumps happen in the same cases and order; `reward.py`/`action_eval.py` consume these keys.
- **`metrics`, `working_messages`, `turn_observations`, and the count dicts are passed by reference** and mutated in place; scalar counters travel back in the directive and `run()` reassigns them.
- **Guard conditions stay in `run()`** (`if not actions:`, `if search_tool_call.has_new_queries:`); only the bodies move.
- **`cfg` inside helpers is `self.search_config`.**

---

### Task 1: `_handle_absent_actions` + `_classify_planned_action`

**Files:** Modify `src/agents/search/search.py` (the `if not actions:` block and the planner-trace classification above it).

**Interfaces:**
- `_AbsentActionsDirective(control, injected_actions, exit_status, consecutive_format_errors, consecutive_rejections, forced_answer_attempted, final_answer, num_turns)`.
- `async _handle_absent_actions(self, *, working_messages, agent_ctx, request_id, sampling_params, recorder, metrics, state, question, latest_evaluation, task_statuses, active_tasks, num_turns, consecutive_format_errors, consecutive_rejections, forced_answer_attempted, final_answer) -> _AbsentActionsDirective`.
- `_classify_planned_action(action_names, search_tags, answer_tag) -> str`.

- [x] **Step 1:** Move the deadend auto-search decision + `_handle_no_action` delegation into `_handle_absent_actions`; return the directive.
- [x] **Step 2:** Replace the inline planner-trace `if/elif/else` with `_classify_planned_action`.
- [x] **Step 3:** In `run()`, apply the directive: `injected_actions`→set `actions`; else `BREAK`/`continue`.
- [x] **Verify:** search/agent unit suites green.

### Task 2: `_run_search_stage` + `_plan_only_observation`

**Files:** Modify `src/agents/search/search.py` (the `has_new_queries` block and the plan-only observation block).

**Interfaces:**
- `_SearchStageResult(plateau, consecutive_rejections, latest_evaluation)`.
- `async _run_search_stage(self, *, search_tool_call, state, recorder, num_turns, agent_ctx, search_cache, active_tasks, task_statuses, task_search_counts, metrics, response_text, turn_observations, working_messages, consecutive_rejections, on_turn) -> _SearchStageResult`.
- `_plan_only_observation(self, *, declared_subquestions, actions, decision_tag, latest_search_decision, metrics) -> str`.

- [x] **Step 1:** Move follow-up counting + `_execute_search_round` + plateau `LoopSnapshot` check + observation/`on_turn` assembly into `_run_search_stage`.
- [x] **Step 2:** Extract the plan/subquestions/decision-only observation string into `_plan_only_observation`.
- [x] **Step 3:** In `run()`, apply `_SearchStageResult` (`plateau`→`continue`).
- [x] **Verify:** search/agent unit suites green.

### Task 3: Verification

- [x] Relevant unit suites pass unchanged (287 tests: `test_agent_loop`, `test_loop_controller`, `test_components`, `test_on_turn_callback`, `test_run_agentic_search`, `test_reward`, `test_sft`, `test_search_tools`, `test_search_query`).
- [x] `ruff check` + syntax clean.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
