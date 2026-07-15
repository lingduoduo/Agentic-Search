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

…

### Hard constraint: behavior-preserving

- Relevant unit suites green before and after — verbatim logic, relocated.
- The `metrics` dict consumed by `reward.py` / `action_eval.py` stays
  byte-identical (`research_followup_queries`, `plateau_early_stop`,
  `decision_prompts` bumps happen in the same order, same conditions).
- `AgentLoopOutput` fields unchanged.
- Guard conditions, branch effects, and order of operations preserved exactly.

## Implementation Plan Context

### Task 1: `_handle_absent_actions` + `_classify_planned_action`

**Files:** Modify `src/agents/search/search.py` (the `if not actions:` block and the planner-trace classification above it).

**Interfaces:**
- `_AbsentActionsDirective(control, injected_actions, exit_status, consecutive_format_errors, consecutive_rejections, forced_answer_attempted, final_answer, num_turns)`.
- `async _handle_absent_actions(self, *, working_messages, agent_ctx, request_id, sampling_params, recorder, metrics, state, question, latest_evaluation, task_statuses, active_tasks, num_turns, consecutive_format_errors, consecutive_rejections, forced_answer_attempted, final_answer) -> _AbsentActionsDirective`.

…

### Task 2: `_run_search_stage` + `_plan_only_observation`

**Files:** Modify `src/agents/search/search.py` (the `has_new_queries` block and the plan-only observation block).

**Interfaces:**
- `_SearchStageResult(plateau, consecutive_rejections, latest_evaluation)`.
- `async _run_search_stage(self, *, search_tool_call, state, recorder, num_turns, agent_ctx, search_cache, active_tasks, task_statuses, task_search_counts, metrics, response_text, turn_observations, working_messages, consecutive_rejections, on_turn) -> _SearchStageResult`.
- `_plan_only_observation(self, *, declared_subquestions, actions, decision_tag, latest_search_decision, metrics) -> str`.

…

### Task 3: Verification

- [x] Relevant unit suites pass unchanged (287 tests: `test_agent_loop`, `test_loop_controller`, `test_components`, `test_on_turn_callback`, `test_run_agentic_search`, `test_reward`, `test_sft`, `test_search_tools`, `test_search_query`).
- [x] `ruff check` + syntax clean.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
