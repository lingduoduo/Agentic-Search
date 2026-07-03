# Simplify Search Control Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the two remaining control-flow-dense blocks of `SearchAgentLoop.run()` — the no-recognised-tag branch (with its deadend auto-search) and the parallel-search-round-plus-plateau block — into helper methods returning a directive `run()` acts on. Continues the directive pattern from the phase-2 turn-handler extractions.

**Architecture:** Two frozen directive dataclasses (`_AbsentActionsDirective`, `_SearchStageResult`) + two async helpers (`_handle_absent_actions`, `_run_search_stage`) + two pure helpers (`_classify_planned_action`, `_plan_only_observation`) on `SearchAgentLoop` in `src/agents/search/search.py`. `run()` keeps each guard, calls the helper, reassigns returned scalars, and acts on the directive. Behavior-preserving; the relevant unit suites are the proof.

**Tech Stack:** Python 3, dataclasses + enum, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-03-simplify-search-control-flow-design.md`.

## Global Constraints

- **Behavior-preserving.** No existing test changes. Logic moves verbatim; `break`→return a BREAK directive, `continue`→return a CONTINUE directive, the auto-search fall-through→return an `injected_actions` directive.
- **`metrics` dict byte-identical.** `research_followup_queries`, `plateau_early_stop`, `decision_prompts` bumps happen in the same cases and order; `reward.py`/`action_eval.py` consume these keys.
- **`metrics`, `working_messages`, `turn_observations`, and the count dicts are passed by reference** and mutated in place; scalar counters travel back in the directive and `run()` reassigns them.
- **Guard conditions stay in `run()`** (`if not actions:`, `if search_tool_call.has_new_queries:`); only the bodies move.
- **`cfg` inside helpers is `self.search_config`.**

---

## File Structure

- **Modify** `src/agents/search/search.py` — add the two directive dataclasses, the two async helpers, the two pure helpers; replace the two inline blocks in `run()` with calls.

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
