# Generated Context Pack

# Control Flow Decompose Run

## Sources

- [Specification: 2026-06-25-control-flow-decompose-run-design.md](../specs/2026-06-25-control-flow-decompose-run-design.md)
- [Plan: 2026-06-25-control-flow-decompose-run.md](../plans/2026-06-25-control-flow-decompose-run.md)

## Specification Context

### Goal

`SearchAgentLoop.run()` (`src/agents/search.py:958`) is ~500 lines after the
LoopController wiring (#333) landed. Two large blocks have clean boundaries and
contain **no loop control flow** (`break`/`continue`), so they can be extracted as
pure helper methods without restructuring control flow:

1. The per-turn model interaction (prompt build → generate → decode → parse).
2. The post-loop derived-metrics computation (~90 lines).

Extracting these shrinks `run()` from ~500 to ~390 lines and gives the biggest
block its own testable surface, at near-zero risk.

### Hard constraint: behavior-preserving

This changes **structure, not behavior**:
- Full `pytest` green before and after — the existing suite (incl. the 2116-line
  `test_agent_loop.py`) is the safety net.
- The `metrics` dict consumed by `training/reward.py` and `training/eval/action_eval.py`
  stays **byte-identical** — same keys, same values, same computation order where it
  matters. Any diff is a regression, not a refactor.
- `AgentLoopOutput` fields unchanged.

## Implementation Plan Context

### Task 1: Extract `_generate_turn`

**Files:**
- Modify: `src/agents/search.py` (`run()` per-turn block `:1001-1019`; add method near the other private helpers)
- Test: `tests/unit/test_agent_loop.py` (no new test — existing suite is the gate)

**Interfaces:**
- Produces: `async _generate_turn(self, working_messages, sampling_params, request_id, turn, metrics) -> tuple[list[int], list[int], str, list[tuple[str, str]]]` returning `(prompt_ids, response_ids, response_text, actions)`.

- [ ] **Step 1: Add the method**

Add to `SearchAgentLoop` (place it just before `run()`, after `_execute_search_round`):

- [ ] **Step 2: Replace the inline block in `run()`**

…

### Task 2: Extract `_finalize_run_metrics` + unit test

**Files:**
- Modify: `src/agents/search.py` (`run()` post-loop block `:1354-1446`; add method)
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Consumes: `self._loop_controller`, `self._has_sufficient_evidence`, `self._mark_exit`, `self.search_config`.
- Produces: `_finalize_run_metrics(self, metrics, *, rounds_used, task_statuses, task_search_counts, active_tasks, agent_ctx, final_answer, latest_evaluation, exit_status) -> None` (mutates `metrics`).

- [ ] **Step 1: Write the failing unit test**

(Use the actual dummy tokenizer/server-manager class names already in
`test_agent_loop.py` — read the file's top to match `DummyTokenizer`/`DummyServerManager`.)

…

### Task 3: Full-suite + metrics-contract verification

**Files:**
- Test: whole unit suite

- [ ] **Step 1: Full unit suite (behavior-preserving proof)**

Run: `pytest tests/unit -q`
Expected: PASS — same pass/skip counts as `main` before this branch (no behavior change).

- [ ] **Step 2: Metrics-contract check (no key meaning changed)**

Run: `grep -oE 'metrics\.get\("[a-z_]+"' src/training/reward.py src/training/eval/action_eval.py | sort -u`
Confirm every key listed is still produced by `_finalize_run_metrics` or `_initial_metrics` (read both). Report any missing key.

- [ ] **Step 3: Lint**

Run: `ruff check . --fix && ruff format .` then re-run `pytest tests/unit -q`.

- [ ] **Step 4: Commit (only if lint changed anything)**

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
