# Simplify search control flow — design

**Date:** 2026-07-03
**Status:** Approved (design) — implementation complete on
`refactor/simplify-search-control-flow`.
**Stacks on:** the earlier turn-handler extractions
(`2026-06-25-control-flow-extract-turn-handlers-design.md`,
`2026-06-25-control-flow-decompose-run-design.md`) — same pattern, next slice.
**Scope:** Extract the two remaining control-flow-dense blocks left inline in
`SearchAgentLoop.run()` — the no-recognised-tag branch (with its deadend
auto-search) and the parallel-search-round-plus-plateau block — into helpers
that return a small directive `run()` acts on. **Behavior-preserving.**

## Goal

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

## Hard constraint: behavior-preserving

- Relevant unit suites green before and after — verbatim logic, relocated.
- The `metrics` dict consumed by `reward.py` / `action_eval.py` stays
  byte-identical (`research_followup_queries`, `plateau_early_stop`,
  `decision_prompts` bumps happen in the same order, same conditions).
- `AgentLoopOutput` fields unchanged.
- Guard conditions, branch effects, and order of operations preserved exactly.

## Design

### Helper 1 — `_handle_absent_actions(...) -> _AbsentActionsDirective`

Extracts the whole `if not actions:` body. Owns the deadend-auto-search decision;
otherwise delegates to the existing `_handle_no_action`. Returns:

```python
@dataclass(frozen=True)
class _AbsentActionsDirective:
    control: TurnControl
    injected_actions: list[tuple[str, str]] | None  # deadend auto-search
    exit_status: str | None
    consecutive_format_errors: int
    consecutive_rejections: int
    forced_answer_attempted: bool
    final_answer: str | None
    num_turns: int
```

`injected_actions is not None` means "proceed this turn with these actions"
(the auto-search path, with `consecutive_format_errors` reset to 0); otherwise
`control` says `CONTINUE` (re-prompt) or `BREAK` (set `exit_status`, stop).

### Helper 2 — `_run_search_stage(...) -> _SearchStageResult`

Extracts the `if search_tool_call.has_new_queries:` body: per-task follow-up
counting, `_execute_search_round`, the `LoopSnapshot` plateau check, and
observation/`on_turn` assembly. `turn_observations` / `working_messages` and the
count dicts are mutated in place. Returns:

```python
@dataclass(frozen=True)
class _SearchStageResult:
    plateau: bool                              # True → caller `continue`s
    consecutive_rejections: int                # reset to 0 on a search round
    latest_evaluation: SearchRoundEvaluation | None
```

### Two small pure helpers

- `_classify_planned_action(action_names, search_tags, answer_tag) -> str` —
  the `search_planned` / `answer_planned` / `turn_parsed` label for the planner
  trace record.
- `_plan_only_observation(...) -> str` — the observation string for a
  plan/subquestions/decision-only turn (no search or fetch).

### `run()` after extraction

Each block collapses to a call + directive application, e.g.:

```python
if not actions:
    d = await self._handle_absent_actions(...)
    consecutive_format_errors = d.consecutive_format_errors
    ...
    if d.injected_actions is not None:
        actions = d.injected_actions
    elif d.control is TurnControl.BREAK:
        exit_status = d.exit_status
        break
    else:
        continue
```

## Testing

- **Primary gate (behavior-preserving proof):** the search/agent unit suites pass
  unchanged — `test_agent_loop`, `test_loop_controller`, `test_components`,
  `test_on_turn_callback`, `test_run_agentic_search`, `test_reward`, `test_sft`,
  `test_search_tools`, `test_search_query` (287 tests). No existing test modified.
- The full `pytest tests/unit` sweep is gated by unrelated web/model-load slow
  tests (see `project_web_test_model_load`); it is not part of this change's gate.

## Files touched

- **Modify:** `src/agents/search/search.py` — add the two directive dataclasses,
  the two async helpers, and the two pure helpers; replace the two inline blocks
  in `run()` with calls.

## Non-goals (deferred)

- The explicit state machine (`DECIDE→SEARCH→EVALUATE→ANSWER→STOP`).
- Consolidating the loop counters into a shared mutable state object — the
  directives still echo scalars back, by design.
- Any change to retrieval, plateau, or answer-gate policy.

## Relationship to other specs

- Continues `2026-06-25-control-flow-extract-turn-handlers-design.md` and
  `2026-06-25-control-flow-decompose-run-design.md` (same directive pattern).
- Preserves the deadend auto-search behavior from
  `2026-06-28-search-agent-deterministic-auto-search-design.md`.
