# Control-flow decompose `run()` — design

**Date:** 2026-06-25
**Status:** Approved (design); implementation plan pending.
**Scope:** The first (smallest) lever of the deferred Phase-2 control-flow work:
extract two cohesive, control-flow-free blocks out of `SearchAgentLoop.run()` into
focused private methods. **Behavior-preserving.** The explicit state machine, flag
consolidation, and component unification remain deferred to later specs.

## Goal

`SearchAgentLoop.run()` (`src/agents/search.py:958`) is ~500 lines after the
LoopController wiring (#333) landed. Two large blocks have clean boundaries and
contain **no loop control flow** (`break`/`continue`), so they can be extracted as
pure helper methods without restructuring control flow:

1. The per-turn model interaction (prompt build → generate → decode → parse).
2. The post-loop derived-metrics computation (~90 lines).

Extracting these shrinks `run()` from ~500 to ~390 lines and gives the biggest
block its own testable surface, at near-zero risk.

## Hard constraint: behavior-preserving

This changes **structure, not behavior**:
- Full `pytest` green before and after — the existing suite (incl. the 2116-line
  `test_agent_loop.py`) is the safety net.
- The `metrics` dict consumed by `training/reward.py` and `training/eval/action_eval.py`
  stays **byte-identical** — same keys, same values, same computation order where it
  matters. Any diff is a regression, not a refactor.
- `AgentLoopOutput` fields unchanged.

## Design

### 1. `_generate_turn(...)`

Extracts the per-turn model interaction (`search.py:1001-1019`):

```python
async def _generate_turn(
    self,
    working_messages: list[dict[str, Any]],
    sampling_params: dict[str, Any],
    request_id: str,
    turn: int,
    metrics: dict[str, float],
) -> tuple[list[int], list[int], str, list[tuple[str, str]]]:
    """Build the prompt, generate, decode, and parse actions for one turn.

    Returns (prompt_ids, response_ids, response_text, actions). Side-effect-free
    on the caller's loop state; the caller applies the returned values.
    """
```

The helper owns: the two `simple_timer` blocks (`build_prompt_turn_{turn}`,
`generate_turn_{turn}` — same metric keys), `build_prompt_ids`,
`generate_response_ids`, `decode_response_ids`, `_parse_actions`, and the debug log.

`run()` keeps the local-state updates from the return value:

```python
prompt_ids, response_ids, response_text, actions = await self._generate_turn(
    working_messages, sampling_params, request_id, turn, metrics
)
final_prompt_ids = prompt_ids
all_response_ids.extend(response_ids)
num_turns += 1
working_messages.append({"role": "assistant", "content": response_text})
```

### 2. `_finalize_run_metrics(...)`

Extracts the post-loop derived-metrics block (`search.py:1354-1446`):

```python
def _finalize_run_metrics(
    self,
    metrics: dict[str, float],
    *,
    rounds_used: int,
    task_statuses: dict[str, bool],
    task_search_counts: dict[str, int],
    active_tasks: dict[str, str],
    agent_ctx: AgentContext,
    final_answer: str | None,
    latest_evaluation: SearchRoundEvaluation | None,
    exit_status: str,
) -> None:
    """Compute the derived/reward metrics and finalize exit status, in place.

    Mutates ``metrics`` (repeated_query_ratio, subquestion_coverage_ratio,
    citation_count, cited_*, answer_allowed, final_evidence_sufficient,
    useful_fetched_pages, unnecessary_fetch_count, answer_when_evidence_insufficient,
    search_budget_exhausted_without_answer, …), applies the exit_status fixups
    (max_turns→search_limit, no_action/format_error_limit→exit_no_action), and
    calls ``self._mark_exit(metrics, exit_status)``.
    """
```

- `cfg` is read via `self.search_config` (not a parameter).
- Uses `self._loop_controller.effective_search_limit`, `self._has_sufficient_evidence`,
  `self._mark_exit` — all already methods/attributes on the loop.
- The exit_status local is finalized inside the helper (run() does not need the
  post-value: `AgentLoopOutput` carries exit via the `exit_*` metric keys).

### What stays in `run()`

Everything with loop control flow or close coupling to the loop's mutable locals:
the `for turn` loop body, the answer-gate, observation assembly, dead-end handling,
the `finally:` client-close, and the post-loop forced-answer hook
(`search.py:1336-1352`, which assigns `final_answer`/`num_turns` and is control-flow
adjacent). These are candidates for the deferred state-machine spec, not this one.

## Testing

- **Primary gate (behavior-preserving proof):** the full unit suite passes
  **unchanged**. No existing test is modified; if any assertion shifts, the
  extraction changed behavior and must be fixed. Diff the `metrics` keys against the
  `reward.py`/`action_eval.py` consumers to confirm none changed meaning.
- **Added coverage:** one focused unit test feeding a synthetic `metrics` dict +
  counters to `_finalize_run_metrics` and asserting the derived keys
  (`repeated_query_ratio`, `subquestion_coverage_ratio`,
  `answer_when_evidence_insufficient`, `search_budget_exhausted_without_answer`,
  the `exit_*` fixups), so the largest extraction has direct unit coverage rather
  than integration coverage alone. Build the loop with a stub server_manager (no
  generation needed — `_finalize_run_metrics` does not call the model).

## Files touched

- **Modify:** `src/agents/search.py` — add the two methods; replace the inline
  blocks in `run()` with calls.
- **Test:** `tests/unit/test_agent_loop.py` — add the `_finalize_run_metrics` unit
  test; all existing tests pass unchanged.

## Non-goals (deferred to later specs)

- The explicit state machine (`DECIDE→SEARCH→EVALUATE→ANSWER→STOP`).
- Consolidating the ~12 control-flow flags into a state object.
- Unifying the loop onto the modular components (`Planner`/`SearchTool`/… ) — the
  inline `_parse_actions` vs `Planner.decide()` duplication stays.
- Extracting the answer-gate / observation-assembly / dead-end blocks (they contain
  `break`/`continue`; extracting them needs returned control-flow sentinels — out of
  scope for the smallest decomposition).

## Relationship to other specs

- Smallest first step of `2026-06-25-control-flow-phase2-refactor-sketch.md`.
- Successor to the LoopController (`2026-06-25-agentic-search-loop-controller-design.md`,
  merged #333) — reuses `self._loop_controller` in `_finalize_run_metrics`.
- Held to `2026-06-25-agent-framework-design-invariants.md`.
