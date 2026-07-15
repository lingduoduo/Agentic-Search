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

### Testing

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

### Non-goals (deferred)

- The explicit state machine (`DECIDE→SEARCH→EVALUATE→ANSWER→STOP`). These
  directives are its precursor, not the machine.
- Consolidating the counters into a shared mutable state object (the directive
  echoes scalars instead — the deliberate consequence of the sentinel choice).
- Extracting the observation-assembly block (entangled with search-round
  execution — a later increment).
- Component unification (`Planner`/`SearchTool`/…).

## Implementation Plan Context

### Global Constraints

- **Behavior-preserving.** No existing test may change. The logic moves verbatim; `break`→`return <directive BREAK>`, `continue`→`return <directive CONTINUE>`.
- **`metrics` dict byte-identical.** All in-place `metrics[...]` bumps happen in the same cases as before; `reward.py`/`action_eval.py` consume these keys.
- **`metrics` and `working_messages` are passed by reference** and mutated in place inside the helpers; the scalar counters travel back in the directive and `run()` reassigns them.
- **Guard conditions stay in `run()`** (`if not actions:` and the answer-`if`); only the bodies move.
- **`cfg` inside helpers is `self.search_config`.**

---

### Task 1: `TurnControl` + `_apply_answer_gate`

**Files:**
- Modify: `src/agents/search.py` (`run()` answer-gate block `:1272-1330`; add enum/dataclass/method)
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Produces: `TurnControl` enum (`CONTINUE`, `BREAK`); `_GateDirective(control, exit_status, final_answer, consecutive_rejections)`; `async _apply_answer_gate(self, *, on_turn, num_turns, rounds_used, active_tasks, task_statuses, latest_evaluation, latest_search_decision, consecutive_rejections, final_answer, metrics, working_messages) -> _GateDirective`.

- [ ] **Step 1: Write the failing tests**

```python

### tests/unit/test_agent_loop.py

def _gate_loop():
    from src.agents.search import SearchAgentLoop, SearchAgentLoopConfig
    return SearchAgentLoop(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=DummyServerManager([]),
        search_config=SearchAgentLoopConfig(max_answer_rejections=3),
    )

def test_apply_answer_gate_rejects_insufficient_evidence():
    import asyncio
    from src.agents.search import TurnControl
    loop = _gate_loop()
    metrics = loop._initial_metrics()
    d = asyncio.run(loop._apply_answer_gate(
        on_turn=None, num_turns=1, rounds_used=1, active_tasks={}, task_statuses={},
        latest_evaluation=None, latest_search_decision=None,
        consecutive_rejections=0, final_answer="draft", metrics=metrics,
        working_messages=[],
    ))
    assert d.control is TurnControl.CONTINUE
    assert d.final_answer is None
    assert d.consecutive_rejections == 1
    assert metrics["answer_rejections"] == 1.0

def test_apply_answer_gate_accepts_with_internal_knowledge():
    import asyncio
    from src.agents.search import TurnControl, SearchAgentLoopConfig, SearchAgentLoop
    loop = SearchAgentLoop(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=DummyServerManager([]),
        search_config=SearchAgentLoopConfig(allow_internal_knowledge_answer=True),
    )
    metrics = loop._initial_metrics()
    d = asyncio.run(loop._apply_answer_gate(
        on_turn=None, num_turns=1, rounds_used=0, active_tasks={}, task_statuses={},
        latest_evaluation=None, latest_search_decision="answer",

_[Section compacted.]_

### Task 2: `_handle_no_action`

**Files:**
- Modify: `src/agents/search.py` (`run()` no-action block `:1154-1215`; add dataclass/method)
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Consumes: `TurnControl` (Task 1), `self._force_final_answer`, `self._build_decision_feedback`, `self._has_sufficient_evidence`.
- Produces: `_NoActionDirective(control, exit_status, consecutive_format_errors, consecutive_rejections, forced_answer_attempted, final_answer, num_turns)`; `async _handle_no_action(self, *, working_messages, agent_ctx, request_id, sampling_params, metrics, latest_evaluation, task_statuses, active_tasks, rounds_used, consecutive_format_errors, consecutive_rejections, forced_answer_attempted, final_answer, num_turns) -> _NoActionDirective`.

- [ ] **Step 1: Write the failing tests**

```python

### tests/unit/test_agent_loop.py

def test_handle_no_action_format_error_limit_breaks():
    import asyncio
    from src.agents.search import (
        SearchAgentLoop, SearchAgentLoopConfig, TurnControl,
    )
    from src.context.search import AgentContext
    loop = SearchAgentLoop(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=DummyServerManager([]),
        search_config=SearchAgentLoopConfig(
            max_consecutive_format_errors=1, force_answer_on_deadend=False
        ),
    )
    metrics = loop._initial_metrics()
    d = asyncio.run(loop._handle_no_action(
        working_messages=[], agent_ctx=AgentContext(), request_id="r",
        sampling_params={}, metrics=metrics, latest_evaluation=None,
        task_statuses={}, active_tasks={}, rounds_used=1,
        consecutive_format_errors=0, consecutive_rejections=0,
        forced_answer_attempted=False, final_answer=None, num_turns=1,
    ))
    assert d.control is TurnControl.BREAK
    assert d.exit_status == "format_error_limit"
    assert d.consecutive_format_errors == 1
    assert metrics["format_error_turns"] == 1.0

def test_handle_no_action_below_limit_reprompts_continue():
    import asyncio
    from src.agents.search import (
        SearchAgentLoop, SearchAgentLoopConfig, TurnControl,
    )
    from src.context.search import AgentContext
    loop = SearchAgentLoop(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=DummyServerManager([]),
        search_config=SearchAgentLoopConfig(
            max_consecutive_format_errors=5,
            require_sufficient_evidence_before_answer=True,

_[Section compacted.]_

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

```bash
git add -A
git commit -m "chore: lint after turn-handler extraction"
```

(If nothing changed, skip.)

---

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
