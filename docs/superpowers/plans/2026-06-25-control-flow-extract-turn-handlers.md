# Extract Turn Handlers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the two control-flow-dense blocks of `SearchAgentLoop.run()` (the no-action/format-error handler and the answer-gate) into helper methods that return a `TurnControl` sentinel directive `run()` acts on.

**Architecture:** A `TurnControl` enum + two frozen directive dataclasses + two async helper methods on `SearchAgentLoop` in `src/agents/search.py`. `run()` keeps the two guard conditions, calls the helper inside each, reassigns the returned scalar counters, and acts on the directive's `control`. Behavior-preserving; the existing suite is the proof.

**Tech Stack:** Python 3, dataclasses + enum, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-25-control-flow-extract-turn-handlers-design.md`. **Stacks on PR #338** (`feat/control-flow-decompose-run`).

## Global Constraints

- **Behavior-preserving.** No existing test may change. The logic moves verbatim; `break`→`return <directive BREAK>`, `continue`→`return <directive CONTINUE>`.
- **`metrics` dict byte-identical.** All in-place `metrics[...]` bumps happen in the same cases as before; `reward.py`/`action_eval.py` consume these keys.
- **`metrics` and `working_messages` are passed by reference** and mutated in place inside the helpers; the scalar counters travel back in the directive and `run()` reassigns them.
- **Guard conditions stay in `run()`** (`if not actions:` and the answer-`if`); only the bodies move.
- **`cfg` inside helpers is `self.search_config`.**

---

## File Structure

- **Modify** `src/agents/search.py` — add `TurnControl`, `_GateDirective`, `_NoActionDirective`, `_apply_answer_gate`, `_handle_no_action`; replace the two inline blocks in `run()` with calls.
- **Test** `tests/unit/test_agent_loop.py` — add focused per-helper unit tests; existing tests pass unchanged.

---

### Task 1: `TurnControl` + `_apply_answer_gate`

**Files:**
- Modify: `src/agents/search.py` (`run()` answer-gate block `:1272-1330`; add enum/dataclass/method)
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Produces: `TurnControl` enum (`CONTINUE`, `BREAK`); `_GateDirective(control, exit_status, final_answer, consecutive_rejections)`; `async _apply_answer_gate(self, *, on_turn, num_turns, rounds_used, active_tasks, task_statuses, latest_evaluation, latest_search_decision, consecutive_rejections, final_answer, metrics, working_messages) -> _GateDirective`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_agent_loop.py
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
        consecutive_rejections=0, final_answer="ans", metrics=metrics,
        working_messages=[],
    ))
    assert d.control is TurnControl.BREAK
    assert d.exit_status == "answered"
    assert metrics["direct_answers"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_agent_loop.py -k apply_answer_gate -v`
Expected: FAIL — `ImportError`/`AttributeError` (TurnControl / `_apply_answer_gate` not defined).

- [ ] **Step 3: Implement**

Add near the top of `src/agents/search.py` (with the other module-level enums/dataclasses, after the imports):

```python
class TurnControl(Enum):
    CONTINUE = "continue"
    BREAK = "break"


@dataclass(frozen=True)
class _GateDirective:
    control: TurnControl
    exit_status: str | None
    final_answer: str | None
    consecutive_rejections: int
```

(Ensure `from enum import Enum` and `from dataclasses import dataclass` are imported — they already are in this file.)

Add the method to `SearchAgentLoop` (place it just before `run()`):

```python
    async def _apply_answer_gate(
        self,
        *,
        on_turn,
        num_turns: int,
        rounds_used: int,
        active_tasks: dict[str, str],
        task_statuses: dict[str, bool],
        latest_evaluation,
        latest_search_decision,
        consecutive_rejections: int,
        final_answer: str | None,
        metrics: dict[str, float],
        working_messages: list[dict[str, Any]],
    ) -> _GateDirective:
        """Apply the answer-gate to a turn that emitted only an <answer>.

        Caller invokes this inside the answer guard. metrics/working_messages are
        mutated in place; scalar updates travel back in the directive.
        """
        cfg = self.search_config
        if (
            cfg.allow_internal_knowledge_answer
            and rounds_used == 0
            and latest_search_decision == "answer"
            and not active_tasks
        ):
            metrics["direct_answers"] += 1.0
            if on_turn is not None:
                await on_turn(num_turns, None, 0)
            return _GateDirective(
                TurnControl.BREAK, "answered", final_answer, consecutive_rejections
            )
        snapshot = LoopSnapshot(
            rounds_used=rounds_used,
            num_subquestions=len(active_tasks),
            evidence_sufficient=self._has_sufficient_evidence(
                latest_evaluation, task_statuses, active_tasks
            ),
            prev_evidence_score=metrics["evidence_score_final"],
            curr_evidence_score=metrics["evidence_score_final"],
            consecutive_rejections=consecutive_rejections,
            model_emitted_answer=True,
        )
        decision = self._loop_controller.final_answer_decision(snapshot)
        if decision.verb is AnswerVerb.ACCEPT:
            if on_turn is not None:
                await on_turn(num_turns, None, 0)
            return _GateDirective(
                TurnControl.BREAK, "answered", final_answer, consecutive_rejections
            )
        if decision.verb is AnswerVerb.FORCE:
            metrics["forced_final_answer"] = 1.0
            if on_turn is not None:
                await on_turn(num_turns, None, 0)
            return _GateDirective(
                TurnControl.BREAK, "answered", final_answer, consecutive_rejections
            )
        # AnswerVerb.REJECT
        metrics["answer_rejections"] += 1
        working_messages.append(
            {
                "role": "user",
                "content": cfg.answer_rejection_template.format(
                    content=self._build_answer_rejection_feedback(
                        latest_evaluation, task_statuses, active_tasks
                    )
                ),
            }
        )
        return _GateDirective(
            TurnControl.CONTINUE, None, None, consecutive_rejections + 1
        )
```

Now replace the gate block in `run()` (`:1278-1330`, the body inside the `if (any(tag == answer_tag …):` guard — keep the guard) with:

```python
                    d = await self._apply_answer_gate(
                        on_turn=on_turn,
                        num_turns=num_turns,
                        rounds_used=rounds_used,
                        active_tasks=active_tasks,
                        task_statuses=task_statuses,
                        latest_evaluation=latest_evaluation,
                        latest_search_decision=latest_search_decision,
                        consecutive_rejections=consecutive_rejections,
                        final_answer=final_answer,
                        metrics=metrics,
                        working_messages=working_messages,
                    )
                    final_answer = d.final_answer
                    consecutive_rejections = d.consecutive_rejections
                    if d.control is TurnControl.BREAK:
                        exit_status = d.exit_status
                        break
                    continue
```

- [ ] **Step 4: Run the helper tests + loop suite**

Run: `pytest tests/unit/test_agent_loop.py -q`
Expected: PASS — the two new tests pass and every pre-existing test passes unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/agents/search.py tests/unit/test_agent_loop.py
git commit -m "refactor: extract _apply_answer_gate + TurnControl directive from run()"
```

---

### Task 2: `_handle_no_action`

**Files:**
- Modify: `src/agents/search.py` (`run()` no-action block `:1154-1215`; add dataclass/method)
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Consumes: `TurnControl` (Task 1), `self._force_final_answer`, `self._build_decision_feedback`, `self._has_sufficient_evidence`.
- Produces: `_NoActionDirective(control, exit_status, consecutive_format_errors, consecutive_rejections, forced_answer_attempted, final_answer, num_turns)`; `async _handle_no_action(self, *, working_messages, agent_ctx, request_id, sampling_params, metrics, latest_evaluation, task_statuses, active_tasks, rounds_used, consecutive_format_errors, consecutive_rejections, forced_answer_attempted, final_answer, num_turns) -> _NoActionDirective`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_agent_loop.py
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
            max_answer_rejections=3,
        ),
    )
    metrics = loop._initial_metrics()
    msgs = []
    d = asyncio.run(loop._handle_no_action(
        working_messages=msgs, agent_ctx=AgentContext(), request_id="r",
        sampling_params={}, metrics=metrics, latest_evaluation=None,
        task_statuses={}, active_tasks={}, rounds_used=0,
        consecutive_format_errors=0, consecutive_rejections=0,
        forced_answer_attempted=False, final_answer=None, num_turns=1,
    ))
    assert d.control is TurnControl.CONTINUE
    assert d.consecutive_rejections == 1
    assert len(msgs) == 1  # a re-prompt was appended
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_agent_loop.py -k handle_no_action -v`
Expected: FAIL — `AttributeError: _handle_no_action`.

- [ ] **Step 3: Implement**

Add the dataclass near `_GateDirective`:

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

Add the method to `SearchAgentLoop` (after `_apply_answer_gate`):

```python
    async def _handle_no_action(
        self,
        *,
        working_messages: list[dict[str, Any]],
        agent_ctx,
        request_id: str,
        sampling_params: dict[str, Any],
        metrics: dict[str, float],
        latest_evaluation,
        task_statuses: dict[str, bool],
        active_tasks: dict[str, str],
        rounds_used: int,
        consecutive_format_errors: int,
        consecutive_rejections: int,
        forced_answer_attempted: bool,
        final_answer: str | None,
        num_turns: int,
    ) -> _NoActionDirective:
        """Handle a turn that produced no recognised action tag."""
        cfg = self.search_config
        consecutive_format_errors += 1
        metrics["format_error_turns"] += 1.0
        if consecutive_format_errors >= cfg.max_consecutive_format_errors:
            exit_status = "format_error_limit"
            if cfg.force_answer_on_deadend and final_answer is None:
                forced_answer_attempted = True
                final_answer, num_turns = await self._force_final_answer(
                    working_messages=working_messages,
                    agent_ctx=agent_ctx,
                    request_id=request_id,
                    sampling_params=sampling_params,
                    metrics=metrics,
                    num_turns=num_turns,
                )
            return _NoActionDirective(
                TurnControl.BREAK, exit_status, consecutive_format_errors,
                consecutive_rejections, forced_answer_attempted, final_answer, num_turns,
            )
        needs_more = (
            cfg.require_sufficient_evidence_before_answer
            and not self._has_sufficient_evidence(
                latest_evaluation, task_statuses, active_tasks
            )
            and consecutive_rejections < cfg.max_answer_rejections
        )
        if needs_more:
            consecutive_rejections += 1
            metrics["answer_rejections"] += 1
            if rounds_used == 0:
                metrics["decision_prompts"] += 1
                feedback = self._build_decision_feedback(None)
            else:
                feedback = (
                    "No action detected. Evidence is still insufficient. "
                    "Issue a <searches> block to gather more evidence before answering."
                )
            working_messages.append(
                {
                    "role": "user",
                    "content": cfg.answer_rejection_template.format(content=feedback),
                }
            )
            return _NoActionDirective(
                TurnControl.CONTINUE, None, consecutive_format_errors,
                consecutive_rejections, forced_answer_attempted, final_answer, num_turns,
            )
        exit_status = "no_action"
        if cfg.force_answer_on_deadend and final_answer is None:
            forced_answer_attempted = True
            final_answer, num_turns = await self._force_final_answer(
                working_messages=working_messages,
                agent_ctx=agent_ctx,
                request_id=request_id,
                sampling_params=sampling_params,
                metrics=metrics,
                num_turns=num_turns,
            )
        return _NoActionDirective(
            TurnControl.BREAK, exit_status, consecutive_format_errors,
            consecutive_rejections, forced_answer_attempted, final_answer, num_turns,
        )
```

Replace the no-action block in `run()` (`:1154-1215`, the body inside `if not actions:` — keep the guard; the `consecutive_format_errors = 0` reset at `:1216` stays after it) with:

```python
                if not actions:
                    d = await self._handle_no_action(
                        working_messages=working_messages,
                        agent_ctx=agent_ctx,
                        request_id=request_id,
                        sampling_params=sampling_params,
                        metrics=metrics,
                        latest_evaluation=latest_evaluation,
                        task_statuses=task_statuses,
                        active_tasks=active_tasks,
                        rounds_used=rounds_used,
                        consecutive_format_errors=consecutive_format_errors,
                        consecutive_rejections=consecutive_rejections,
                        forced_answer_attempted=forced_answer_attempted,
                        final_answer=final_answer,
                        num_turns=num_turns,
                    )
                    consecutive_format_errors = d.consecutive_format_errors
                    consecutive_rejections = d.consecutive_rejections
                    forced_answer_attempted = d.forced_answer_attempted
                    final_answer = d.final_answer
                    num_turns = d.num_turns
                    if d.control is TurnControl.BREAK:
                        exit_status = d.exit_status
                        break
                    continue
                consecutive_format_errors = 0
```

- [ ] **Step 4: Run the helper tests + loop suite**

Run: `pytest tests/unit/test_agent_loop.py -q`
Expected: PASS — the two new tests pass and every pre-existing test passes unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/agents/search.py tests/unit/test_agent_loop.py
git commit -m "refactor: extract _handle_no_action directive from run()"
```

---

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

## Self-Review

**Spec coverage:** `TurnControl` + `_apply_answer_gate` extraction + tests (Task 1) · `_handle_no_action` extraction + tests (Task 2) · full-suite + metrics-contract gate (Task 3) · guards stay in `run()`, scalars reapplied, metrics/working_messages by reference (Global Constraints). All spec sections map to a task.

**Placeholder scan:** the helper bodies are reproduced in full (the move is a transformation — `break`→`return BREAK directive` — not a pure copy, so the exact code is shown, not referenced by line range). Test bodies use the file's real fixtures (`DummyTokenizerWithEncode`, `DummyServerManager`) — read-first pointers, concrete.

**Type consistency:** `TurnControl` (CONTINUE/BREAK), `_GateDirective(control, exit_status, final_answer, consecutive_rejections)`, `_NoActionDirective(control, exit_status, consecutive_format_errors, consecutive_rejections, forced_answer_attempted, final_answer, num_turns)`, and both helper signatures are used identically between the method defs and the `run()` call sites. The REJECT path's `metrics["answer_rejections"] += 1` (in helper) + `consecutive_rejections + 1` (in directive) matches the original's two separate mutations.
