# Agentic Search LoopController Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract a stateless `LoopController` that owns the search loop's two decisions (keep-searching? / how-to-answer?) and ship four default-on control-flow behaviors on top: plateau early-stop, graceful dead-end answer, smarter answer-gating, adaptive search budget.

**Architecture:** A new pure-over-snapshot `LoopController` (no I/O, no model calls) holds only config and exposes three methods. `SearchAgentLoop.run()` stays the single state owner and consults the controller at three points, replacing scattered inline flag-logic. Behavior changes are default-on; metrics are added (never mutated) and the reward gains one additive penalty.

**Tech Stack:** Python 3, dataclasses + enums, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-25-agentic-search-loop-controller-design.md`

## Global Constraints

- **Add metrics, never mutate existing key semantics.** New keys only; existing keys in `_initial_metrics` (`src/agents/search.py:344`) keep their meaning.
- **`forced_final_answer` is mutually exclusive with `answer_when_evidence_insufficient`** — a forced salvage sets the former and must NOT set the latter.
- **Reward `_zeroed` preset stays byte-stable** — any new penalty defaults to a non-zero serving value but is set to `0.0` in the `_zeroed` preset.
- **Plateau early-stop is conservative:** `evidence_plateau_min_gain = 0.05`, gated on `plateau_requires_sufficient = True` (only stops when evidence already sufficient).
- **Adaptive budget keyed on subquestion count only**; single-subquestion questions get today's limit unchanged.
- **Full `pytest` green before every commit after the first.**
- **The loop preserves append-only history** — the forced-answer turn appends messages; it never branches or rewrites history.

---

## File Structure

- **Create** `src/agents/components/loop_controller.py` — `LoopSnapshot`, `StopReason`, `AnswerVerb`, `StopDecision`, `AnswerDecision`, `LoopController`. One responsibility: the two control decisions, pure over a snapshot.
- **Create** `tests/unit/test_loop_controller.py` — pure unit tests for the controller (no tokenizer/server).
- **Modify** `src/agents/search.py` — add config knobs (`SearchAgentLoopConfig`, ~`:235`), new metric keys (`_initial_metrics`, `:344`), instantiate the controller in `__init__` (`:251`), and wire it into `run()` (`:900`) at three points.
- **Modify** `src/training/reward.py` — add `forced_final_answer_penalty` (near `:218`), price it, add it to `_zeroed` (`:282`).
- **Modify** `tests/unit/test_agent_loop.py` — integration tests + baseline updates for shifted metrics.
- **Modify** `tests/unit/test_reward.py` — pricing test for the new penalty + `_zeroed` stability.

---

### Task 1: Config knobs + metric keys

**Files:**
- Modify: `src/agents/search.py` (`SearchAgentLoopConfig` ends ~`:235`; `_initial_metrics` `:344-378`)
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Produces: new `SearchAgentLoopConfig` fields `evidence_plateau_min_gain: float = 0.05`, `plateau_requires_sufficient: bool = True`, `search_budget_per_subquestion: int = 1`, `max_search_limit_cap: int = 10`, `force_answer_on_deadend: bool = True`. New metric keys `forced_final_answer`, `plateau_early_stop`, `effective_search_limit`, `adaptive_budget_bonus` (all `0.0`).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_agent_loop.py
from src.agents.search import SearchAgentLoopConfig

def test_loop_controller_config_defaults():
    cfg = SearchAgentLoopConfig()
    assert cfg.evidence_plateau_min_gain == 0.05
    assert cfg.plateau_requires_sufficient is True
    assert cfg.search_budget_per_subquestion == 1
    assert cfg.max_search_limit_cap == 10
    assert cfg.force_answer_on_deadend is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_agent_loop.py::test_loop_controller_config_defaults -v`
Expected: FAIL — `AttributeError`/`TypeError` (fields not defined).

- [ ] **Step 3: Implement**

In `src/agents/search.py`, replace the existing `evidence_plateau_min_gain` line in `SearchAgentLoopConfig` (currently `evidence_plateau_min_gain: float | None = None` at ~`:235`) and add the new fields directly after it:

```python
    # Plateau early-stop: stop searching when a round's evidence gain falls below
    # this threshold. Default-on (0.05). plateau_requires_sufficient gates it so a
    # plateau only stops the loop when evidence is already sufficient.
    evidence_plateau_min_gain: float | None = 0.05
    plateau_requires_sufficient: bool = True
    # Adaptive search budget: +N rounds per extra subquestion, capped.
    search_budget_per_subquestion: int = 1
    max_search_limit_cap: int = 10
    # Dead-ends emit a best-effort answer from collected evidence instead of None.
    force_answer_on_deadend: bool = True
```

In `_initial_metrics` (`:345`), add these keys before the closing `}`:

```python
            "forced_final_answer": 0.0,
            "plateau_early_stop": 0.0,
            "effective_search_limit": 0.0,
            "adaptive_budget_bonus": 0.0,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_agent_loop.py::test_loop_controller_config_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/search.py tests/unit/test_agent_loop.py
git commit -m "feat: add LoopController config knobs and metric keys"
```

---

### Task 2: LoopController types + `effective_search_limit`

**Files:**
- Create: `src/agents/components/loop_controller.py`
- Test: `tests/unit/test_loop_controller.py`

**Interfaces:**
- Consumes: `SearchAgentLoopConfig` (Task 1 fields).
- Produces: `LoopSnapshot` (frozen dataclass, 7 fields), `StopReason`/`AnswerVerb` enums, `StopDecision`/`AnswerDecision`, `LoopController(cfg)` with `effective_search_limit(num_subquestions: int) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_loop_controller.py
from src.agents.components.loop_controller import LoopController
from src.agents.search import SearchAgentLoopConfig

def _ctl(**over):
    return LoopController(SearchAgentLoopConfig(**over))

def test_effective_limit_single_subquestion_unchanged():
    ctl = _ctl(max_search_limit=4, search_budget_per_subquestion=1, max_search_limit_cap=10)
    assert ctl.effective_search_limit(1) == 4
    assert ctl.effective_search_limit(0) == 4

def test_effective_limit_scales_then_caps():
    ctl = _ctl(max_search_limit=4, search_budget_per_subquestion=1, max_search_limit_cap=6)
    assert ctl.effective_search_limit(3) == 6     # 4 + (3-1)=6
    assert ctl.effective_search_limit(20) == 6    # capped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_loop_controller.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement**

```python
# src/agents/components/loop_controller.py
"""LoopController: the search loop's two control decisions, pure over a snapshot.

Owns no mutable state (only config). ``SearchAgentLoop.run`` builds a
``LoopSnapshot`` of the relevant loop state and consults the controller, keeping
all mutable state in the loop itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class LoopSnapshot:
    rounds_used: int
    num_subquestions: int
    evidence_sufficient: bool
    prev_evidence_score: float
    curr_evidence_score: float
    consecutive_rejections: int
    model_emitted_answer: bool


class StopReason(Enum):
    CONTINUE = "continue"
    BUDGET_EXHAUSTED = "budget_exhausted"
    PLATEAU = "plateau"


class AnswerVerb(Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    FORCE = "force"


@dataclass(frozen=True)
class StopDecision:
    reason: StopReason


@dataclass(frozen=True)
class AnswerDecision:
    verb: AnswerVerb
    feedback: str = ""


class LoopController:
    """Stateless policy for the loop's keep-searching / how-to-answer decisions."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg

    def effective_search_limit(self, num_subquestions: int) -> int:
        cfg = self._cfg
        base = cfg.max_search_limit or cfg.max_turns
        bonus = cfg.search_budget_per_subquestion * max(0, num_subquestions - 1)
        return max(base, min(base + bonus, cfg.max_search_limit_cap))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_loop_controller.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add src/agents/components/loop_controller.py tests/unit/test_loop_controller.py
git commit -m "feat: LoopController types and effective_search_limit"
```

---

### Task 3: `should_continue_searching`

**Files:**
- Modify: `src/agents/components/loop_controller.py`
- Test: `tests/unit/test_loop_controller.py`

**Interfaces:**
- Produces: `LoopController.should_continue_searching(s: LoopSnapshot) -> StopDecision`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_loop_controller.py
from src.agents.components.loop_controller import LoopSnapshot, StopReason

def _snap(**over):
    base = dict(rounds_used=1, num_subquestions=1, evidence_sufficient=True,
                prev_evidence_score=0.5, curr_evidence_score=0.9,
                consecutive_rejections=0, model_emitted_answer=False)
    base.update(over)
    return LoopSnapshot(**base)

def test_continue_while_evidence_climbing():
    ctl = _ctl(max_search_limit=5, evidence_plateau_min_gain=0.05)
    d = ctl.should_continue_searching(_snap(prev_evidence_score=0.2, curr_evidence_score=0.9))
    assert d.reason is StopReason.CONTINUE

def test_stop_budget_exhausted():
    ctl = _ctl(max_search_limit=2, max_search_limit_cap=10)
    d = ctl.should_continue_searching(_snap(rounds_used=2, num_subquestions=1))
    assert d.reason is StopReason.BUDGET_EXHAUSTED

def test_plateau_stops_only_when_sufficient():
    ctl = _ctl(max_search_limit=5, evidence_plateau_min_gain=0.05, plateau_requires_sufficient=True)
    stalled = dict(prev_evidence_score=0.80, curr_evidence_score=0.82)  # gain 0.02 < 0.05
    assert ctl.should_continue_searching(_snap(evidence_sufficient=True, **stalled)).reason is StopReason.PLATEAU
    assert ctl.should_continue_searching(_snap(evidence_sufficient=False, **stalled)).reason is StopReason.CONTINUE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_loop_controller.py -k should_continue or continue or budget or plateau -v`
Expected: FAIL — `AttributeError: should_continue_searching`.

- [ ] **Step 3: Implement**

Append to `LoopController` in `src/agents/components/loop_controller.py`:

```python
    def should_continue_searching(self, s: LoopSnapshot) -> StopDecision:
        cfg = self._cfg
        if s.rounds_used >= self.effective_search_limit(s.num_subquestions):
            return StopDecision(StopReason.BUDGET_EXHAUSTED)
        if cfg.evidence_plateau_min_gain is not None:
            gain = s.curr_evidence_score - s.prev_evidence_score
            if gain < cfg.evidence_plateau_min_gain and (
                s.evidence_sufficient or not cfg.plateau_requires_sufficient
            ):
                return StopDecision(StopReason.PLATEAU)
        return StopDecision(StopReason.CONTINUE)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_loop_controller.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/components/loop_controller.py tests/unit/test_loop_controller.py
git commit -m "feat: LoopController.should_continue_searching (budget + plateau)"
```

---

### Task 4: `final_answer_decision`

**Files:**
- Modify: `src/agents/components/loop_controller.py`
- Test: `tests/unit/test_loop_controller.py`

**Interfaces:**
- Produces: `LoopController.final_answer_decision(s: LoopSnapshot) -> AnswerDecision`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_loop_controller.py
from src.agents.components.loop_controller import AnswerVerb

def test_accept_when_sufficient():
    ctl = _ctl(require_sufficient_evidence_before_answer=True, max_answer_rejections=3)
    assert ctl.final_answer_decision(_snap(evidence_sufficient=True)).verb is AnswerVerb.ACCEPT

def test_reject_then_force_at_cap():
    ctl = _ctl(require_sufficient_evidence_before_answer=True, max_answer_rejections=3)
    assert ctl.final_answer_decision(_snap(evidence_sufficient=False, consecutive_rejections=1)).verb is AnswerVerb.REJECT
    assert ctl.final_answer_decision(_snap(evidence_sufficient=False, consecutive_rejections=3)).verb is AnswerVerb.FORCE

def test_accept_when_gate_disabled():
    ctl = _ctl(require_sufficient_evidence_before_answer=False)
    assert ctl.final_answer_decision(_snap(evidence_sufficient=False)).verb is AnswerVerb.ACCEPT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_loop_controller.py -k final_answer or accept or reject -v`
Expected: FAIL — `AttributeError: final_answer_decision`.

- [ ] **Step 3: Implement**

Append to `LoopController`:

```python
    _REJECT_FEEDBACK = (
        "Evidence is still insufficient for the question. Issue another search "
        "to gather more evidence before answering."
    )
    _FORCE_FEEDBACK = (
        "You cannot gather more evidence (budget reached). Give your best answer "
        "now, grounded only in the evidence already collected. State explicitly "
        "what remains uncertain, and cite evidence labels."
    )

    def final_answer_decision(self, s: LoopSnapshot) -> AnswerDecision:
        cfg = self._cfg
        if s.evidence_sufficient or not cfg.require_sufficient_evidence_before_answer:
            return AnswerDecision(AnswerVerb.ACCEPT)
        if s.consecutive_rejections >= cfg.max_answer_rejections:
            return AnswerDecision(AnswerVerb.FORCE, feedback=self._FORCE_FEEDBACK)
        return AnswerDecision(AnswerVerb.REJECT, feedback=self._REJECT_FEEDBACK)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_loop_controller.py -v`
Expected: PASS (all controller tests)

- [ ] **Step 5: Commit**

```bash
git add src/agents/components/loop_controller.py tests/unit/test_loop_controller.py
git commit -m "feat: LoopController.final_answer_decision (accept/reject/force)"
```

---

### Task 5: Instantiate controller + wire adaptive budget

**Files:**
- Modify: `src/agents/search.py` (`__init__` `:251`; `run()` budget check — the `_partition_search_requests` call uses `rounds_used` vs `max_search_limit`, `:1031`)
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Consumes: `LoopController` (Tasks 2-4).
- Produces: `self._loop_controller` on `SearchAgentLoop`; `metrics["effective_search_limit"]` / `metrics["adaptive_budget_bonus"]` populated per run.

- [ ] **Step 1: Write the failing test**

Follow the existing fake-model harness in `tests/unit/test_agent_loop.py` (a scripted `server_manager` returning canned responses). Add:

```python
def test_adaptive_budget_raises_limit_for_multiple_subquestions(make_loop):
    # make_loop is the existing helper that builds SearchAgentLoop with a fake
    # server. Script: declare 3 subquestions, then keep searching.
    loop = make_loop(max_search_limit=2, search_budget_per_subquestion=1, max_search_limit_cap=10,
                     responses=SCRIPT_THREE_SUBQUESTIONS_THEN_SEARCHES)
    out = run_loop(loop)
    assert out.metrics["effective_search_limit"] == 4.0   # base 2 + (3-1)
    assert out.metrics["adaptive_budget_bonus"] == 2.0
```

(Reuse whatever `make_loop`/`run_loop` fixtures already exist in the file; mirror an existing multi-subquestion test for the response script.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_agent_loop.py::test_adaptive_budget_raises_limit_for_multiple_subquestions -v`
Expected: FAIL — metric is `0.0` (limit not adaptive yet).

- [ ] **Step 3: Implement**

In `SearchAgentLoop.__init__` (`src/agents/search.py`, after `self.search_config = cfg` at `:280`), add:

```python
        from .components.loop_controller import LoopController
        self._loop_controller = LoopController(cfg)
```

In `run()`, where the effective limit is needed for the search-budget decision, compute it from the controller using the current subquestion count and record the metrics. Replace the use of `rounds_used` against `cfg.max_search_limit` inside `_partition_search_requests` by passing an `effective_limit`. Concretely, just before the `_partition_search_requests` call (`:1031`):

```python
                effective_limit = self._loop_controller.effective_search_limit(
                    len(active_tasks)
                )
                metrics["effective_search_limit"] = float(effective_limit)
                metrics["adaptive_budget_bonus"] = float(
                    effective_limit - (cfg.max_search_limit or cfg.max_turns)
                )
```

Then change `_partition_search_requests` (`:501`) to accept and use `effective_limit` instead of reading `cfg.max_search_limit` for the overflow decision. Update its signature and the single call site.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_agent_loop.py::test_adaptive_budget_raises_limit_for_multiple_subquestions -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/search.py tests/unit/test_agent_loop.py
git commit -m "feat: wire adaptive search budget via LoopController"
```

---

### Task 6: Wire plateau early-stop

**Files:**
- Modify: `src/agents/search.py` (`run()` — after the search round at `:1155-1166`)
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Consumes: `should_continue_searching`, `metrics["evidence_score_final"]`.
- Produces: `metrics["plateau_early_stop"]`; loop stops issuing searches after a plateau.

- [ ] **Step 1: Write the failing test**

```python
def test_plateau_stops_searching_when_sufficient(make_loop):
    # Script: evidence becomes sufficient and stalls (gain < 0.05) across rounds.
    loop = make_loop(max_search_limit=5, evidence_plateau_min_gain=0.05,
                     plateau_requires_sufficient=True,
                     responses=SCRIPT_STALLED_SUFFICIENT_EVIDENCE)
    out = run_loop(loop)
    assert out.metrics["plateau_early_stop"] == 1.0
    assert out.metrics["rounds_used"] < 5.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_agent_loop.py::test_plateau_stops_searching_when_sufficient -v`
Expected: FAIL — `plateau_early_stop` stays `0.0`, `rounds_used` reaches 5.

- [ ] **Step 3: Implement**

In `run()`, immediately after `latest_evaluation = round_result.evaluation` (`:1165`), build a snapshot and consult the controller. `prev_evidence_score` is the value of `metrics["evidence_score_final"]` captured *before* this round executed; capture it just before the `_execute_search_round` call (`:1155`) into a local `prev_evidence_for_round`. Then:

```python
                from .components.loop_controller import LoopSnapshot, StopReason
                snapshot = LoopSnapshot(
                    rounds_used=rounds_used,
                    num_subquestions=len(active_tasks),
                    evidence_sufficient=self._has_sufficient_evidence(
                        latest_evaluation, task_statuses, active_tasks
                    ),
                    prev_evidence_score=prev_evidence_for_round,
                    curr_evidence_score=metrics["evidence_score_final"],
                    consecutive_rejections=consecutive_rejections,
                    model_emitted_answer=False,
                )
                stop = self._loop_controller.should_continue_searching(snapshot)
                if stop.reason is StopReason.PLATEAU:
                    metrics["plateau_early_stop"] = 1.0
                    working_messages.append(
                        {"role": "user", "content": cfg.search_limit_template}
                    )
                    continue
```

This appends the existing search-limit observation (so the model moves to answer) and skips further searching this turn — evidence is already sufficient, so the model's next `<answer>` passes the gate normally (ACCEPT path, per spec).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_agent_loop.py::test_plateau_stops_searching_when_sufficient -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/search.py tests/unit/test_agent_loop.py
git commit -m "feat: plateau early-stop via LoopController"
```

---

### Task 7: Wire forced-final-answer (graceful dead-end)

**Files:**
- Modify: `src/agents/search.py` (`run()` dead-end exits: no-action `:1001`, format-error `:970-972`, post-loop budget exit `:1281`)
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Consumes: `force_answer_on_deadend`, `agent_ctx` (evidence presence), `final_answer`.
- Produces: `metrics["forced_final_answer"]`; a bounded single forced generation; mutual exclusivity with `answer_when_evidence_insufficient`.

- [ ] **Step 1: Write the failing test**

```python
def test_deadend_forces_answer_from_evidence(make_loop):
    # Script: one good search round (evidence collected), then a dead-end
    # (no recognised action) repeatedly.
    loop = make_loop(max_search_limit=5, force_answer_on_deadend=True,
                     responses=SCRIPT_SEARCH_THEN_DEADEND)
    out = run_loop(loop)
    assert out.final_answer is not None
    assert out.metrics["forced_final_answer"] == 1.0
    assert out.metrics["search_budget_exhausted_without_answer"] == 0.0
    # Mutual exclusivity: a forced answer is NOT also counted as voluntary-insufficient.
    assert out.metrics["answer_when_evidence_insufficient"] == 0.0

def test_deadend_with_no_evidence_does_not_fabricate(make_loop):
    loop = make_loop(responses=SCRIPT_IMMEDIATE_FORMAT_ERRORS)
    out = run_loop(loop)
    assert out.metrics["forced_final_answer"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_agent_loop.py -k deadend -v`
Expected: FAIL — `final_answer` is `None`, `forced_final_answer` is `0.0`.

- [ ] **Step 3: Implement**

Add a helper on `SearchAgentLoop` (near the other `_build_*` helpers):

```python
    async def _force_final_answer(
        self,
        *,
        working_messages: list[dict[str, Any]],
        agent_ctx: AgentContext,
        request_id: str,
        sampling_params: dict[str, Any],
        metrics: dict[str, float],
        num_turns: int,
    ) -> tuple[str | None, int]:
        """One bounded generation that forces a best-effort answer from evidence.

        Returns (answer_or_None, new_num_turns). No-op (returns (None, num_turns))
        when no evidence was collected — never fabricates.
        """
        if agent_ctx.num_rounds == 0:
            return None, num_turns
        working_messages.append(
            {"role": "user", "content": self.search_config.answer_rejection_template.format(
                content=self._loop_controller._FORCE_FEEDBACK
            )}
        )
        prompt_ids = await self.build_prompt_ids(working_messages)
        response_ids = await self.generate_response_ids(
            prompt_ids=prompt_ids, sampling_params=sampling_params,
            request_id=f"{request_id}_force",
        )
        num_turns += 1
        text = self.decode_response_ids(response_ids)
        working_messages.append({"role": "assistant", "content": text})
        answer_actions = [c for t, c in self._parse_actions(text) if t == self.search_config.answer_tag]
        if answer_actions:
            metrics["forced_final_answer"] = 1.0
            return answer_actions[0].strip(), num_turns
        return None, num_turns  # bounded fallback handled by caller's tentative answer
```

At each dead-end exit in `run()` (the `no_action` break `:1001`, the `format_error_limit` break `:970-972`), before breaking, when `cfg.force_answer_on_deadend` and `final_answer is None`, call `_force_final_answer` and set `final_answer` to its result if non-None. In the post-loop derived-metrics block, set the mutual-exclusivity guard: where `answer_when_evidence_insufficient` is computed (`:1273`), add `and metrics["forced_final_answer"] == 0.0` to its condition so a forced answer never also counts as voluntary-insufficient.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_agent_loop.py -k deadend -v`
Expected: PASS (both)

- [ ] **Step 5: Commit**

```bash
git add src/agents/search.py tests/unit/test_agent_loop.py
git commit -m "feat: graceful dead-end forced-final-answer via LoopController"
```

---

### Task 8: Route the answer-gate through `final_answer_decision`

**Files:**
- Modify: `src/agents/search.py` (`run()` answer-gate block `:1052-1094`)
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Consumes: `final_answer_decision`.
- Produces: gate decisions (ACCEPT/REJECT/FORCE) come from the controller; behavior matches today's reject-then-force, now via FORCE.

- [ ] **Step 1: Write the failing test**

```python
def test_answer_gate_forces_after_max_rejections(make_loop):
    # Script: model answers with insufficient evidence repeatedly.
    loop = make_loop(require_sufficient_evidence_before_answer=True, max_answer_rejections=2,
                     responses=SCRIPT_PERSISTENT_INSUFFICIENT_ANSWER)
    out = run_loop(loop)
    assert out.final_answer is not None        # forced through after cap
    assert out.metrics["forced_final_answer"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_agent_loop.py::test_answer_gate_forces_after_max_rejections -v`
Expected: FAIL — `forced_final_answer` is `0.0` (old path force-passes without flagging).

- [ ] **Step 3: Implement**

In the answer-gate block (`:1052`), replace the hand-rolled `require_sufficient_evidence_before_answer` + `consecutive_rejections >= max_answer_rejections` branching with a controller call:

```python
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
                        ...  # existing accept path: on_turn + exit_status="answered"; break
                    elif decision.verb is AnswerVerb.FORCE:
                        metrics["forced_final_answer"] = 1.0
                        ...  # accept current final_answer unconditionally; break
                    else:  # REJECT
                        final_answer = None
                        consecutive_rejections += 1
                        metrics["answer_rejections"] += 1
                        working_messages.append({"role": "user", "content":
                            cfg.answer_rejection_template.format(content=decision.feedback)})
                        continue
```

Preserve the existing `allow_internal_knowledge_answer` direct-answer short-circuit (`:1057-1067`) ahead of this block unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_agent_loop.py::test_answer_gate_forces_after_max_rejections -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/search.py tests/unit/test_agent_loop.py
git commit -m "feat: route answer-gate through LoopController.final_answer_decision"
```

---

### Task 9: Reward — `forced_final_answer_penalty`

**Files:**
- Modify: `src/training/reward.py` (config near `:218`; pricing near `:661`; `_zeroed` `:282`)
- Test: `tests/unit/test_reward.py`

**Interfaces:**
- Consumes: `metrics["forced_final_answer"]`.
- Produces: `forced_final_answer_penalty: float = -0.05`, applied; `_zeroed` sets it to `0.0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_reward.py
def test_forced_final_answer_penalty_applied():
    cfg = RewardConfig()  # serving defaults
    assert cfg.forced_final_answer_penalty == -0.05
    pen = cfg.forced_final_answer_penalty * 1.0
    assert pen == -0.05

def test_zeroed_preset_zeroes_forced_penalty():
    cfg = RewardConfig._zeroed()
    assert cfg.forced_final_answer_penalty == 0.0
```

(Match the exact `RewardConfig` constructor / `_zeroed` accessor names already in `test_reward.py`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_reward.py -k forced -v`
Expected: FAIL — field does not exist.

- [ ] **Step 3: Implement**

In `reward.py`, after `answer_when_evidence_insufficient_penalty` (`:218`):

```python
    # Applied once when the loop forced a best-effort answer at a dead-end/cap.
    # A salvage: milder than the voluntary-insufficient penalty, far better than
    # returning nothing. Mutually exclusive with answer_when_evidence_insufficient.
    forced_final_answer_penalty: float = -0.05
```

In the `_zeroed` classmethod (`:282`, alongside the other `..._penalty=0.0` lines):

```python
            forced_final_answer_penalty=0.0,
```

In the reward computation (near where `answer_when_evidence_insufficient_penalty` is applied, `:661`):

```python
        forced_answer_pen = (
            cfg.forced_final_answer_penalty * metrics.get("forced_final_answer", 0.0)
        )
```

and add `forced_answer_pen` to the returned total and to the breakdown dict (mirror the existing `answer_when_evidence_insufficient_penalty` entries at `:743`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_reward.py -k forced -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/training/reward.py tests/unit/test_reward.py
git commit -m "feat: forced_final_answer_penalty in reward (zeroed preset stable)"
```

---

### Task 10: Baseline updates + full-suite verification

**Files:**
- Modify: `tests/unit/test_agent_loop.py` (existing tests whose `rounds_used` / answer outcomes shifted)
- Test: whole suite

**Interfaces:**
- Consumes: all prior tasks.
- Produces: green `pytest`; documented metrics-key contract check.

- [ ] **Step 1: Run the full suite to find shifted baselines**

Run: `pytest tests/unit/test_agent_loop.py tests/unit/test_reward.py tests/unit/test_loop_controller.py -v`
Expected: some existing `test_agent_loop.py` assertions FAIL where `rounds_used`, `final_answer`, or budget-exhaust metrics changed because plateau-stop / forced-answer are now default-on.

- [ ] **Step 2: Update each shifted assertion to the new expected value**

For each failure, confirm the new value is correct per the spec (fewer rounds on plateau; non-None answer at dead-ends; `search_budget_exhausted_without_answer` now `0.0` where evidence existed). Update the literal expected values. Do NOT weaken assertions to `>=`/`any` — set the exact new expected number.

- [ ] **Step 3: Verify the metrics-key contract is intact**

Run: `python -c "from src.agents.search import SearchAgentLoop; from src.training import reward; print('ok')"` and grep that every key read in `reward.py` / `action_eval.py` still exists in `_initial_metrics`:

Run: `grep -oE "metrics.get\\(\"[a-z_]+\"" src/training/reward.py src/training/eval/action_eval.py | sort -u`
Expected: every listed key is present in `_initial_metrics` (Task 1 added keys; none removed).

- [ ] **Step 4: Run the full unit suite**

Run: `pytest tests/unit -q`
Expected: PASS (all). Then `ruff check . --fix && ruff format .`.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_agent_loop.py
git commit -m "test: update loop baselines for default-on LoopController behaviors"
```

---

## Self-Review

**Spec coverage:** plateau early-stop (Tasks 3,6) · graceful dead-end (Task 7) · smarter gating (Tasks 4,8) · adaptive budget (Tasks 2,5) · additive metrics (Task 1) · reward `forced_final_answer_penalty` + `_zeroed` stability (Task 9) · mutual exclusivity (Task 7 guard) · TDD controller-first (Tasks 2-4 before wiring) · baseline updates (Task 10). All spec sections map to a task.

**Placeholder scan:** Task 5/7/8 reference "the existing fake-model harness / `make_loop` / `SCRIPT_*`" — these are deliberate pointers to the existing `test_agent_loop.py` fixtures (2116 lines) rather than re-inventing a harness; the engineer mirrors a neighboring test for the response script. All controller code (Tasks 2-4) and reward code (Task 9) is complete and literal.

**Type consistency:** `LoopSnapshot`(7 fields), `StopReason`/`AnswerVerb`, `StopDecision.reason`, `AnswerDecision.verb`/`feedback`, `effective_search_limit(int)->int`, `should_continue_searching`/`final_answer_decision(LoopSnapshot)` are used identically across Tasks 2-8. `forced_final_answer` metric key consistent across Tasks 1,7,8,9.
