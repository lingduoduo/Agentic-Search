# Generated Context Pack

# Agentic Search Loop Controller

## Sources

- [Specification: 2026-06-25-agentic-search-loop-controller-design.md](../specs/2026-06-25-agentic-search-loop-controller-design.md)
- [Plan: 2026-06-25-agentic-search-loop-controller.md](../plans/2026-06-25-agentic-search-loop-controller.md)

## Specification Context

### Goal

Improve the control flow of agentic search by making the loop's two implicit
decisions explicit, and shipping four behavior improvements on top of them:

1. **Plateau early-stop** — stop searching when marginal evidence gain stalls.
2. **Graceful dead-end answer** — emit a best-effort answer at a dead-end instead
   of returning nothing.
3. **Smarter answer-gating** — targeted per-subquestion feedback, then a clean
   forced-answer path.
4. **Adaptive search budget** — scale the search-round budget by subquestion count.

### Architecture (Approach 2: extract a `LoopController`)

**New file:** `src/agents/components/loop_controller.py`. It owns the *policy* for
the two decisions; `run()` remains the *state owner* and consults it. The
controller is pure-over-snapshot (no I/O, no model calls), holding only `cfg`.

```python
@dataclass(frozen=True)
class LoopSnapshot:
    rounds_used: int
    num_subquestions: int          # len(active_tasks)
    evidence_sufficient: bool      # _has_sufficient_evidence(...)
    prev_evidence_score: float     # metrics["evidence_score_final"] before round
    curr_evidence_score: float     # after round
    consecutive_rejections: int
    model_emitted_answer: bool

class StopReason(Enum):
    CONTINUE = "continue"
    BUDGET_EXHAUSTED = "budget_exhausted"
    PLATEAU = "plateau"

class AnswerVerb(Enum):
    ACCEPT = "accept"   # take the model's <answer>
    REJECT = "reject"   # send feedback, let it search more
    FORCE  = "force"    # one final answer turn, accept unconditionally

@dataclass(frozen=True)
class StopDecision:   reason: StopReason
@dataclass(frozen=True)
class AnswerDecision: verb: AnswerVerb; feedback: str = ""

class LoopController:
    def __init__(self, cfg: SearchAgentLoopConfig): ...
    def effective_search_limit(self, num_subquestions: int) -> int: ...
    def should_continue_searching(self, s: LoopSnapshot) -> StopDecision: ...
    def final_answer_decision(self, s: LoopSnapshot) -> AnswerDecision: ...
```

**Why stateless:** `run()` is already the single state owner; the Phase-2 state
machine wants one state home. A stateful controller would create a second one.

_[Section compacted.]_

### Testing strategy

**1. `LoopController` unit tests (new — `tests/unit/test_loop_controller.py`).**
Pure decisions, zero model/tokenizer setup. Written **first (TDD)**.
- `should_continue_searching`: CONTINUE while evidence climbing · STOP(PLATEAU)
  only when `gain<0.05 AND sufficient` · no plateau-stop when insufficient ·
  STOP(BUDGET_EXHAUSTED) at `effective_limit`.
- `effective_search_limit`: 1 subq → base · n subq → clamp formula · cap respected.
- `final_answer_decision`: ACCEPT when sufficient · REJECT(+feedback) while
  `rejections < cap` · FORCE at cap · FORCE on dead-end-with-evidence.

**2. Loop integration tests (extend `test_agent_loop.py`).** Existing fake
model/server harness.
- Plateau path: stalled-evidence script → loop stops early, `rounds_used` drops,
  `plateau_early_stop=1`.
- Forced-answer path: budget/no-action dead-end with evidence → exactly one extra
  turn, `final_answer` non-None, `forced_final_answer=1`,
  `search_budget_exhausted_without_answer=0`.
- Bounded fallback: forced turn emits no `<answer>` → tentative/templated, no loop.
- No-evidence opt-out: immediate format errors → keep today's empty exit.
- Baseline updates: existing tests whose `rounds_used`/answer expectations shift
  are updated in the same commit (enumerated in the implementation plan).

**3. Reward tests (extend `test_reward.py`).** New `forced_final_answer_penalty`
pricing; `_zeroed` preset stays byte-stable; mutual-exclusivity (forced answer
priced `-0.05`, not `-0.2`).

**Verification gate:** full `pytest` green before any commit; diff the metrics dict

_[Section compacted.]_

### Non-goals

- Full state-machine refactor / `run()` decomposition (Phase 2).
- Unifying the loop onto `Planner`/`SearchTool` components (Phase 2).
- Difficulty-classifier-based budgeting (subquestion count only).
- Fabricating answers when no evidence was collected.
- **Human-in-the-loop / approval gating.** The `LoopController` adds no
  interventional HITL. Today's HITL is observational only (the `on_turn`
  progress callback streamed over SSE); `Plan.requires_human_approval` is a
  defined-but-unenforced flag. A blocking approval gate is serving-only and
  training-incompatible, so it belongs in its own later spec as an injected
  async `ApprovalFn` callable (same model/env/agent decoupling), not here.

## Implementation Plan Context

### Global Constraints

- **Add metrics, never mutate existing key semantics.** New keys only; existing keys in `_initial_metrics` (`src/agents/search.py:344`) keep their meaning.
- **`forced_final_answer` is mutually exclusive with `answer_when_evidence_insufficient`** — a forced salvage sets the former and must NOT set the latter.
- **Reward `_zeroed` preset stays byte-stable** — any new penalty defaults to a non-zero serving value but is set to `0.0` in the `_zeroed` preset.
- **Plateau early-stop is conservative:** `evidence_plateau_min_gain = 0.05`, gated on `plateau_requires_sufficient = True` (only stops when evidence already sufficient).
- **Adaptive budget keyed on subquestion count only**; single-subquestion questions get today's limit unchanged.
- **Full `pytest` green before every commit after the first.**
- **The loop preserves append-only history** — the forced-answer turn appends messages; it never branches or rewrites history.

---

### Task 1: Config knobs + metric keys

**Files:**
- Modify: `src/agents/search.py` (`SearchAgentLoopConfig` ends ~`:235`; `_initial_metrics` `:344-378`)
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Produces: new `SearchAgentLoopConfig` fields `evidence_plateau_min_gain: float = 0.05`, `plateau_requires_sufficient: bool = True`, `search_budget_per_subquestion: int = 1`, `max_search_limit_cap: int = 10`, `force_answer_on_deadend: bool = True`. New metric keys `forced_final_answer`, `plateau_early_stop`, `effective_search_limit`, `adaptive_budget_bonus` (all `0.0`).

- [ ] **Step 1: Write the failing test**

```python

### tests/unit/test_agent_loop.py

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

_[Section compacted.]_

### Task 2: LoopController types + `effective_search_limit`

**Files:**
- Create: `src/agents/components/loop_controller.py`
- Test: `tests/unit/test_loop_controller.py`

**Interfaces:**
- Consumes: `SearchAgentLoopConfig` (Task 1 fields).
- Produces: `LoopSnapshot` (frozen dataclass, 7 fields), `StopReason`/`AnswerVerb` enums, `StopDecision`/`AnswerDecision`, `LoopController(cfg)` with `effective_search_limit(num_subquestions: int) -> int`.

- [ ] **Step 1: Write the failing test**

```python

### tests/unit/test_loop_controller.py

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

### src/agents/components/loop_controller.py

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

_[Section compacted.]_

### Task 3: `should_continue_searching`

**Files:**
- Modify: `src/agents/components/loop_controller.py`
- Test: `tests/unit/test_loop_controller.py`

**Interfaces:**
- Produces: `LoopController.should_continue_searching(s: LoopSnapshot) -> StopDecision`.

- [ ] **Step 1: Write the failing test**

```python

### tests/unit/test_loop_controller.py

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

_[Section compacted.]_

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
