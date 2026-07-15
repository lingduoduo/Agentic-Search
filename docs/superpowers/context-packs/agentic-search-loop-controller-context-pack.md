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

**Why stateless:** `run()` is already the single state owner; the Phase-2 state
machine wants one state home. A stateful controller would create a second one.
Pure decisions also unit-test with zero setup.

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

…

## Implementation Plan Context

### Global Constraints

- **Add metrics, never mutate existing key semantics.** New keys only; existing keys in `_initial_metrics` (`src/agents/search.py:344`) keep their meaning.
- **`forced_final_answer` is mutually exclusive with `answer_when_evidence_insufficient`** — a forced salvage sets the former and must NOT set the latter.
- **Reward `_zeroed` preset stays byte-stable** — any new penalty defaults to a non-zero serving value but is set to `0.0` in the `_zeroed` preset.
- **Plateau early-stop is conservative:** `evidence_plateau_min_gain = 0.05`, gated on `plateau_requires_sufficient = True` (only stops when evidence already sufficient).

…

### Task 1: Config knobs + metric keys

**Files:**
- Modify: `src/agents/search.py` (`SearchAgentLoopConfig` ends ~`:235`; `_initial_metrics` `:344-378`)
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Produces: new `SearchAgentLoopConfig` fields `evidence_plateau_min_gain: float = 0.05`, `plateau_requires_sufficient: bool = True`, `search_budget_per_subquestion: int = 1`, `max_search_limit_cap: int = 10`, `force_answer_on_deadend: bool = True`. New metric keys `forced_final_answer`, `plateau_early_stop`, `effective_search_limit`, `adaptive_budget_bonus` (all `0.0`).

- [ ] **Step 1: Write the failing test**

- [ ] **Step 2: Run test to verify it fails**

…

### Task 2: LoopController types + `effective_search_limit`

**Files:**
- Create: `src/agents/components/loop_controller.py`
- Test: `tests/unit/test_loop_controller.py`

**Interfaces:**
- Consumes: `SearchAgentLoopConfig` (Task 1 fields).
- Produces: `LoopSnapshot` (frozen dataclass, 7 fields), `StopReason`/`AnswerVerb` enums, `StopDecision`/`AnswerDecision`, `LoopController(cfg)` with `effective_search_limit(num_subquestions: int) -> int`.

- [ ] **Step 1: Write the failing test**

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_loop_controller.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement**

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_loop_controller.py -v`

…

### Task 3: `should_continue_searching`

**Files:**
- Modify: `src/agents/components/loop_controller.py`
- Test: `tests/unit/test_loop_controller.py`

**Interfaces:**
- Produces: `LoopController.should_continue_searching(s: LoopSnapshot) -> StopDecision`.

- [ ] **Step 1: Write the failing test**

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_loop_controller.py -k should_continue or continue or budget or plateau -v`
Expected: FAIL — `AttributeError: should_continue_searching`.

- [ ] **Step 3: Implement**

Append to `LoopController` in `src/agents/components/loop_controller.py`:

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_loop_controller.py -v`
Expected: PASS

…

### Task 4: `final_answer_decision`

**Files:**
- Modify: `src/agents/components/loop_controller.py`
- Test: `tests/unit/test_loop_controller.py`

**Interfaces:**
- Produces: `LoopController.final_answer_decision(s: LoopSnapshot) -> AnswerDecision`.

- [ ] **Step 1: Write the failing test**

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_loop_controller.py -k final_answer or accept or reject -v`
Expected: FAIL — `AttributeError: final_answer_decision`.

- [ ] **Step 3: Implement**

Append to `LoopController`:

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_loop_controller.py -v`
Expected: PASS (all controller tests)

- [ ] **Step 5: Commit**

---

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
