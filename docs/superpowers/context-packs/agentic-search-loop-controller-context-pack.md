# Generated Context Pack

# Agentic Search Loop Controller

## Sources

- [Specification: 2026-06-25-agentic-search-loop-controller-design.md](../archive/specs/2026-06-25-agentic-search-loop-controller-design.md)
- [Plan: 2026-06-25-agentic-search-loop-controller.md](../archive/plans/2026-06-25-agentic-search-loop-controller.md)

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

## Implementation Plan Context

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

### Task 8: Route the answer-gate through `final_answer_decision`

**Files:**
- Modify: `src/agents/search.py` (`run()` answer-gate block `:1052-1094`)
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Consumes: `final_answer_decision`.
- Produces: gate decisions (ACCEPT/REJECT/FORCE) come from the controller; behavior matches today's reject-then-force, now via FORCE.

- [ ] **Step 1: Write the failing test**

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_agent_loop.py::test_answer_gate_forces_after_max_rejections -v`
Expected: FAIL — `forced_final_answer` is `0.0` (old path force-passes without flagging).

- [ ] **Step 3: Implement**

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
