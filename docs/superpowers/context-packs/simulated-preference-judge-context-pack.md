# Generated Context Pack

# Simulated Preference Judge

## Sources

- [Specification: 2026-07-09-simulated-preference-judge-design.md](../archive/specs/2026-07-09-simulated-preference-judge-design.md)
- [Plan: 2026-07-09-simulated-preference-judge.md](../archive/plans/2026-07-09-simulated-preference-judge.md)

## Specification Context

### Overview

**Date:** 2026-07-09
**Status:** Approved (design)
**Branch:** `feat/simulated-preference-judge`

## Implementation Plan Context

### Task 1: `SimulatedPreferenceJudge` core + export

**Files:**
- Create: `src/training/judge.py`
- Modify: `src/training/__init__.py` (add exports inside the existing `try:` block, after the `from .reward import ...` lines)
- Test: `tests/unit/test_simulated_judge.py`

**Interfaces:**
- Consumes: `BatchJudgeFn` from `src/training/reward.py` (type alias `Callable[[list[str], list[str]], list[float]]`).
- Produces:
  - `SimulatedPreferenceJudge` dataclass with `max_words: int = 40`, `jitter_scale: float = 0.05`.
  - `SimulatedPreferenceJudge.score(answer: str) -> float`
  - `SimulatedPreferenceJudge.as_batch_judge_fn() -> BatchJudgeFn`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_simulated_judge.py`:

…

### Task 2: `judge_gold_agreement` validation helper

**Files:**
- Modify: `src/training/judge.py` (append module-level function)
- Modify: `src/training/__init__.py` (add export)
- Test: `tests/unit/test_simulated_judge.py` (append tests)

**Interfaces:**
- Produces: `judge_gold_agreement(pairs: list[tuple[float, bool]]) -> dict[str, float]` returning keys `mean_score_correct`, `mean_score_incorrect`, `gap`, `n_correct`, `n_incorrect`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_simulated_judge.py`:

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_simulated_judge.py -k agreement -v`
Expected: FAIL with `ImportError: cannot import name 'judge_gold_agreement'`

…

### Task 3: GRPO integration test (proves the seam)

This task adds no production code — it proves the judge drives real GRPO advantages through the existing `score_prompt_group` seam, and guards against regressions in that wiring.

**Files:**
- Test: `tests/unit/test_simulated_judge.py` (append)

**Interfaces:**
- Consumes: `GRPORolloutSample` from `src/training/grpo.py`, `AgentLoopOutput` from `src/agents/core/base.py`, `score_prompt_group` from `src/training/grpo.py`.
- `AgentLoopOutput` required constructor args: `prompt_ids: list[int]`, `response_ids: list[int]`, `response_mask: list[int]`, `num_turns: int`; `final_answer` is an optional field.

…

### Final verification

- [ ] Run the whole new test file: `pytest tests/unit/test_simulated_judge.py -v` → all PASS.
- [ ] Run the broader training suite for regressions: `pytest tests/unit -k "grpo or judge or reward" -v` → all PASS.
- [ ] Lint: `ruff check src/training/judge.py examples/run_bamboogle_synthetic_grpo.py tests/unit/test_simulated_judge.py --fix && ruff format src/training/judge.py examples/run_bamboogle_synthetic_grpo.py tests/unit/test_simulated_judge.py`

---

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
