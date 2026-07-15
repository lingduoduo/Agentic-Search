# Generated Context Pack

# Reward Dimensions Consolidation

## Sources

- [Specification: 2026-07-09-reward-dimensions-consolidation-design.md](../specs/2026-07-09-reward-dimensions-consolidation-design.md)
- [Plan: 2026-07-09-reward-dimensions-consolidation.md](../plans/2026-07-09-reward-dimensions-consolidation.md)

## Specification Context

### Overview

Date: 2026-07-09
Status: Approved (brainstorming)
Branch/PR: feat/simulated-grpo-demo (PR #388, added as related GRPO follow-on)
Related: Simulated-Judge GRPO Demo

## Implementation Plan Context

### Task 1: Mapping constant + pure `group_reward_components`

**Files:**
- Modify: `src/training/reward.py` (add near the top, after the regex constants / before `normalize_answer_text`, or just below the module docstring imports)
- Test: `tests/unit/test_reward_shapes.py`

**Interfaces:**
- Produces:
  - `REWARD_DIMENSIONS: dict[str, tuple[str, ...]]` — dimension name → member component keys.
  - `_NON_DIMENSION_KEYS: frozenset[str]` — keys that are metadata/rollups, not dimension members.
  - `group_reward_components(components: dict[str, float]) -> dict[str, float]` — returns the 4 subtotals; sums `components.get(key, 0.0)` per dimension.

- [ ] **Step 1: Write the failing test**

…

### Task 2: Expose `dim_*` keys + `reward_dimensions()` method + invariant/completeness tests

**Files:**
- Modify: `src/training/reward.py` — append 4 keys in `reward_components`; add `reward_dimensions` method.
- Test: `tests/unit/test_reward_shapes.py`

**Interfaces:**
- Consumes: `REWARD_DIMENSIONS`, `_NON_DIMENSION_KEYS`, `group_reward_components` (Task 1).
- Produces:
  - `reward_components(...)` now includes `dim_correctness`, `dim_citation_support`, `dim_retrieval_quality`, `dim_search_efficiency`.
  - `SearchRewardFunction.reward_dimensions(output, ground_truth, judge_fn) -> dict[str, float]` — the 4 subtotals.

- [ ] **Step 1: Write the failing test**

…

### Task 3: Behavior-preserving refactor into 4 dimension helpers

**Files:**
- Modify: `src/training/reward.py` — split `_reward_components_from_correctness`.
- Test: existing `tests/unit/test_reward_shapes.py` + consumer suites (no new tests; the value assertions + partition invariant guard the refactor).

**Interfaces:**
- Consumes: `REWARD_DIMENSIONS`, `group_reward_components` (Task 1); the existing
  private helpers `_citation_support`, `_search_quality`,
  `_unsupported_claim_penalty`, `_fetch_usefulness_reward`, `_aggregate_total_reward`.
- Produces: `_correctness_component`, `_citation_components`,
  `_retrieval_components`, `_efficiency_components` methods; a slimmed
  `_reward_components_from_correctness`. No change to public API or any value.

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
