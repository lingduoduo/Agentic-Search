# Generated Context Pack

# Reward Dimensions Consolidation

## Sources

- [Specification: 2026-07-09-reward-dimensions-consolidation-design.md](../specs/2026-07-09-reward-dimensions-consolidation-design.md)
- [Plan: 2026-07-09-reward-dimensions-consolidation.md](../plans/2026-07-09-reward-dimensions-consolidation.md)

## Specification Context

### Non-goals

- No change to any weight default, penalty, preset, or the `total` formula.
- No collapse of the config API to 4 knobs (a separate, breaking option we
  explicitly declined).
- No removal/pruning of any existing term.
- No changes to `grpo.py`, the trainers, or the example scripts.

### Testing

New tests in `tests/unit/test_reward_shapes.py` (fast, no model):

1. `group_reward_components` sums each bucket correctly from a hand-built flat dict
   with distinct per-key values.
2. Tolerates a components dict missing optional keys (no `human_feedback`) — no
   `KeyError`.
3. Partition invariant on a real `reward_components(...)` output: `sum(4 dims) ==
   terminal_reward + shaping_total`, and `== total / reward_scale` with a
   non-unity `reward_scale`, within tolerance.
4. `reward_components()` output contains the 4 `dim_*` keys; `reward_dimensions()`
   returns exactly those 4 values.
5. Partition-completeness guard: every numeric key in a `reward_components` output

…

### Risks

- **Future term escapes the buckets** — mitigated by test 5 (completeness guard).
- **Double-counting / gap in the map** — mitigated by test 3 (partition invariant)
  and test 5 (each key in exactly one bucket).

## Implementation Plan Context

### Global Constraints

- Add to branch `feat/simulated-grpo-demo` (PR #388) — never commit to `main`.
- Purely additive: do NOT change any existing weight default, penalty, preset, `reward_mode` handling, or the `total` computation. The original 23 keys keep their names and values.
- The `reward_components` return type stays `dict[str, float]` (flat `dim_*` keys, no nesting).
- Dimensions are pre-scale: `sum(4 dims) == terminal_reward + shaping_total == total / reward_scale`.
- `human_feedback` is NOT a dimension member.
- Match repo ruff formatting (pre-commit runs ruff).

---

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
