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
   is a member of exactly one dimension in `REWARD_DIMENSIONS` or in
   `_NON_DIMENSION_KEYS` — fails loudly if a future shaping term is added without a
   bucket assignment.

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

Add to `tests/unit/test_reward_shapes.py` (top-level, near other imports add: `from src.training.reward import REWARD_DIMENSIONS, group_reward_components`):

```python
def test_group_reward_components_sums_each_bucket():
    flat = {
        "correctness": 1.0,
        "citation_support": 0.3,
        "unsupported_claim_penalty": -0.1,
        "fetch_usefulness_reward": 0.1,
        "format_reward": 0.05,
        "search_quality": 0.15,
        "subquestion_coverage": 0.2,
        "evidence_gain": 0.1,
        "early_stop_bonus": 0.0,
        "answer_when_evidence_insufficient_penalty": -0.2,
        "forced_final_answer_penalty": -0.05,
        "search_budget_exhausted_without_answer_penalty": -0.2,
        "per_search_penalty": -0.02,
        "unnecessary_search_penalty": -0.05,
        "duplicate_query_penalty": -0.1,
        "budget_penalty": -0.1,
        "unnecessary_fetch_penalty": -0.1,
        "retriever_cost": -0.05,

_[Section compacted.]_

### terms. Each member is a key produced by SearchRewardFunction.reward_components.

REWARD_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "correctness": ("correctness",),
    "citation_support": (
        "citation_support",
        "unsupported_claim_penalty",
        "fetch_usefulness_reward",
        "format_reward",
    ),
    "retrieval_quality": (
        "search_quality",
        "subquestion_coverage",
        "evidence_gain",
        "early_stop_bonus",
        "answer_when_evidence_insufficient_penalty",
        "forced_final_answer_penalty",
        "search_budget_exhausted_without_answer_penalty",
    ),
    "search_efficiency": (
        "per_search_penalty",
        "unnecessary_search_penalty",
        "duplicate_query_penalty",
        "budget_penalty",
        "unnecessary_fetch_penalty",
        "retriever_cost",
        "rerank_cost",
    ),
}

### reward_components keys that are metadata or rollups, not dimension members.

_NON_DIMENSION_KEYS: frozenset[str] = frozenset(
    {"reward_mode", "terminal_reward", "shaping_total", "total", "human_feedback"}
)


def group_reward_components(components: dict[str, float]) -> dict[str, float]:
    """Roll the flat reward_components breakdown up into the 4 reward dimensions.

    Sums each dimension's member terms (missing keys count as 0.0), returning a
    dict with exactly the four keys in :data:`REWARD_DIMENSIONS`. The result is
    the pre-scale decomposition: ``sum(result.values())`` equals
    ``terminal_reward + shaping_total`` for a full components dict.
    """
    return {
        dimension: sum(float(components.get(key, 0.0)) for key in members)
        for dimension, members in REWARD_DIMENSIONS.items()
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_reward_shapes.py -k group_reward_components -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/training/reward.py tests/unit/test_reward_shapes.py
git commit -m "feat(reward): REWARD_DIMENSIONS map + group_reward_components

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

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

Add to `tests/unit/test_reward_shapes.py`. Reuse the existing helpers in that file for building an `AgentLoopOutput` if present; otherwise build a minimal output inline as shown. (Check the file for an existing factory like `_make_output`/`_output` first and use it.)

```python
def test_reward_components_includes_dimension_keys_and_partition_invariant():
    from src.training.reward import (
        SearchRewardConfig,
        SearchRewardFunction,
        _NON_DIMENSION_KEYS,
        REWARD_DIMENSIONS,
    )
    from src.agents.core.base import AgentLoopOutput

    output = AgentLoopOutput(
        prompt_ids=[1],
        response_ids=[2, 3],
        response_mask=[1, 1],
        num_turns=2,
        metrics={
            "rounds_used": 2.0,
            "search_rounds": 2.0,
            "repeated_search_queries": 1.0,
            "subquestion_coverage_ratio": 1.0,
            "final_evidence_sufficient": 1.0,
            "search_quality_score": 1.0,
            "answer_allowed": 1.0,
        },

_[Section compacted.]_

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

- [ ] **Step 1: Baseline the current behavior**

Run: `python3 -m pytest tests/unit/test_reward_shapes.py tests/unit/test_reward.py tests/unit/test_reward_human_signal.py tests/unit/test_grpo.py tests/unit/test_bamboogle_eval.py tests/unit/test_search_agent_grpo_trainer.py tests/unit/test_simulated_judge.py tests/unit/test_readme_examples.py -q`
Expected: PASS (all green before the refactor — this is the regression baseline).

- [ ] **Step 2: Extract the four dimension helpers**

Add four methods to `SearchRewardFunction`, each returning `{component_key: weighted_value}`
for its dimension, moving the exact expressions out of the monolith:
- `_correctness_component(self, correctness) -> {"correctness": cfg.correctness_weight * correctness}`
- `_citation_components(self, answer, ctx, metrics)` → `citation_support`, `unsupported_claim_penalty`, `fetch_usefulness_reward`, `format_reward`

_[Section compacted.]_

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
