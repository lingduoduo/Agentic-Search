# Reward Dimensions Consolidation Implementation Plan

> **For agentic workers:** Use superpowers:test-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add an additive 4-bucket view (correctness / citation_support / retrieval_quality / search_efficiency) over the existing 23 `reward_components` terms, changing no weights, presets, or the `total` formula.

**Architecture:** All in `src/training/reward.py`: a `REWARD_DIMENSIONS` mapping constant + `_NON_DIMENSION_KEYS`, a pure `group_reward_components()` function, 4 new `dim_*` keys appended in `reward_components()`, and a `SearchRewardFunction.reward_dimensions()` convenience method. Tests in `tests/unit/test_reward_shapes.py`.

**Tech Stack:** Python 3, existing `src/training/reward.py`.

## Global Constraints

- Add to branch `feat/simulated-grpo-demo` (PR #388) — never commit to `main`.
- Purely additive: do NOT change any existing weight default, penalty, preset, `reward_mode` handling, or the `total` computation. The original 23 keys keep their names and values.
- The `reward_components` return type stays `dict[str, float]` (flat `dim_*` keys, no nesting).
- Dimensions are pre-scale: `sum(4 dims) == terminal_reward + shaping_total == total / reward_scale`.
- `human_feedback` is NOT a dimension member.
- Match repo ruff formatting (pre-commit runs ruff).

---

## File Structure

- Modify: `src/training/reward.py` — add `REWARD_DIMENSIONS`, `_NON_DIMENSION_KEYS`, `group_reward_components()`, 4 `dim_*` keys in `reward_components()`, and `reward_dimensions()`.
- Modify: `tests/unit/test_reward_shapes.py` — add the dimension tests.

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
        "rerank_cost": -0.02,
    }
    dims = group_reward_components(flat)
    assert set(dims) == {
        "correctness", "citation_support", "retrieval_quality", "search_efficiency",
    }
    assert dims["correctness"] == pytest.approx(1.0)
    assert dims["citation_support"] == pytest.approx(0.3 - 0.1 + 0.1 + 0.05)
    assert dims["retrieval_quality"] == pytest.approx(
        0.15 + 0.2 + 0.1 + 0.0 - 0.2 - 0.05 - 0.2
    )
    assert dims["search_efficiency"] == pytest.approx(
        -0.02 - 0.05 - 0.1 - 0.1 - 0.1 - 0.05 - 0.02
    )


def test_group_reward_components_tolerates_missing_keys():
    # Only correctness present; every other member defaults to 0.0.
    dims = group_reward_components({"correctness": 0.7})
    assert dims["correctness"] == pytest.approx(0.7)
    assert dims["citation_support"] == pytest.approx(0.0)
    assert dims["retrieval_quality"] == pytest.approx(0.0)
    assert dims["search_efficiency"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_reward_shapes.py -k group_reward_components -v`
Expected: FAIL with `ImportError: cannot import name 'REWARD_DIMENSIONS'` / `group_reward_components`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/training/reward.py` (module level, after the `_ANSWER_TAG_RE` regex constant near the top):

```python
# Four conceptual reward dimensions grouping the fine-grained reward_components
# terms. Each member is a key produced by SearchRewardFunction.reward_components.
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

# reward_components keys that are metadata or rollups, not dimension members.
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
        final_answer="Paris [R1Q1D1]",
    )
    # Non-unity scale exercises the total == scale * (terminal + shaping) path.
    fn = SearchRewardFunction(
        SearchRewardConfig.second_pass()  # shaped mode, several terms active
    )
    comps = fn.reward_components(output, "Paris", lambda p, g: 1.0)

    for key in ("dim_correctness", "dim_citation_support",
                "dim_retrieval_quality", "dim_search_efficiency"):
        assert key in comps

    dim_sum = (
        comps["dim_correctness"]
        + comps["dim_citation_support"]
        + comps["dim_retrieval_quality"]
        + comps["dim_search_efficiency"]
    )
    assert dim_sum == pytest.approx(
        comps["terminal_reward"] + comps["shaping_total"]
    )

    # Completeness: every numeric key is in exactly one dimension or excluded.
    members = {k for ks in REWARD_DIMENSIONS.values() for k in ks}
    dim_keys = {
        "dim_correctness", "dim_citation_support",
        "dim_retrieval_quality", "dim_search_efficiency",
    }
    for key, value in comps.items():
        if key in _NON_DIMENSION_KEYS or key in dim_keys:
            continue
        assert key in members, f"reward_components key {key!r} has no dimension"


def test_reward_dimensions_matches_dim_keys():
    from src.training.reward import SearchRewardConfig, SearchRewardFunction
    from src.agents.core.base import AgentLoopOutput

    output = AgentLoopOutput(
        prompt_ids=[1],
        response_ids=[2],
        response_mask=[1],
        num_turns=1,
        metrics={"rounds_used": 0.0, "search_rounds": 0.0, "answer_allowed": 1.0},
        final_answer="Paris",
    )
    fn = SearchRewardFunction(SearchRewardConfig.second_pass())
    comps = fn.reward_components(output, "Paris", lambda p, g: 1.0)
    dims = fn.reward_dimensions(output, "Paris", lambda p, g: 1.0)
    assert dims == {
        "correctness": comps["dim_correctness"],
        "citation_support": comps["dim_citation_support"],
        "retrieval_quality": comps["dim_retrieval_quality"],
        "search_efficiency": comps["dim_search_efficiency"],
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_reward_shapes.py -k "dimension_keys or reward_dimensions" -v`
Expected: FAIL — `dim_*` keys absent / `reward_dimensions` attribute missing.

- [ ] **Step 3: Write minimal implementation**

In `src/training/reward.py`, inside `_reward_components_from_correctness`, just before `return components` (after the `components` dict is fully built and the optional `human_feedback` key is set), add:

```python
        components.update(
            {f"dim_{name}": value for name, value in group_reward_components(components).items()}
        )
```

Then add a method to `SearchRewardFunction` (near `reward_components`):

```python
    def reward_dimensions(
        self,
        output: AgentLoopOutput,
        ground_truth: str,
        judge_fn: Callable[[str, str], float],
    ) -> dict[str, float]:
        """Return the four grouped reward dimensions for one rollout.

        A convenience rollup over :meth:`reward_components`:
        ``correctness``, ``citation_support``, ``retrieval_quality``,
        ``search_efficiency`` (pre-scale; they sum to
        ``terminal_reward + shaping_total``).
        """
        components = self.reward_components(output, ground_truth, judge_fn)
        return group_reward_components(components)
```

Note: `group_reward_components(components)` after the `dim_*` keys are added still
returns correct sums because the `dim_*` keys are not dimension members — but to
keep `reward_dimensions` independent of ordering, it re-groups from the full
components dict (the `dim_*` keys are ignored by the mapping).

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_reward_shapes.py -k "dimension_keys or reward_dimensions" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full reward + grpo test suites (no regressions)**

Run: `python3 -m pytest tests/unit/test_reward_shapes.py tests/unit/test_grpo.py -q`
Expected: PASS — all pre-existing tests still green (the 23 original keys unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/training/reward.py tests/unit/test_reward_shapes.py
git commit -m "feat(reward): expose dim_* keys + reward_dimensions() rollup

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

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
- `_retrieval_components(self, metrics)` → `search_quality`, `subquestion_coverage`, `evidence_gain`, `early_stop_bonus`, `answer_when_evidence_insufficient_penalty`, `forced_final_answer_penalty`, `search_budget_exhausted_without_answer_penalty`
- `_efficiency_components(self, metrics)` → `per_search_penalty`, `unnecessary_search_penalty`, `duplicate_query_penalty`, `budget_penalty`, `unnecessary_fetch_penalty`, `retriever_cost`, `rerank_cost`

Each expression is copied verbatim from the original inline computation so values are byte-identical.

- [ ] **Step 3: Slim `_reward_components_from_correctness`**

Rewrite the body to: merge the four helper sub-dicts into `components`, compute
`dims = group_reward_components(components)`, set `terminal_reward = dims["correctness"]`
and `shaping_total = dims["citation_support"] + dims["retrieval_quality"] +
dims["search_efficiency"]`, compute `total` via `_aggregate_total_reward`, add
`human_feedback` if present, then append `reward_mode`, `terminal_reward`,
`shaping_total`, `total`, and the `dim_*` keys. Delete the old 18-term
`shaping_total` sum and the explicit `components = {...}` literal.

- [ ] **Step 4: Verify behavior is unchanged**

Run the same command as Step 1.
Expected: PASS — identical results (every per-term value, `total`, and the
partition invariant unchanged). A single differing value means the refactor
altered arithmetic; fix the offending helper expression.

- [ ] **Step 5: Commit**

```bash
git add src/training/reward.py
git commit -m "refactor(reward): split reward_components into 4 dimension helpers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:**
- `REWARD_DIMENSIONS` + `_NON_DIMENSION_KEYS` → Task 1. ✓
- `group_reward_components` pure fn (missing-key tolerant) → Task 1 (+ tests 1,2). ✓
- 4 `dim_*` keys in `reward_components` → Task 2 Step 3. ✓
- `reward_dimensions()` method → Task 2 Step 3. ✓
- Partition invariant (`sum dims == terminal + shaping`) → Task 2 test. ✓
- Completeness guard (every key bucketed or excluded) → Task 2 test. ✓
- Backward-compat (original 23 keys unchanged) → Task 2 Step 5 (full suite). ✓

**2. Placeholder scan:** none. Task 2 test notes to reuse an existing output factory "if present" — the inline `AgentLoopOutput` is provided as the concrete fallback, so no placeholder.

**3. Type consistency:** `group_reward_components(dict[str,float]) -> dict[str,float]` used identically in Task 1 def, Task 2 `reward_components` update, and `reward_dimensions`. `dim_*` key names identical across impl and both tests. `REWARD_DIMENSIONS` dimension names (`correctness`/`citation_support`/`retrieval_quality`/`search_efficiency`) match the `dim_*` suffixes and the spec mapping exactly.
