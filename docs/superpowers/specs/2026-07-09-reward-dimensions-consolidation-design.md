# Reward Dimensions Consolidation — Design

Date: 2026-07-09
Status: Approved (brainstorming)
Branch/PR: feat/reward-dimensions (split from PR #388 to keep each PR single-purpose)
Related: [Simulated-Judge GRPO Demo](2026-07-09-simulated-grpo-demo-design.md)

## Problem

`SearchRewardConfig` has ~22 tunable weights/penalties and `SearchRewardFunction.reward_components()`
returns a flat 23-key breakdown. The four conceptual reward dimensions the system
actually optimizes — **correctness, citation support, retrieval quality, search
efficiency** — are spread across those ~18 shaping terms with no first-class,
readable rollup. Reading a reward breakdown requires knowing which of 23 keys
belongs to which concern.

Goal: expose a 4-bucket view over the existing terms, **without** removing or
renaming anything. Backward-compatible: all weights, presets, keys, `total`
computation, and existing tests keep working.

## Non-goals

- No change to any weight default, penalty, preset, or the `total` formula.
- No collapse of the config API to 4 knobs (a separate, breaking option we
  explicitly declined).
- No removal/pruning of any existing term.
- No changes to `grpo.py`, the trainers, or the example scripts.

## Approach (additive 4-bucket view, all in `src/training/reward.py`)

1. **Single source-of-truth mapping** — module-level constant:

   ```python
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

   _NON_DIMENSION_KEYS: frozenset[str] = frozenset({
       "reward_mode", "terminal_reward", "shaping_total", "total", "human_feedback",
   })
   ```

   4 + 7 + 7 = 18 shaping terms + `correctness` (terminal). Every dimension member
   is a key already produced by `reward_components()`.

2. **Pure grouping function** — `group_reward_components(components: dict[str, float])
   -> dict[str, float]`. For each dimension, sum its member keys via
   `components.get(key, 0.0)` (tolerates missing optional keys). Returns exactly the
   4 dimension subtotals. No class dependency; independently testable.

3. **Expose in `reward_components()`** — after building the flat dict, add 4 keys:
   `dim_correctness`, `dim_citation_support`, `dim_retrieval_quality`,
   `dim_search_efficiency` (values from `group_reward_components`). The `dim_`
   prefix avoids colliding with existing `correctness` / `citation_support` keys.
   The dict stays `dict[str, float]` — consumers such as
   `grpo.py._select_reward_component` (which reads by exact name) are unaffected.

4. **Convenience method** — `SearchRewardFunction.reward_dimensions(output,
   ground_truth, judge_fn) -> dict[str, float]` returning just the 4 subtotals
   (calls `reward_components` then `group_reward_components`, or reads the `dim_*`
   keys). For callers that want only the rollup.

## Invariant

`sum(4 dims) == terminal_reward + shaping_total == total / reward_scale`
(within float tolerance). The dimensions are the **pre-scale** decomposition,
matching the existing pre-scale `terminal_reward` / `shaping_total` keys.
`human_feedback` is added post-aggregation and is intentionally **not** a
dimension member (orthogonal external signal).

## Compatibility audit (done)

Grepped the test suite for exact key-set / length assertions on
`SearchRewardFunction.reward_components`:
- `test_llm_agent_generation.py:3322` — asserts on a **hand-built literal**
  `reward_components` passed into `assign_group_relative_advantages` (identity
  passthrough), not the real function. Unaffected.
- `test_reward_shapes.py:171` — asserts on `CompositeRewardConfig.compute_breakdown`,
  a different method. Unaffected.

No test pins the exact key-set of `reward_components`, so adding 4 keys is safe.

## Testing

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

## Risks

- **Future term escapes the buckets** — mitigated by test 5 (completeness guard).
- **Double-counting / gap in the map** — mitigated by test 3 (partition invariant)
  and test 5 (each key in exactly one bucket).
