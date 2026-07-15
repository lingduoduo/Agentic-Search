# Generated Context Pack

# Spec: Agent Framework Optimization — Planner, Search, Reranker, Evidence Judge, Answer Generator

## Sources

- [Specification: 2026-06-25-agent-framework-optimization-design.md](../specs/2026-06-25-agent-framework-optimization-design.md)

## Specification Context

### Scope decisions (confirmed with user)

- **Dimensions:** all four — answer quality, cost/latency, GRPO reward, robustness/code quality.
- **Approach:** focused, high-impact — **one optimization per component** + two reward terms, every
  change unit-testable here. (The "comprehensive" alternative with training-internal changes was
  declined because those can't be verified without a real run.)
- **Backward compatibility:** new reward terms default to weight `0.0` (presets byte-stable); new
  behaviors that could change loop output are conservative (off-by-default thresholds).
- **This PR delivers:** this spec, an implementation plan + task breakdown under
  docs/superpowers/plans/, and the code + tests.

### The optimizations (one per component + reward)

| Component | Optimization | Dimensions | Behavior change |
|---|---|---|---|
| **Planner** | Duplicate-query guard: `decide(text, previous_queries=())` flags a repeat search via a `is_duplicate` field on `SearchAction`; bounded fallback query (first line / capped length) instead of dumping raw text | Cost, Robustness | New optional arg; old call sites unaffected |
| **Search Tool** | Per-instance result cache keyed by `(retriever, normalized_query)`; wrap web call in try/except → degrade to vdb on *exception* as well as on unconfigured | Cost/latency, Robustness | Cache returns same docs; degradation is logged |
| **Reranker Tool** | `max_candidates` window (rerank only the top-N by current order) + skip when `len(docs) <= 1` | Cost/latency | Default `max_candidates=None` (off) keeps current behavior |
| **Evidence Judge** | `marginal_gain(prev, curr)` + `should_stop(prev, curr, min_gain)` plateau signal; verdict gains nothing new (helpers are pure/static) | Quality, Cost | Additive API only |
| **Answer Generator** | Order citations by first appearance in the answer text; collapse duplicate doc contents to one citation | Quality | Output list re-ordered/deduped |
| **Reward (GRPO)** | One new zero-default term: `early_stop_bonus` × `early_stops`; surfaced in `retriever_aware()` preset | GRPO reward | Presets byte-stable at weight 0 |

_[Section compacted.]_

### Scope correction (discovered during planning)

Two facts changed the original two-reward-term plan:

1. **The production `SearchAgentLoop` does not consume the `Planner`/`SearchTool`/`RerankerTool`/
   `AnswerGenerator` objects** — only `EvidenceJudge.score_round` (a static helper). Those four are the
   **standalone component API** (exercised by `tests/unit/test_components.py` and available to
   component-based consumers such as the GRPO rollout path). Optimizing them improves the documented
   component contract and its tests; it does **not** silently change production loop behavior. This PR
   does **not** rewire the loop to adopt the components (that would be a refactor — out of scope).
2. **Duplicate-search penalization already exists** in the reward as `duplicate_query_penalty` ×
   `repeated_search_queries`. A second `duplicate_search_penalty` term would double-count, so it is
   **dropped**. The Planner/SearchTool dedup optimizations still pay off through the *existing*
   penalty plus fewer wasted rounds (lower `per_search_penalty` / `retriever_cost`). Only the genuinely
   new signal — **`early_stop_bonus`** — is added.

### Design notes / decisions

- **Planner stays pure.** It does not own state. The loop passes `state.previous_queries` into
  `decide(...)`; the planner only *flags* a duplicate (`SearchAction.is_duplicate=True`). The **loop**
  decides whether to skip — so dedup policy lives in one place and the planner stays testable in
  isolation. Fallback query is bounded to the first non-empty line, capped at 256 chars.
- **Search cache is per-loop-run, not global.** The cache lives on the `SearchTool` instance the loop
  creates per question, so there is no cross-question leakage and no eviction policy to design (YAGNI).
  Normalization for the key = `strip()` + collapse internal whitespace + `casefold()`. A cache hit does
  **not** re-hit the backend; whether the round is recorded is the loop's call (it skips duplicates
  before they reach the tool, so the cache is a second safety net for non-identical-but-equivalent
  queries).
- **Reranker window is opt-in.** `max_candidates=None` (default) preserves today's behavior exactly;
  set to an int to bound cost. Reranking ≤1 doc is a guaranteed no-op, so we skip it and avoid a
  needless cross-encoder call.
- **Early-stop is opt-in, observed before it is acted on.** The judge exposes the *signal*
  (`should_stop`); the loop, when `evidence_plateau_min_gain` is configured (default `None` = disabled),
  *counts* plateau rounds into the `early_stops` metric at the round-scoring site — a side-effect-free
  counter that does not alter control flow. `early_stop_bonus` (default 0) lets GRPO learn to value

_[Section compacted.]_

### Testing Strategy

- **Per-component units:**
  - Planner: duplicate flagged when query in `previous_queries`; not flagged otherwise; fallback query
    bounded for multi-line/over-length raw text; existing parse tests unchanged.
  - SearchTool: second identical query served from cache without a second backend call (spy/counter);
    web `RetrieveFn` that raises degrades to vdb (result returned, warning logged).
  - RerankerTool: `max_candidates=N` reranks only N; ≤1 doc returns without calling `rerank_fn`;
    default `None` preserves full-set rerank.
  - EvidenceJudge: `marginal_gain` arithmetic; `should_stop` true iff gain `< min_gain`; bounds.
  - AnswerGenerator: citations ordered by appearance; duplicate contents collapsed; markers still valid.
- **Reward:** `early_stop_bonus` in isolation; **regression** that `sparse_final_only`, `second_pass`,
  `third_pass_with_format`, and `retriever_aware` totals are unchanged when the new weight is 0.
- **Loop regression:** `test_agent_loop.py` — with `evidence_plateau_min_gain` set, a plateau round
  increments `early_stops`; with it `None` (default), `early_stops` stays 0 and behavior is unchanged.
  Existing tests pass.
- **Smoke:** the existing GRPO smoke test
  (`test_search_agent_grpo_trainer.py::test_grpo_smoke_step_with_retriever_aware_reward`) stays green.
- **No test-count regression**; new modules' new branches covered.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
