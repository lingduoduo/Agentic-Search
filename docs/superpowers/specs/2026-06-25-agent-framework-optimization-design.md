# Spec: Agent Framework Optimization — Planner, Search, Reranker, Evidence Judge, Answer Generator

> Status: **APPROVED** (brainstorming complete 2026-06-25). One PR delivers spec + plan + code + tests on a feature branch.

## Objective

Optimize the five modular search-loop components and the GRPO reward that prices their actions, hitting
four dimensions at once — **answer quality, cost/latency, GRPO reward shaping, and robustness** —
without a refactor and without changing any public component signature in a breaking way.

The components already exist and are wired through `SearchAgentState`
([src/agents/components/](src/agents/components/), [src/agents/search.py](src/agents/search.py)). This PR makes each one *cheaper, safer, or sharper*
with one targeted change apiece, plus two new reward terms so GRPO can price the new behaviors.

**Why now.** Reading the current code surfaced five concrete weaknesses (see below). Each is a small,
self-contained, deterministically testable fix. Recent work (PR #330) already moved on cost; this
continues that line across every component in one coherent pass.

**Constraint (confirmed with user).** No GPU this session, so the GRPO piece is **code + unit/smoke
tests only** — reward terms and metrics wiring, proven green by deterministic tests. Actual training
convergence is proven later on a GPU.

## Scope decisions (confirmed with user)

- **Dimensions:** all four — answer quality, cost/latency, GRPO reward, robustness/code quality.
- **Approach:** focused, high-impact — **one optimization per component** + two reward terms, every
  change unit-testable here. (The "comprehensive" alternative with training-internal changes was
  declined because those can't be verified without a real run.)
- **Backward compatibility:** new reward terms default to weight `0.0` (presets byte-stable); new
  behaviors that could change loop output are conservative (off-by-default thresholds).
- **This PR delivers:** this spec, an implementation plan + task breakdown under
  [docs/superpowers/plans/](docs/superpowers/plans/), and the code + tests.

## Current weaknesses (the optimization surface)

1. **Planner** ([planner.py:72](src/agents/components/planner.py#L72)) — on unparseable output it searches the *entire raw
   generation* as a query (can dump a whole reasoning trace into the retriever); it has no awareness of
   prior queries, so it re-issues a duplicate search that costs a full round.
2. **Search Tool** ([search_tool.py](src/agents/components/search_tool.py)) — no result cache (same query+backend re-hits the
   network); a web-backend *exception* (not just "unconfigured") propagates and crashes the loop.
3. **Reranker Tool** ([reranker_tool.py:30](src/agents/components/reranker_tool.py#L30)) — reranks the *entire* accumulated doc set
   every call (cross-encoder cost grows each round) and doesn't skip the trivial ≤1-doc case.
4. **Evidence Judge** ([evidence_judge.py](src/agents/components/evidence_judge.py)) — produces a score but exposes no marginal-gain /
   plateau signal, so the loop can't stop early when more searching won't help.
5. **Answer Generator** ([answer_generator.py:29](src/agents/components/answer_generator.py#L29)) — emits citations in retrieval order
   (not order-of-appearance in the answer) and doesn't collapse duplicate doc contents.

## The optimizations (one per component + reward)

| Component | Optimization | Dimensions | Behavior change |
|---|---|---|---|
| **Planner** | Duplicate-query guard: `decide(text, previous_queries=())` flags a repeat search via a `is_duplicate` field on `SearchAction`; bounded fallback query (first line / capped length) instead of dumping raw text | Cost, Robustness | New optional arg; old call sites unaffected |
| **Search Tool** | Per-instance result cache keyed by `(retriever, normalized_query)`; wrap web call in try/except → degrade to vdb on *exception* as well as on unconfigured | Cost/latency, Robustness | Cache returns same docs; degradation is logged |
| **Reranker Tool** | `max_candidates` window (rerank only the top-N by current order) + skip when `len(docs) <= 1` | Cost/latency | Default `max_candidates=None` (off) keeps current behavior |
| **Evidence Judge** | `marginal_gain(prev, curr)` + `should_stop(prev, curr, min_gain)` plateau signal; verdict gains nothing new (helpers are pure/static) | Quality, Cost | Additive API only |
| **Answer Generator** | Order citations by first appearance in the answer text; collapse duplicate doc contents to one citation | Quality | Output list re-ordered/deduped |
| **Reward (GRPO)** | One new zero-default term: `early_stop_bonus` × `early_stops`; surfaced in `retriever_aware()` preset | GRPO reward | Presets byte-stable at weight 0 |
| **Loop wiring** | `SearchAgentLoop` gains an opt-in `evidence_plateau_min_gain` config: when set, it consults `EvidenceJudge.should_stop` at the round-scoring site and *emits* an `early_stops` metric counting plateau rounds. Default `None` → metric never fires, behavior byte-identical | observability | Side-effect-free counter; opt-in only |

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
  reaching a plateau. Actually terminating the search early to bank the saved round is a deliberate
  **follow-up** validated on GPU first; this PR ships the signal + reward hook so behavior stays
  byte-identical by default and there is no untested control-flow gamble. The evidence safety rail is
  untouched.
- **Citation ordering** matches reading order so the rendered answer's `[R1Q1D1]` markers and the
  citation list agree; duplicate doc *contents* (same text retrieved under different keys across rounds)
  collapse to the first-cited key.

## Tech Stack

- Python 3.11+, existing package (`pip install -e .`). No new dependencies.
- Components: `src/agents/components/` (dataclasses + async, current style).
- Reward/training: `src/training/reward.py` (config dataclass + `metrics.get(...)` pattern).
- Tests: `pytest`, `tests/unit/` (mirrors current layout).

## Project Structure (files touched)

```
src/agents/components/planner.py          → previous_queries arg, is_duplicate flag, bounded fallback
src/agents/components/search_tool.py      → result cache + web-exception degradation
src/agents/components/reranker_tool.py    → max_candidates window + ≤1-doc skip
src/agents/components/evidence_judge.py   → marginal_gain() + should_stop() static helpers
src/agents/components/answer_generator.py → appearance-order + dedup citations
src/agents/search.py                      → opt-in evidence_plateau_min_gain early-stop + early_stops metric
src/training/reward.py                    → early_stop_bonus term (zero-default)

tests/unit/test_components.py             → new cases per component optimization
tests/unit/test_reward.py                 → new terms in isolation + preset regression
tests/unit/test_agent_loop.py             → dedup-skip, early-stop, web-exception degradation
docs/superpowers/specs/2026-06-25-agent-framework-optimization-design.md   → this spec
docs/superpowers/plans/2026-06-25-agent-framework-optimization-plan.md      → plan
docs/superpowers/plans/2026-06-25-agent-framework-optimization-tasks.md     → task breakdown
```

## Testing Strategy

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

## Boundaries

- **Always:** run `pytest` + `ruff check --fix` + `ruff format` before commit; keep new reward terms
  default-weight 0 so existing presets are byte-stable; keep component signatures backward-compatible
  (new args optional); degrade-don't-crash on backend failure; commit spec **and** plan on the PR branch.
- **Ask first:** changing any existing preset's *default* weights; making `max_candidates` or early-stop
  on-by-default; touching the GRPO trainer internals (advantage/loss); adding a dependency.
- **Never:** commit to `main` (feature branch + PR); weaken/delete existing tests to pass; remove the
  evidence safety rail; introduce global mutable cache state.

## Success Criteria

1. Each of the five components gains exactly its one optimization, unit-tested in isolation.
2. `SearchTool` serves a repeated query from cache (no second backend call) and survives a raising web
   backend by degrading to vdb.
3. Planner flags a duplicate query (`SearchAction.is_duplicate`) when it repeats a prior query and
   bounds the fallback query for unparseable/over-long input.
4. EvidenceJudge exposes `marginal_gain` + `should_stop`; with `evidence_plateau_min_gain` set, the
   loop counts plateau rounds into `early_stops` (default `None` → metric 0, behavior unchanged).
5. AnswerGenerator citations are appearance-ordered and de-duplicated; markers remain valid.
6. `SearchRewardConfig` gains `early_stop_bonus`; with weight 0 every existing reward test is unchanged
   (regression green); `retriever_aware()` surfaces it.
7. Full suite green; GRPO smoke step still passes; no test-count regression.
