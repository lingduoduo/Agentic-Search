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
| **Reward (GRPO)** | Two new zero-default terms: `duplicate_search_penalty` × `duplicate_searches`, `early_stop_bonus` × `early_stops`; surfaced in `retriever_aware()` preset | GRPO reward | Presets byte-stable at weight 0 |
| **Loop wiring** | `SearchAgentLoop` passes `previous_queries` to the planner and skips a flagged duplicate; consults the judge's `should_stop`; emits `duplicate_searches` and `early_stops` metrics | integration | New conservative behavior, metrics additive |

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
- **Early-stop is reward-shaped, not hard-wired.** The judge exposes the *signal*; the loop emits an
  `early_stops` metric when it stops on a plateau instead of spending another round. `early_stop_bonus`
  (default 0) lets GRPO learn to value it. The existing evidence safety rail is untouched — early-stop
  only *prevents extra searches*, it never forces a premature answer below the evidence floor.
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
src/agents/search.py                      → pass previous_queries, skip dup, early-stop, emit metrics
src/training/reward.py                    → duplicate_search_penalty + early_stop_bonus (zero-default)

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
- **Reward:** each new term in isolation; **regression** that `sparse_final_only`, `second_pass`,
  `third_pass_with_format`, and `retriever_aware` totals are unchanged when new weights are 0.
- **Loop regression:** `test_agent_loop.py` — a duplicate planned query consumes no extra round; a
  plateau triggers `early_stops`; a raising web backend degrades instead of crashing. Existing tests pass.
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
3. Planner flags a duplicate query and the loop skips it (no extra `search_rounds`), emitting a
   `duplicate_searches` metric.
4. EvidenceJudge exposes `marginal_gain` + `should_stop`; the loop emits `early_stops` on a plateau.
5. AnswerGenerator citations are appearance-ordered and de-duplicated; markers remain valid.
6. `SearchRewardConfig` gains `duplicate_search_penalty` + `early_stop_bonus`; with weights 0 every
   existing reward test is unchanged (regression green); `retriever_aware()` surfaces them.
7. Full suite green; GRPO smoke step still passes; no test-count regression.
