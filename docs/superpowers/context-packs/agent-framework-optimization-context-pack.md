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

…

### Testing Strategy

- **Per-component units:**
  - Planner: duplicate flagged when query in `previous_queries`; not flagged otherwise; fallback query
    bounded for multi-line/over-length raw text; existing parse tests unchanged.
  - SearchTool: second identical query served from cache without a second backend call (spy/counter);
    web `RetrieveFn` that raises degrades to vdb (result returned, warning logged).
  - RerankerTool: `max_candidates=N` reranks only N; ≤1 doc returns without calling `rerank_fn`;
    default `None` preserves full-set rerank.
  - EvidenceJudge: `marginal_gain` arithmetic; `should_stop` true iff gain `< min_gain`; bounds.

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
