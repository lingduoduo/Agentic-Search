# Generated Context Pack

# Agent Framework Optimization — Task Breakdown

## Sources

- [Plan: 2026-06-25-agent-framework-optimization-tasks.md](../archive/plans/2026-06-25-agent-framework-optimization-tasks.md)

## Implementation Plan Context

### Global acceptance (spec Success Criteria)

1. Each component gains exactly one optimization, unit-tested in isolation — T1–T5.
2. SearchTool serves a repeat from cache and survives a raising web backend — T2.
3. Planner flags duplicates + bounds fallback — T1.
4. EvidenceJudge `marginal_gain`/`should_stop`; opt-in loop `early_stops` — T4, T7.
5. AnswerGenerator citations appearance-ordered + de-duplicated — T5.
6. `early_stop_bonus` zero-default; presets byte-stable; `retriever_aware()` surfaces it — T6.
7. Full suite + GRPO smoke green; no test-count regression — T8.

### Out of scope (deliberate)

- Rewiring the production `SearchAgentLoop` to consume the Planner/SearchTool/RerankerTool/AnswerGenerator
  objects (a refactor — the loop uses only `EvidenceJudge.score_round` today).
- A second `duplicate_search_penalty` reward term — would double-count the existing
  `duplicate_query_penalty` × `repeated_search_queries`.
- Acting on the evidence plateau to terminate the loop early — deferred to a GPU-validated follow-up;
  this PR ships the detection metric + reward hook only.
- Any real GRPO training run (no GPU this session).

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
