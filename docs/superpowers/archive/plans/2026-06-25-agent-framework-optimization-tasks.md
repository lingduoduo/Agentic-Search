# Agent Framework Optimization — Task Breakdown

Companion to [the plan](2026-06-25-agent-framework-optimization-plan.md) and
[the spec](../specs/2026-06-25-agent-framework-optimization-design.md). Each task is independently
testable and ends with a commit. Order is the recommended execution order; T1–T5 are independent of
each other, T7 depends on T4, T6 is independent, T8 gates the PR.

| ID | Component | Deliverable | Verify | Acceptance |
|----|-----------|-------------|--------|------------|
| **T1** | Planner | `SearchAction.is_duplicate` flag + `decide(text, previous_queries=())` + bounded fallback query | `pytest tests/unit/test_components.py -k planner` | Duplicate query (whitespace/case-insensitive) flagged; new query not flagged; fallback ≤256 chars, first non-empty line; existing planner tests green |
| **T2** | Search Tool | Per-instance `(backend, normalized-query)` result cache + degrade-to-vdb on web exception | `pytest tests/unit/test_components.py -k search_tool` | Repeated query served from cache (1 backend call); per-backend keys distinct; raising web fn degrades to vdb; existing tests green |
| **T3** | Reranker Tool | `max_candidates: int \| None = None` window + ≤1-doc skip | `pytest tests/unit/test_components.py -k reranker` | ≤1 doc → reranker not called; `max_candidates=N` scores only top-N, tail preserved; `None` reranks full set (existing tests green) |
| **T4** | Evidence Judge | Static `marginal_gain(prev, curr)` + `should_stop(prev, curr, min_gain)` | `pytest tests/unit/test_components.py -k "marginal_gain or should_stop"` | `marginal_gain` = delta; `should_stop` true iff gain `< min_gain`; pure, no state |
| **T5** | Answer Generator | Citations ordered by first appearance in answer + collapse duplicate doc contents | `pytest tests/unit/test_components.py -k answer_generator` | Out-of-order markers → appearance order; identical contents → one citation (first marker); single-round tests unchanged; markers valid |
| **T6** | Reward | `early_stop_bonus: float = 0.0` term, wired into `compute`, `_zeroed`, `retriever_aware()` | `pytest tests/unit/test_reward.py` | `early_stop_bonus * early_stops` in breakdown; weight 0 → all preset/regression totals byte-identical |
| **T7** | Loop | Opt-in `evidence_plateau_min_gain` config → `early_stops` metric (uses T4 `should_stop`) | `pytest tests/unit/test_agent_loop.py` | Plateau round increments `early_stops` when configured; default `None` → metric 0, behavior byte-identical; existing loop tests green |
| **T8** | Gate + PR | Full suite + GRPO smoke + lint, push, open PR | `pytest tests/unit -q && pytest tests/unit/test_search_agent_grpo_trainer.py && ruff check . && ruff format .` | All green, no test-count regression; PR opened with spec+plan+tasks linked |

## Global acceptance (spec Success Criteria)

1. Each component gains exactly one optimization, unit-tested in isolation — T1–T5.
2. SearchTool serves a repeat from cache and survives a raising web backend — T2.
3. Planner flags duplicates + bounds fallback — T1.
4. EvidenceJudge `marginal_gain`/`should_stop`; opt-in loop `early_stops` — T4, T7.
5. AnswerGenerator citations appearance-ordered + de-duplicated — T5.
6. `early_stop_bonus` zero-default; presets byte-stable; `retriever_aware()` surfaces it — T6.
7. Full suite + GRPO smoke green; no test-count regression — T8.

## Out of scope (deliberate)

- Rewiring the production `SearchAgentLoop` to consume the Planner/SearchTool/RerankerTool/AnswerGenerator
  objects (a refactor — the loop uses only `EvidenceJudge.score_round` today).
- A second `duplicate_search_penalty` reward term — would double-count the existing
  `duplicate_query_penalty` × `repeated_search_queries`.
- Acting on the evidence plateau to terminate the loop early — deferred to a GPU-validated follow-up;
  this PR ships the detection metric + reward hook only.
- Any real GRPO training run (no GPU this session).
