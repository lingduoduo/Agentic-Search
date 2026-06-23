# Tasks: Agent Framework Optimization (Modular Components + GRPO Action Policy)

Phase 3 of the spec-driven workflow. Derived from [SPEC.md](../../../SPEC.md) and
[the plan](agent-framework-grpo-plan.md). Defaults locked 2026-06-23 (web=5× vdb, flat rerank cost,
Δevidence_score reward, reward-driven stopping).

**Conventions:** each task is one focused session, ≤ ~5 files, has acceptance + a verify command.
Tasks are dependency-ordered. `[P]` = parallelizable with its sibling. Checkpoints are hard gates.

---

## Phase A0 — Training durability (parallel with A/B; must precede C)

- [ ] **T-A0.1: Checkpointable GRPO outer loop**
  - Acceptance: a `train_loop(trainer, prompts, max_steps, ckpt_every, ckpt_dir, start_step=0)` runs N
    `step_async()` calls; every `ckpt_every` it saves `policy`, `optimizer.state_dict()`, scheduler, and
    `step` to `ckpt_dir`. A `--resume <ckpt_dir>` path restores all four and continues from `step`.
  - Verify: `pytest tests/unit/test_train_loop.py -v` — run 4 steps, kill, resume, assert identical step
    counter + optimizer state tensor-equal.
  - Files: `src/training/ppo/train_loop.py` (new), `examples/run_sft_grpo.py` (use the loop), `tests/unit/test_train_loop.py`.

- [ ] **T-A0.2: Step-level timeout + skip + retrieval retries** `[P]`
  - Acceptance: `step_async()` wrapped in `asyncio.wait_for`; on timeout/exception the step is logged and
    skipped (loop continues, prior progress intact). Retrieval HTTP calls in `_retrieve_many`/`_fetch_pages`
    get 1–2 bounded retries with backoff before degrading to empty.
  - Verify: `pytest tests/unit/test_train_loop.py -k "timeout or retry"` — injected hang skips one step;
    injected transient failure succeeds on retry.
  - Files: `src/training/ppo/train_loop.py`, `src/agents/search.py` (retry wrap at ~442–453), test.

- [ ] **T-A0.3: Concurrency cap + per-step metric persistence** `[P]`
  - Acceptance: rollout fan-out always honors `max_concurrent` (default set, not None); each step appends
    one JSONL row (`step`, `loss`, `mean_reward`, `mean_kl`, action mix) to `ckpt_dir/metrics.jsonl`.
  - Verify: `pytest tests/unit/test_train_loop.py -k "concurrency or metrics"` — semaphore bounds in-flight
    rollouts; metrics file grows one row per step.
  - Files: `src/training/ppo/search_agent_grpo_trainer.py` (~228), `src/training/ppo/train_loop.py`, test.

> **Checkpoint 0 (gate):** smoke run survives a mid-run kill→resume; a hung search server skips a step
> instead of stalling. `pytest tests/unit/test_train_loop.py` green.

---

## Phase A — Foundation (sequential; no behavior change)

- [x] **T-A.1: `SearchAgentState` + value types** ✅ done
  - Acceptance: `SearchAgentState` dataclass with the six canonical fields + helpers (`record_search`,
    `record_rerank`, `set_evidence`, `set_citations`); `Retriever` enum (`WEB`, `VECTOR_DB`); `Citation`
    dataclass. `question` preserved across ops; `previous_queries` deduped & ordered; `search_rounds`
    increments once per retriever call (not on rerank); `set_evidence` clamps to [0,1].
  - Deviations from spec draft: named `SearchAgentState` (a `AgentState` already exists for
    orchestration); reuses existing `RetrievedDocument` (not `ContextDocument`); `set_answer`→`set_citations`
    (final answer text is a loop output, not one of the six fields).
  - Verify: `pytest tests/unit/test_agent_state.py -v` — **9 passed**; lint clean; 62 state-consumer tests green.
  - Files: `src/agents/state.py`, `tests/unit/test_agent_state.py`.

- [x] **T-A.2: Extract EvidenceJudge + AnswerGenerator (behavior-preserving)** `[P]` ✅ done
  - `EvidenceJudge` wraps `SearchResultEvaluator` → continuous `evidence_score ∈ [0,1]`
    (blends query sufficiency with squashed top scores; monotonic in quality);
    `AnswerGenerator` resolves `[RxQyDz]` markers to structured `Citation`s via `AgentContext`.
  - Verify: `pytest tests/unit/test_components.py` — green (9 of the 13 belong here).
  - Files: `src/agents/components/{evidence_judge,answer_generator}.py`, `tests/unit/test_components.py`.

- [x] **T-A.3: Extract SearchTool + RerankerTool (DI, behavior-preserving)** `[P]` ✅ done
  - `SearchTool(retrieve_fn).run(state, query)` records the round; `RerankerTool(rerank_fn).run(state)`
    reorders `retrieved_docs` (no round counted). Dependencies injected for isolation; concrete backends
    (web/vdb URLs, cross-encoder) wired in Phase B where the new actions actually need them.
  - Verify: `pytest tests/unit/test_components.py` — green (4 of the 13 belong here).
  - Files: `src/agents/components/{search_tool,reranker_tool}.py`, `tests/unit/test_components.py`.

- [ ] **T-A.4: Wire components into `SearchAgentLoop`** — ⚠️ **DECISION NEEDED (recommend deferring into Phase B)**
  - Finding after reading the loop: its retrieval path is **batch/multi-query with per-run caching and
    source dedup** (`_retrieve_with_cache`→`_retrieve_many`→`SearchClient.retrieve`, [src/agents/search.py:441](../../../src/agents/search.py)),
    and evidence/citation/answer logic is already integrated via `AgentContext` + `SearchResultEvaluator`.
    A pure-no-op re-wire onto the components would be high-churn/high-risk (must keep 62 loop tests
    byte-green) for **zero behavioral payoff** in Phase A.
  - **Recommendation:** defer wiring into **Phase B**, at the exact points the new actions need it
    (SearchTool must grow web/vdb routing + batch there anyway; rerank action introduces the seam). This
    is the surgical path — wire once, when it changes behavior, instead of a no-op rewrite now + rewrite in B.
  - Alternative if wanted now: a parallel `SearchAgentState` mirror threaded through the loop (populated
    from existing data) — moderate effort, low risk, gives Phase B a ready seam.

> **Checkpoint 1 (gate):** full suite green with new actions **disabled** — proves the refactor changed
> no behavior. `pytest` green, no test-count regression.

---

## Phase B — New actions (B.2 ∥ B.3)

- [ ] **T-B.1: Planner tag vocabulary → typed decision**
  - Acceptance: `Planner.decide(state) -> PlannerDecision` parsing `<search retriever="web|vdb">…</search>`,
    `<rerank/>`, `<answer>…`. Malformed/unknown tags → safe default (single vdb search). Round-trips with
    the loop's existing tag parser.
  - Verify: `pytest tests/unit/test_components.py -k planner -v` incl. malformed-tag cases.
  - Files: `src/agents/components/planner.py`, test.

- [ ] **T-B.2: SearchTool web vs vector-DB routing + degradation** `[P]`
  - Acceptance: `SearchTool` routes to one of two configured URLs by `Retriever`; missing web key/server
    → degrade to VDB + log (no crash). `--web_search_url` flag added to the example + loop config; training
    config points web URL at the cached corpus.
  - Verify: `pytest tests/unit/test_components.py -k "retriever or degrade" -v`.
  - Files: `src/agents/components/search_tool.py`, `src/agents/search.py` (config), `examples/run_agentic_search.py`, test.

- [ ] **T-B.3: RerankerTool as a policy action** `[P]`
  - Acceptance: loop dispatches `<rerank/>` to `RerankerTool`; updates doc order + scores in `AgentState`;
    increments a `rerank_calls` metric. No new docs fetched.
  - Verify: `pytest tests/unit/test_components.py -k rerank -v`.
  - Files: `src/agents/components/reranker_tool.py`, `src/agents/search.py`, test.

- [ ] **T-B.4: Dispatch all four actions in the loop**
  - Acceptance: a single loop test drives a trajectory hitting `search(web)`, `search(vdb)`, `rerank`,
    `answer`; each updates `AgentState` correctly; degradation path covered.
  - Verify: `pytest tests/unit/test_agent_loop.py -k "actions or retriever or rerank" -v`.
  - Files: `src/agents/search.py`, `tests/unit/test_agent_loop.py`.

> **Checkpoint 2 (gate):** loop executes every action path incl. degradation. `pytest` green.

---

## Phase C — Reward & training (needs A0 + B; requires Checkpoint 0 & 2)

- [ ] **T-C.1: Extend `SearchRewardConfig` (new terms, default 0)**
  - Acceptance: add `retriever_cost` (per-retriever multiplier; web=5× vdb when enabled), `rerank_cost`
    (flat per call), `evidence_gain` (reward on Δ`evidence_score`/round). **All default 0** → existing
    presets byte-stable.
  - Verify: `pytest tests/unit/test_reward.py -v` — new-term unit tests **and** regression: existing
    presets (`sparse_final_only`, `second_pass`, `third_pass_with_format`) produce identical totals.
  - Files: `src/training/reward.py`, `tests/unit/test_reward.py`.

- [ ] **T-C.2: Surface action metrics for the reward**
  - Acceptance: loop emits `web_searches`, `vdb_searches`, `rerank_calls`, `evidence_score_final`,
    `evidence_gain_total` into `output.metrics`; reward reads them.
  - Verify: `pytest tests/unit/test_reward.py -k "metrics or gain" -v`.
  - Files: `src/agents/search.py`, `src/training/reward.py`, test.

- [ ] **T-C.3: `retriever_aware()` reward preset + GRPO smoke**
  - Acceptance: a new preset wires non-zero new weights (web 5× vdb, flat rerank cost, Δevidence_gain);
    **existing preset defaults untouched**. `--smoke` GRPO step completes one step with the new action
    space, no NaN, reward breakdown shows the new terms.
  - Verify: `python3 -m src.training.ppo.search_agent_grpo_trainer --smoke` exits 0; `pytest tests/unit/test_reward.py -k retriever_aware`.
  - Files: `src/training/reward.py`, `src/training/ppo/search_agent_grpo_trainer.py`, test.

> **Checkpoint 3 (gate):** `--smoke` GRPO run green using the new action space + durable loop.

---

## Phase D — Eval (acceptance)

- [ ] **T-D.1: Eval logging + baseline vs trained comparison**
  - Acceptance: eval logs mean `search_rounds`, web/vdb mix, rerank rate, correctness; produces a
    baseline (heuristic) vs trained (policy) table.
  - Verify: `python3 -m src.training.eval.bamboogle --compare`; trained mean `search_rounds` ≤ baseline at
    ≥ baseline correctness (Objective success metric).
  - Files: `src/training/eval/bamboogle.py`.

---

## Suggested PR slicing
- **PR 1:** Phase A0 (durability) — independently valuable, mergeable alone.
- **PR 2:** Phases A + B (refactor + new actions) — behind Checkpoint 1 & 2.
- **PR 3:** Phases C + D (reward + training + eval).

Each PR carries a copy of [SPEC.md](../../../SPEC.md) + this plan/tasks on its branch (repo convention).
