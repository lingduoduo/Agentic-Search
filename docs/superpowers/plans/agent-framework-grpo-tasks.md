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

> **Checkpoint 1 (gate):** ✅ reached. Foundation + four components landed additively; 162 tests green
> across agents/training/new modules, no regression. T-A.4 (loop wiring) **deferred into Phase B** by
> decision — the loop is untouched, so behavior is provably unchanged.

---

## Phase B — New actions

- [x] **T-B.1: Planner tag vocabulary → typed decision** ✅ done
  - `Planner.decide(text) -> SearchAction|RerankAction|AnswerAction`, parsing
    `<search retriever="web|vdb">`, `<rerank/>`, `<answer>`. Precedence search>rerank>answer; malformed →
    safe vector-DB search. 7 tests incl. malformed/unknown-retriever cases.
  - Files: `src/agents/components/planner.py`, `tests/unit/test_components.py`.

- [x] **T-B.2: SearchTool web vs vector-DB routing + degradation** ✅ done
  - `SearchTool(vector_db_fn, web_fn=None).run(state, query, retriever)` selects backend; WEB degrades to
    vdb (logged) when web unconfigured. Backward compatible with the Phase A signature. 3 tests.
  - Files: `src/agents/components/search_tool.py`, `tests/unit/test_components.py`.

- [x] **T-B.4 (routing slice): per-round web/vdb wiring in `SearchAgentLoop`** ✅ done
  - `web_search_url` config + second `SearchClient`; action regex tolerates attributes;
    `_parse_round_retriever` reads the choice; threaded through execute/cache/retrieve; cache keyed by
    (retriever, query); both clients closed; system prompt teaches the attribute. 2 loop tests
    (web routing + degradation). This is the user-confirmed **per-round** grain + the deferred T-A.4 seam.
  - Files: `src/agents/search.py`, `tests/unit/test_agent_loop.py`, `tests/unit/test_on_turn_callback.py`.

- [x] **T-B.3: Reranker as a policy action** ✅ done (in Phase C, priced by `rerank_cost`)
  - Landed as a **per-search flag** `<search rerank="true">` rather than a standalone `<rerank/>`: the
    round's results are reranked **before they are labeled**, so the positional `[RxQyDz]` citations the
    model sees always match the reranked order — sidestepping the citation-shift problem cleanly. The
    reranker is injected (`loop._reranker`, callable `(query, docs)->docs`); None → logged no-op. Counts
    `rerank_calls` (consumed by `rerank_cost`). 2 loop tests (rerank reorders + no-op without reranker).
  - Files: `src/agents/search.py`, `tests/unit/test_agent_loop.py`.

> **Checkpoint 2 (gate):** ✅ reached — loop executes `search(web)`, `search(vdb)`, `rerank`, `answer`
> + degradation; no regression. (Rerank landed in Phase C alongside its `rerank_cost` pricing.)

---

## Phase C — Reward & training (PR; stacked on Phase B)

- [x] **T-C.1: Extend `SearchRewardConfig` (new terms, default 0)** ✅ done
  - Added `retriever_cost_vdb`, `retriever_cost_web` (web priced 5× vdb in the preset), `rerank_cost`
    (flat per call), `evidence_gain_weight` (reward on Δ`evidence_score`/round). **All default 0**; also
    added to `_zeroed()`. Components `retriever_cost`/`rerank_cost`/`evidence_gain` added to the breakdown.
  - Verify: `pytest tests/unit/test_reward.py` — 5 new-term tests **and** regression: existing presets
    (`sparse_final_only`/`second_pass`/`third_pass_with_format`) leave the new terms at 0. **66 pass.**
  - Files: `src/training/reward.py`, `tests/unit/test_reward.py`.

- [x] **T-C.2: Surface action metrics for the reward** ✅ done
  - Loop emits `web_searches`, `vdb_searches`, `evidence_score_final`, `evidence_gain_total` (cumulative
    positive Δ) into `output.metrics`; per-round continuous score reuses `EvidenceJudge.score_round`
    (finally wiring the component into the loop). `rerank_calls` initialized to 0 (populated once the
    rerank action lands). Reward reads all via `metrics.get(...)`.
  - Verify: `pytest tests/unit/test_agent_loop.py -k action_metrics`.
  - Files: `src/agents/search.py`, `src/agents/components/evidence_judge.py` (public `score_round`), test.

- [x] **T-C.3: `retriever_aware()` reward preset** ✅ done (GRPO `--smoke` = manual)
  - `SearchRewardConfig.retriever_aware()` builds on `second_pass` and sets web=5×vdb cost, flat rerank
    cost, Δevidence-gain. Existing preset defaults untouched.
  - Verify: `pytest tests/unit/test_reward.py -k retriever_aware`. The end-to-end GRPO `--smoke` run needs
    a real model + torch and is a **manual/integration step** (not a unit test).
  - Files: `src/training/reward.py`, test.

> **Checkpoint 3 (gate):** ✅ reward terms + metrics + preset + **rerank action (priced)** landed;
> 237-test slice green, presets byte-stable.

> **T-A0.2 retrieval retries — already covered (no code needed):** `SearchClient` already retries with
> exponential backoff (`max_retries=3`, 4xx excluded, [src/context/retrieval/client.py:83](../../../src/context/retrieval/client.py));
> the loop's `_retrieve_many` try/except is the final graceful degrade after retries are exhausted.

---

## Phase D — Eval (acceptance) — requires a real trained checkpoint

- [x] **T-D.1 (scaffold): action-eval aggregation + comparison** ✅ done
  - `src/training/eval/action_eval.py`: `aggregate_action_metrics(samples)` → mean correctness /
    search_rounds / web / vdb / web_fraction / rerank_rate / evidence; `compare_action_evals(baseline,
    trained)` encodes the headline success criterion (**fewer rounds AND correctness preserved**);
    `format_comparison_table` renders it. 6 unit tests. Operates on the metrics already on
    `output.metrics`, so it's ready to consume real eval rollouts.
- [ ] **T-D.1 (run): baseline-vs-trained numbers** — ⏳ needs a converged GRPO checkpoint
  - Produce the checkpoint via `train_loop` + `SearchRewardConfig.retriever_aware()`, then feed
    baseline (heuristic) and trained rollouts through `action_eval`. **Manual/integration step.**
  - Files: `src/training/eval/action_eval.py` (done), `src/training/eval/bamboogle.py` (wire-up, manual).

---

## PR slicing (revised after T-A.4 deferral)
- **PR 1 (this one):** Phase A — `SearchAgentState` foundation + four extracted components, additive +
  fully unit-tested. Loop untouched.
- **PR 2 (this one):** Phase B — Planner tags + web/vdb routing (component + live loop) **+ the deferred
  T-A.4 loop seam**. Rerank-as-action deferred to PR 4. Behind Checkpoint 2.
- **PR 3:** Phase A0 (durability) — can land any time before Phase C.
- **PR 4:** Phases C + D — reward terms (`retriever_cost`, `rerank_cost`, `evidence_gain`) + **rerank
  loop action (priced)** + eval.

Each PR carries a copy of [SPEC.md](../../../SPEC.md) + this plan/tasks on its branch (repo convention).
