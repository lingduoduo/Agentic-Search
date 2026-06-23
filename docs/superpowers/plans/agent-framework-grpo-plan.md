# Implementation Plan: Agent Framework Optimization (Modular Components + GRPO Action Policy)

Companion to [SPEC.md](../../../SPEC.md). Phase 2 of the spec-driven workflow. **Awaiting approval.**

## Component dependency graph

```
AgentState (foundation — everything depends on it)
   │
   ├── EvidenceJudge   (reads retrieved_docs → writes evidence_score)   [wraps SearchResultEvaluator]
   ├── SearchTool      (web | vector_db → appends retrieved_docs)       [wraps SearchClient/search_runner]
   ├── RerankerTool    (reorders retrieved_docs)                        [wraps Reranker]
   ├── AnswerGenerator (retrieved_docs → answer + citations)           [extracts current <answer> synthesis]
   └── Planner         (AgentState → action)  ── depends on all tools' action vocab
            │
   SearchAgentLoop  (orchestrates Planner → tool → EvidenceJudge → repeat → AnswerGenerator)
            │
   SearchRewardConfig (+retriever_cost, +rerank_cost, +evidence_gain)
            │
   SearchAgentGRPOTrainer (reused; consumes extended reward + new action metrics)
```

## Implementation order (sequential unless noted)

### Phase A0 — Training durability (prerequisite for long GRPO runs)
The current stack is single-step, foreground, no-resume (`examples/run_sft_grpo.py:156` calls `step_async()` once;
only `policy.save_pretrained()` is saved — no optimizer/scheduler/step state; no step-level timeout; retrieval
failures silently return empty docs at [src/agents/search.py:444](../../../src/agents/search.py)). Adding a live/cached
web action and running multi-hour GRPO makes durability load-bearing, not optional.

0a. **Checkpointable outer loop**: `for step in range(start_step, max_steps): await trainer.step_async(...)`.
    Every K steps save a **full** checkpoint — `policy`, `optimizer.state_dict()`, scheduler, `step` — and resume by
    loading all four. (Today only the model is saved.)
0b. **Step-level timeout + skip**: wrap `step_async()` in `asyncio.wait_for`; on timeout/exception, log + skip the
    step, keep prior progress. Add bounded retries (1–2, backoff) to the retrieval HTTP calls in
    [src/agents/search.py:442](../../../src/agents/search.py) so transient web blips don't poison rollouts.
0c. **Concurrency cap**: always set `max_concurrent` on rollout fan-out
    ([src/training/ppo/search_agent_grpo_trainer.py:228](../../../src/training/ppo/search_agent_grpo_trainer.py)) —
    uncapped, B×G live web searches saturate/rate-limit the search server.
0d. **Per-step metric persistence** (jsonl minimum; wandb if available) so a multi-hour run is observable.
   - *Verify:* kill a run mid-loop and resume to identical step/optimizer state; a forced search timeout skips one
     step without aborting; metrics jsonl grows one row per step.

> Checkpoint 0: a smoke run survives a mid-run kill and resumes; a hung search server skips a step instead of stalling.

### Phase A — Foundation (no behavior change)
1. **`AgentState`** in [src/agents/state.py](../../../src/agents/state.py): the six canonical fields + helpers
   (`record_search`, `record_rerank`, `set_evidence`, `set_answer`). Add `Retriever` enum, `Citation` dataclass.
   - *Verify:* `tests/unit/test_agent_state.py` — invariants (immutable question, deduped queries, round counting).
2. **Component extraction (behavior-preserving):** move the existing implicit logic into
   `src/agents/components/{evidence_judge,search_tool,reranker_tool,answer_generator}.py`, each operating on
   `AgentState`. EvidenceJudge wraps `SearchResultEvaluator` and maps its verdict/scores to a continuous
   `evidence_score ∈ [0,1]`.
   - *Verify:* `tests/unit/test_components.py` (mocked deps) + existing `test_agent_loop.py` still green.

> Checkpoint 1: full suite green with new actions **disabled** — proves the refactor is a no-op.

### Phase B — New actions (parallelizable: B2 ∥ B3)
3. **Planner** in `src/agents/components/planner.py`: parse policy LM tags → typed `PlannerDecision`.
   Add new vocab: `retriever` attribute on `<search>` (`web`/`vdb`) and a `<rerank/>` tag. Malformed → safe default.
4. **SearchTool web vs vector-DB** [B2]: route to one of two configured URLs by `Retriever`; degrade web→vdb
   when key/server absent (log). Add `--web_search_url` to [examples/run_agentic_search.py](../../../examples/run_agentic_search.py) and loop config.
5. **RerankerTool as action** [B3]: expose `Reranker` to reorder `state.retrieved_docs` in place; updates scores.
6. **Wire into `SearchAgentLoop`**: Planner decision dispatches search(web|vdb) / rerank / answer; each updates
   `AgentState`; EvidenceJudge runs after every search round.
   - *Verify:* loop test exercising all four action paths; degradation path test.

> Checkpoint 2: loop executes every action; behavior with new tags present matches intent.

### Phase C — Reward & training
7. **Extend `SearchRewardConfig`** [src/training/reward.py](../../../src/training/reward.py): add `retriever_cost`
   (per-retriever multiplier on search penalty: web > vdb), `rerank_cost`, `evidence_gain` (reward on
   Δ`evidence_score`). All **default 0** → existing presets byte-stable.
   - *Verify:* `tests/unit/test_reward.py` new-term tests + regression (presets unchanged at weight 0).
8. **Surface new metrics** from the loop (`web_searches`, `vdb_searches`, `rerank_calls`, `evidence_score_final`,
   `evidence_gain_total`) into `output.metrics` for the reward function.
9. **Define a training preset** (e.g. `retriever_aware()`) wiring non-zero new weights; do **not** change existing preset defaults (Ask-first boundary).
   - *Verify:* GRPO `--smoke` step completes; reward breakdown includes new terms.

> Checkpoint 3: `--smoke` GRPO run green; ready for a real training run + eval.

### Phase D — Eval (acceptance)
10. Extend eval ([src/training/eval/bamboogle.py](../../../src/training/eval/bamboogle.py)) to log mean
    `search_rounds`, web/vdb mix, rerank rate, correctness. Compare trained vs. heuristic baseline.
    - *Verify:* trained policy ≤ baseline mean rounds at ≥ baseline correctness (Objective success metric).

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Refactor silently changes loop behavior | Checkpoint 1 gate: full suite must pass with new actions disabled before any new behavior. |
| Web backend cost/latency/rate-limits inside rollouts | **Resolved: cached/offline web corpus for training, live web only at serving** (Open Q#2). Training rollouts are deterministic/retry-free; Phase A0 covers durability if live web is later enabled. |
| New reward terms destabilize training | All default to 0; introduce via a *new* preset; tune one term at a time; KL-to-reference already guards policy drift. |
| Reranker latency in rollouts | Make rerank action optional + priced; cap candidate count; reuse existing batch path. |
| `evidence_score` mapping is arbitrary | Calibrate the heuristic→[0,1] map on eval data; keep heuristic gate as safety rail regardless. |

## Parallelization
- Phase A0 (durability) is independent of A/B and can proceed in parallel with the refactor, but **must land before Phase C** (real training runs).
- Phase A is strictly sequential (foundation).
- Within Phase B, **B2 (SearchTool) and B3 (RerankerTool) are independent** and can be built in parallel; both merge before step 6.
- Phases C and D are sequential after B.

## Out of scope (this spec)
- Learned stop-classifier head (reward-driven stopping only).
- Learned fusion weights / adaptive MMR λ / learned query-router (separate future specs).
- New model architecture or network heads.
