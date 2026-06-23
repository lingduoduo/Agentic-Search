# Spec: Agent Framework Optimization — Modular Components + GRPO Action Policy

> Status: **DRAFT — awaiting approval** (Phase 1 of spec-driven workflow: Specify → Plan → Tasks → Implement)

## Objective

Refactor the agentic search loop into five **explicit, independently testable components** wired
through a single `AgentState`, then **expand the GRPO action space** so the policy learns *how much*
to search, *which retriever* to use (web vs. vector-DB), and *when* to rerank — instead of relying
on hard-wired heuristics.

**Why now.** The pieces exist but are entangled:
- "Planning" is implicit in the model's `<think>`/`<subquestions>` tags ([src/agents/search.py](src/agents/search.py)).
- Loop state is spread across `AgentContext`, a free-form `metrics` dict, and per-turn locals — there is no single `AgentState`.
- The reranker is backend-only ([src/internal/retrieval/reranker.py](src/internal/retrieval/reranker.py)); the agent cannot choose to call it.
- Retriever source choice is the **heuristic** M10 router ([src/internal/routing/router.py](src/internal/routing/router.py)); the agent loop hits a single `search_url`, so "web vs vector-DB" is not a learnable action.
- Stopping is a **heuristic** gate (`SearchResultEvaluator`, [src/training/evaluation.py](src/training/evaluation.py)), not shaped by reward.

**Users.** (1) ML engineers training the search agent via GRPO; (2) the serving stack
([src/internal/servers/web/app.py](src/internal/servers/web/app.py)) that runs the same loop at inference.

**Success looks like.** The same `SearchAgentLoop` runs in both train and serve; every action
(search-web, search-vdb, rerank, answer) is selectable by the policy and priced by the reward; and
on a held-out eval the trained policy spends **fewer search rounds for equal-or-better answer
correctness** than the heuristic baseline.

### Scope decisions (confirmed with user)
- **Scope:** Refactor *and* GRPO (both).
- **Learnable actions:** search count/budget, **web vs. vector-DB retriever**, **invoke reranker**. (Stopping is *not* a separate head — see assumptions.)
- **Evidence Judge:** keep the existing heuristic; expose a continuous `evidence_score ∈ [0,1]` into `AgentState`.
- **This session delivers:** this `SPEC.md` + a phased implementation plan ([docs/superpowers/plans/agent-framework-grpo-plan.md](docs/superpowers/plans/agent-framework-grpo-plan.md)). No code until approved.

### Assumptions (correct me before I proceed)
1. **Stopping is reward-driven, not a dedicated action head.** The policy learns to stop by choosing
   `answer` early; "stop immediately" = 0 extra search rounds. The heuristic `SearchResultEvaluator`
   remains only as a safety rail that blocks answers when `evidence_score` is below a floor (preserving
   current `answer_when_evidence_insufficient_penalty` behavior). If you want a separate learned stop
   classifier, that's a bigger change — tell me.
2. **`AgentState` is the single source of truth** threaded through all five components. `AgentContext`
   becomes an internal detail of the Search Tool / Evidence Judge, not the loop's primary state.
3. **Two retriever backends are reachable from the loop**: a vector-DB endpoint (demo/hybrid, the
   current `search_url`) and a web endpoint. Wiring is via **two configured URLs**; if a web key/server
   is absent, the web action degrades to vector-DB and is logged (no crash). **Training** points the web
   URL at a **cached/offline web corpus** (Open Q#2 resolved); **serving** points it at live
   google/serp/browser. The loop code is identical either way — only the configured URL differs.
4. **Reranker is exposed as an agent tool** that operates on the *current* `retrieved_docs` (re-orders
   in place, updates scores). It does not fetch new docs.
5. **GRPO machinery is reused, not rebuilt.** We extend `SearchRewardConfig` with retriever/rerank
   cost terms and an evidence-gain bonus; the trainer ([src/training/ppo/search_agent_grpo_trainer.py](src/training/ppo/search_agent_grpo_trainer.py)) and group sampling ([src/training/grpo.py](src/training/grpo.py)) are unchanged in structure.
6. **No new model architecture.** The policy is the same causal LM emitting action tags; the new
   actions are new tags (`<rerank/>`, retriever attribute on `<search>`), not new network heads.

---

## Tech Stack

- **Language:** Python 3.11+ (existing package, `pip install -e .`).
- **Agent loop:** existing `src/agents/` (async, HF tokenizers, VERL-style `response_mask`).
- **Training:** existing GRPO stack in `src/training/` (PyTorch, `AutoModelForCausalLM`).
- **Retrieval:** existing `src/internal/retrieval/` + retrieval servers (`/retrieve`, `/search`).
- **Reranker:** existing cross-encoder `src/internal/retrieval/reranker.py` (`ms-marco-MiniLM-L12-v2` / Cohere).
- **Tests:** `pytest` (unit + regression). No new frameworks.

---

## Commands

```bash
# Setup (one-time)
pip install -e . && pip install -r requirements.txt

# Retrieval backends (two URLs the loop can target)
python3 -m src.internal.servers.retrieval.hybrid --corpus_path data/corpus.jsonl   # vector-DB (port 8001)
python3 -m src.internal.servers.retrieval.serp                                     # web (needs SERP_API_KEY)

# Tests — overall + the new modules
pytest                                                  # full suite (regression gate)
pytest tests/unit/test_agent_state.py -v                # new: AgentState
pytest tests/unit/test_components.py -v                 # new: Planner/SearchTool/RerankerTool/EvidenceJudge/AnswerGenerator
pytest tests/unit/test_agent_loop.py -v                 # existing loop, must still pass
pytest tests/unit/test_reward.py -v                     # extended reward terms

# Lint / format / types
ruff check . --fix && ruff format .

# Smoke: run the loop locally with retriever selection
python3 -m examples.run_agentic_search --mode search \
  --question "Compare dense and sparse retrieval" \
  --search_url http://localhost:8001/retrieve \
  --web_search_url http://localhost:8002/retrieve   # NEW flag

# Train (GRPO) — small smoke run
python3 -m src.training.ppo.search_agent_grpo_trainer --smoke
```

---

## Project Structure

```
src/agents/
  state.py            → SearchAgentState (the 6 canonical fields) — primary search-loop state
                        (added alongside the existing orchestration AgentState; reuses RetrievedDocument)
  search.py           → SearchAgentLoop — orchestrates components, owns AgentState
  components/         → NEW: explicit, single-responsibility components
    planner.py        →   Planner: AgentState → PlannerDecision (action + params)
    search_tool.py    →   SearchTool: executes query vs chosen retriever (web | vector_db)
    reranker_tool.py  →   RerankerTool: re-orders retrieved_docs via cross-encoder
    evidence_judge.py →   EvidenceJudge: wraps SearchResultEvaluator → evidence_score ∈ [0,1]
    answer_generator.py→  AnswerGenerator: retrieved_docs → answer + citations

src/training/
  reward.py           → SearchRewardConfig extended: retriever_cost, rerank_cost, evidence_gain
  ppo/search_agent_grpo_trainer.py → unchanged structure; uses extended reward

tests/unit/
  test_agent_state.py     → state transitions, immutability of question, dedup of previous_queries
  test_components.py      → each component in isolation (mocked deps)
  test_reward.py          → new reward terms + regression on existing presets
docs/superpowers/
  specs/  → this spec (copy committed on the PR branch per repo convention)
  plans/  → agent-framework-grpo-plan.md
```

---

## Code Style

Match existing dataclass + async style. The canonical state is named **`SearchAgentState`**
to avoid clobbering the pre-existing orchestration `AgentState` in the same module; it reuses
the existing `RetrievedDocument` dataclass (surgical — same module, no new cross-package coupling):

```python
@dataclass(slots=True)
class SearchAgentState:
    """Single source of truth threaded through the search loop for one question."""
    question: str                                   # treated as read-only for the run
    previous_queries: list[str] = field(default_factory=list)   # dedup-ordered
    retrieved_docs: list[RetrievedDocument] = field(default_factory=list)
    evidence_score: float = 0.0                     # [0,1] from EvidenceJudge, latest round
    search_rounds: int = 0                          # retriever calls so far (budget driver)
    citations: list[Citation] = field(default_factory=list)     # set by AnswerGenerator

    def record_search(self, query: str, docs: list[RetrievedDocument]) -> None:
        if query not in self.previous_queries:      # dedup query, but the round still counts
            self.previous_queries.append(query)
        self.retrieved_docs.extend(docs)
        self.search_rounds += 1
    # also: record_rerank(docs), set_evidence(score)->clamp[0,1], set_citations(cites)
```

Components are **pure where possible** — `(AgentState, deps) -> result`, no hidden globals:

```python
class Planner:
    def decide(self, state: AgentState) -> PlannerDecision:
        """Parse the policy LM's action tags into a typed decision.
        Returns one of: SearchAction(query, retriever), RerankAction(), AnswerAction()."""
```

Conventions: snake_case; typed dataclasses for all cross-component payloads; retriever is an enum
`Retriever.WEB | Retriever.VECTOR_DB`; every component has a `*_test.py` peer; degrade-don't-crash on
backend failure (log + fall back), matching the M10 constructors' pattern.

---

## Testing Strategy

- **Framework:** `pytest`, tests in `tests/unit/` (mirrors current layout). Integration tests stay gated.
- **Unit (per component):** Planner tag-parsing (incl. malformed tags), SearchTool retriever routing +
  degradation, RerankerTool reordering, EvidenceJudge score in `[0,1]` and monotonic with result
  quality, AnswerGenerator citation extraction.
- **State:** `AgentState` invariants — `question` never mutated; `previous_queries` deduped & ordered;
  `search_rounds` increments once per retriever call.
- **Reward:** each new term in isolation; **regression** that existing presets (`sparse_final_only`,
  `second_pass`, `third_pass_with_format`) produce unchanged totals when new weights are 0.
- **Loop regression:** existing `test_agent_loop.py` must pass unchanged (refactor preserves behavior
  with new actions disabled).
- **Training smoke:** a `--smoke` GRPO run completes one step on a tiny model without NaNs.
- **Coverage expectation:** new modules ≥ 90% line coverage; no net decrease in suite count
  (currently ~2079 tests).
- **Acceptance metric (Objective):** on the eval set ([src/training/eval/bamboogle.py](src/training/eval/bamboogle.py)),
  trained policy uses fewer mean `search_rounds` at ≥ baseline correctness.

---

## Boundaries

- **Always:** run `pytest` + `ruff` before commit; thread all state through `AgentState`; keep new
  reward terms default-weight 0 so existing presets are byte-stable; degrade-don't-crash when web key
  or reranker is missing; commit a copy of this spec **and** the plan on the PR branch (repo convention).
- **Ask first:** changing `SearchRewardConfig` preset *defaults* (vs. adding new zero-weight fields);
  adding a new model/network head; adding a dependency; touching the M10 routing layer or retrieval
  server response contracts; introducing a learned stop-classifier (out of current scope).
- **Never:** commit secrets/API keys; commit directly to `main` (feature branch + PR per repo rules);
  delete or weaken existing tests to make new code pass; remove the heuristic evidence safety rail.

---

## Success Criteria

1. `AgentState` with exactly the six fields is the loop's primary state; `AcentContext`/metrics derive from it.
2. Five components exist as separate modules under `src/agents/components/`, each unit-tested in isolation.
3. The policy can emit and the loop can execute: `search(retriever=web)`, `search(retriever=vector_db)`,
   `rerank`, `answer` — verified by a loop test exercising each path.
4. `SearchRewardConfig` gains `retriever_cost`, `rerank_cost`, `evidence_gain` terms; with all new
   weights = 0, every existing reward test is unchanged (regression green).
5. A GRPO `--smoke` run completes one training step using the new action space.
6. Full suite green; new modules ≥ 90% coverage; no test count regression.
7. (Stretch / acceptance) trained policy reduces mean `search_rounds` at equal-or-better correctness on the eval set.

---

## Open Questions — all resolved (defaults locked 2026-06-23)

1. **Stop action:** ✅ **RESOLVED** — reward-driven stopping (assumption #1); no dedicated learned stop
   head. Heuristic evidence gate stays as the safety rail.
2. **Web backend for training:** ✅ **RESOLVED** — **cached/offline web corpus** for training,
   **live web only at serving**. Durability via Phase A0 if live web is later enabled in rollouts.
   See [the plan](docs/superpowers/plans/agent-framework-grpo-plan.md).
3. **Retriever cost ratio:** ✅ **RESOLVED (default)** — web priced at **5×** a vector-DB round
   via `retriever_cost`. Single tunable; revisit after first training run.
4. **Reranker price:** ✅ **RESOLVED (default)** — **flat `rerank_cost` per call** (not per-candidate).
5. **evidence_gain reward:** ✅ **RESOLVED (default)** — reward **Δ`evidence_score` per round**
   (shapes mid-trajectory search informativeness), not terminal-only.
