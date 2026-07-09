# Simulated Preference Judge → Synthetic AI-Feedback for GRPO

**Date:** 2026-07-09
**Status:** Approved (design)
**Branch:** `feat/simulated-preference-judge`

## Problem

Today the training stack has no learned reward model and no LLM-as-judge. GRPO
rewards come from heuristic scorers that all require **ground truth** (F1 /
exact-match), and the only human-preference signal is a direct ±1 thumbs value
loaded from the feedback DB (`load_feedback_examples`, `src/training/data.py`).

We want to demonstrate **RL from AI feedback (RLAIF)**: an LLM-style judge that
scores agent answers and feeds those scores into GRPO as synthetic preference
data. Standing up a real judge LLM (endpoint, secret, per-run call volume) is
premature. Instead we ship a **simulated, deterministic pointwise judge** behind
the interface GRPO already expects, so the real LLM becomes a later drop-in swap.

## Non-Goals (YAGNI)

- **No real LLM judge adapter** in this change. The `BatchJudgeFn` interface is
  already threaded through GRPO; a real judge is a later one-liner behind it.
- **No reward-model training** (no Bradley-Terry / value head). That is the
  separate "flavor B" and is explicitly out of scope.
- **No new dataset.** We reuse the existing bamboogle seed prompts.
- **No changes to the GRPO core algorithm.** The judge plugs into the existing
  `batch_judge_fn` parameter of `score_prompt_group` / `score_prompt_batch`.

## Key Decisions

1. **Judge mode: pointwise score.** The judge returns a scalar in `[0, 1]` per
   answer. GRPO performs the group-relative comparison itself
   (`compute_grpo_outcome_advantages`, `src/training/reward.py`). This fits the
   existing `BatchJudgeFn` seam exactly; no ranking schema or chosen/rejected
   pairs are introduced.

2. **Online, function GRPO calls.** The judge is invoked inside the GRPO loop
   (one batched call per prompt group), not an offline cached dataset. It scores
   the *current* policy's fresh rollouts.

3. **Simulated, not real LLM.** A deterministic reference implementation stands
   in for the LLM. Real LLM = drop-in swap behind the same interface.

4. **Reference-free.** The judge scores answer *quality/form* from the answer's
   own features and ignores the `ground_truths` argument. Gold answers are used
   **only for validation** (measuring judge↔correctness agreement), never as a
   judge input.

5. **Seed corpus: bamboogle.** The 5 `data/bamboogle_train/*.parquet` prompts
   are already in the pipeline's expected schema (`data_source='bamboogle'`,
   `reward_model.ground_truth.target`) and small enough that the whole loop runs
   on a laptop (~5 judge calls/epoch).

## Interface (reused, unchanged)

```python
# src/training/reward.py (already exists)
BatchJudgeFn = Callable[[list[str], list[str]], list[float]]
```

`score_prompt_group(..., batch_judge_fn=...)` and
`score_prompt_batch(..., batch_judge_fn=...)` in `src/training/grpo.py` already
accept and consume this via `_score_answers`. A reference-free judge ignores the
second (`ground_truths`) argument.

## Components

### 1. `src/training/judge.py` — `SimulatedPreferenceJudge`

A deterministic, reference-free pointwise quality scorer that mimics an
LLM-as-judge.

- **Input:** an answer string (and, when available, the rollout's
  `AgentLoopOutput` metadata for citation/evidence signal).
- **Output:** a score in `[0, 1]`.
- **Signal (answer-intrinsic, reference-free):**
  - non-degeneracy — non-empty, not trivially repetitive
  - length within a sane band (penalize empty / runaway)
  - presence of a concrete answer span (not pure hedging)
  - citation / evidence support when `loop_output` metadata is present
- **Determinism:** any tie-breaking jitter is derived from a hash of the answer
  text (no `random`/`Date.now()`), so identical inputs always produce identical
  scores. Tests rely on this.
- **API:**
  - `score(answer: str, *, loop_output=None) -> float`
  - `as_batch_judge_fn() -> BatchJudgeFn` — returns a
    `(answers, ground_truths) -> list[float]` closure that ignores
    `ground_truths`.
- Scores are clamped to `[0, 1]`.

### 2. `examples/run_bamboogle_synthetic_grpo.py` — runnable demo

Mirrors `run_bamboogle_eval.py` (server/agent wiring: `LocalServerManager` /
`OpenAIServerManager`, `SearchAgentLoop`) and `run_sft_grpo.py` (GRPO wiring).

Flow:
1. Load the bamboogle prompts (reuse the existing loader path).
2. Sample `N` rollouts per prompt via existing `sample_prompt_batch`.
3. Score each group with `SimulatedPreferenceJudge.as_batch_judge_fn()` passed
   as `batch_judge_fn` to `score_prompt_batch`.
4. Obtain GRPO group-relative advantages (existing math).
5. **Dump the synthetic dataset to JSONL** for tracking (see schema below).
6. Print the validation report.

CLI mirrors the bamboogle eval where sensible (`--model`, `--local`,
`--server_url`, `--search_url`, `--num_rollouts`, `--output`,
`--temperature`, `--max_tokens`).

### 3. Synthetic-dataset JSONL dump (for tracking)

One record per prompt group:

```json
{
  "prompt": "Who was president ... when Citibank was founded?",
  "data_source": "bamboogle",
  "gold": ["james madison"],
  "rollouts": [
    {"answer": "James Madison", "judge_score": 0.87, "advantage": 0.21,
     "exact_match": 1, "contains_match": 1},
    {"answer": "", "judge_score": 0.05, "advantage": -0.63,
     "exact_match": 0, "contains_match": 0}
  ]
}
```

The dump is the artifact for inspecting what synthetic preferences the judge
produced and how they map to GRPO advantages.

### 4. Validation report

A small function comparing judge scores to bamboogle gold (exact/contains
match) and reporting an agreement/correlation number.

**Honest caveat (kept in the spec):** a purely reference-free judge scores
form/quality, which may correlate weakly with *factual correctness* on hard
two-hop questions. A low correlation is an expected, informative result — it
demonstrates the loop end-to-end and motivates a grounded/LLM judge later. The
stub is not claimed to be a good judge.

## Data Flow

```
bamboogle prompts
  → SearchAgentLoop rollouts (N per prompt)
  → SimulatedPreferenceJudge.as_batch_judge_fn()   [reference-free, per group]
  → correctness scores
  → SearchRewardFunction reward components
  → GRPO group-relative advantages
  → JSONL synthetic-dataset dump
gold answers → validation report (agreement vs judge score)
```

## Error Handling

- Judge scores clamped to `[0, 1]`; empty/degenerate answers → low score.
- Deterministic, offline — no network, no secret, no RNG.
- Single-sample groups get advantage `0.0` (existing GRPO behavior, unchanged).

## Testing (deterministic, offline)

- **Judge stability:** identical inputs → identical scores; empty answer →
  low score; well-formed cited answer → higher score.
- **Batch adapter:** `as_batch_judge_fn()` returns correct length; ignores
  `ground_truths`.
- **GRPO integration:** `score_prompt_group` with the sim judge yields
  non-degenerate (non-all-zero) advantages across a varied group.
- **Validation metric:** agreement computation runs and returns a number on a
  small synthetic set.

## Files Touched

| File | Change |
|------|--------|
| `src/training/judge.py` | new — `SimulatedPreferenceJudge` |
| `examples/run_bamboogle_synthetic_grpo.py` | new — runnable demo + JSONL dump |
| `tests/unit/test_simulated_judge.py` | new — judge + adapter + integration + validation tests |
| `src/training/__init__.py` | export `SimulatedPreferenceJudge` (follow existing export pattern) |

No changes to `src/training/grpo.py` or `src/training/reward.py` — the seam
already exists.

## Future Work (not this change)

- Real LLM judge adapter behind `BatchJudgeFn` (vLLM/OpenAI-compatible endpoint
  or Anthropic API — consult the `claude-api` skill for model IDs/params).
- Optional grounded/hybrid judge term for factual-QA correlation.
- Offline cached synthetic-preference dataset (front half of reward-model
  training, "flavor B").
