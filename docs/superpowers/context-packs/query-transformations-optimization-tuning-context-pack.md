# Generated Context Pack

# Query Transformations Optimization Tuning

## Sources

- [Specification: 2026-06-23-query-transformations-optimization-tuning-design.md](../specs/2026-06-23-query-transformations-optimization-tuning-design.md)
- [Plan: 2026-06-23-query-transformations-optimization-tuning.md](../plans/2026-06-23-query-transformations-optimization-tuning.md)

## Specification Context

### 2. Acceptance Criteria (all three gates required)

Measured with the **existing** harnesses; no new eval tooling.

1. **Eval-harness deltas** — `python -m src.internal.retrieval.eval_runner
   --dataset data/eval/qa_pairs.jsonl --top_k 10` against a fixed local/served
   index shows **recall@10 and nDCG@10 ≥ the M8 baseline** (`data/eval/baseline_metrics.json`),
   with a measurable gain on at least one of recall@10 / nDCG@10 / MRR for the
   tuned default config. The p99 transform-latency SLO gate (`qt_slo_exceeded`)
   must still pass at the configured `slo_ms`.
2. **Benchmark grid** — `run_query_transform_benchmark` over the candidate
   `QueryTransformConfig` combinations produces a sorted table; the chosen tuned

…

### 3. Scope — Tuning Levers (by existing file)

Each lever stays inside the listed file. No new files.

## Implementation Plan Context

### Task 1: Canonical query-rewrite leg (end-to-end)

Adds a single normalized/canonical rewrite of the query (typo + verbosity cleanup), distinct from step-back's broadening. Wired through config → enhancer → pipeline → bundle → variant ordering → cache signature.

**Files:**
- Modify: `src/context/query_transform.py` (config field, signature, bundle field + ordering, `_build_jobs`, `_assemble`, `from_env`)
- Modify: `src/context/query_enhancer.py` (add `_REWRITE_PROMPT` + `QueryEnhancer.rewrite`)
- Test: `tests/unit/test_query_enhancer.py`, `tests/unit/test_query_transform.py`

**Interfaces:**

…

### Task 2: Tighten transform prompts for recall

Quality lever validated by the eval gate (Task 5), not by asserting exact LLM text. Keep changes surgical; verify existing fallback tests stay green.

**Files:**
- Modify: `src/context/query_enhancer.py` (`_DECOMPOSE_PROMPT`, `_HYDE_PROMPT`, `_STEP_BACK_PROMPT`)
- Modify: `src/internal/retrieval/multi_query.py` (paraphrase prompt)
- Test: `tests/unit/test_query_enhancer.py`, `tests/unit/retrieval/test_multi_query.py` (existing — must stay green)

- [ ] **Step 1: Run the existing prompt tests as a baseline**

Run: `pytest tests/unit/test_query_enhancer.py tests/unit/retrieval/test_multi_query.py -v`
Expected: PASS (record count)

- [ ] **Step 2: Tighten `_DECOMPOSE_PROMPT`**

…

### Task 3: Refine adaptive MMR λ + fix docstring/code mismatch

`adaptive_mmr_lambda` claims "≤3 → 0.8" but checks `n <= 2`. Fix the mismatch and add a graded mid-range so medium queries lean slightly toward relevance.

**Files:**
- Modify: `src/internal/retrieval/fusion_learner.py` (`adaptive_mmr_lambda`)
- Test: `tests/unit/retrieval/test_fusion_learner.py`

**Interfaces:**
- Produces: `adaptive_mmr_lambda(query: str) -> float` with thresholds: ≤3 tokens → 0.8; 4–6 → 0.6; 7–9 → 0.5; ≥10 → 0.3.

- [ ] **Step 1: Write the failing test**

In `tests/unit/retrieval/test_fusion_learner.py`:

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/retrieval/test_fusion_learner.py::test_adaptive_mmr_lambda_tiers -v`

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
