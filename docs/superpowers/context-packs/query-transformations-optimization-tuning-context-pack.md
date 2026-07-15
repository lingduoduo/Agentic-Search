# Generated Context Pack

# Query Transformations Optimization Tuning

## Sources

- [Specification: 2026-06-23-query-transformations-optimization-tuning-design.md](../specs/2026-06-23-query-transformations-optimization-tuning-design.md)
- [Plan: 2026-06-23-query-transformations-optimization-tuning.md](../plans/2026-06-23-query-transformations-optimization-tuning.md)

## Specification Context

### Non-goals

- No new files/modules. No new wrapper layers. No public API/interface changes
  to `transform(...)` / `retrieval_variants(...)` signatures.
- No frontend changes.
- No swap of the retrieval backend, reranker, or embedding model.
- No changes to default behavior when `QT_*` flags are unset (must stay zero-overhead).

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
   defaults are justified by that table (committed as a short results note in the plan).
3. **Unit tests** — every tuned behavior (prompt change, fusion weight, dedup
   threshold, router calibration, new rewrite mode) is covered by unit tests; full
   `pytest` suite stays green (≥ current 2011 tests, no regressions).

A change that improves the grid but regresses the eval gate, or vice versa, is rejected.

### 3. Scope — Tuning Levers (by existing file)

Each lever stays inside the listed file. No new files.

### Tests + lint

pytest
pytest tests/unit/retrieval -v
ruff check . --fix && ruff format .
```

### 7. Testing Strategy

- **Unit (primary):** deterministic tests with a stub LLM for every prompt/mode
  change; numeric tests for fusion weights, MMR λ, dedup threshold, variant
  ordering, router thresholds/fallback. New `rewrite()` mode gets its own tests.
- **Eval (gate):** `eval_runner` recall@10 / nDCG@10 / MRR vs
  `data/eval/baseline_metrics.json`; p99 SLO via `qt_slo_exceeded`.
- **Benchmark (selection):** `run_query_transform_benchmark` grid table justifies
  chosen defaults; results summarized in the plan.
- No new integration dependencies; integration suite untouched.

## Implementation Plan Context

### Global Constraints

- Never commit to `main`. Branch `feat/query-transform-optimization-tuning` is already created.
- All `QT_*` env flags MUST keep defaulting to disabled (zero overhead when unset).
- No new files/modules; edits stay inside the listed pre-existing files.
- No change to existing public signatures `transform(...)` / `retrieval_variants(...)`; the only new public surfaces are the `rewrite` config flag/field/method and the `"rewrite"` router label (pre-approved).
- Fallback-safe: any LLM failure in a leg degrades that leg to empty/None; pipeline continues.
- Acceptance gates (all three): `eval_runner` recall@10/nDCG@10 ≥ `data/eval/baseline_metrics.json` with a measurable gain on one metric and p99 SLO held; `query_transform_benchmark` grid justifies tuned defaults; full `pytest` green with no regressions.
- `ruff check . --fix && ruff format .` clean before each commit.

---

### Task 1: Canonical query-rewrite leg (end-to-end)

Adds a single normalized/canonical rewrite of the query (typo + verbosity cleanup), distinct from step-back's broadening. Wired through config → enhancer → pipeline → bundle → variant ordering → cache signature.

**Files:**
- Modify: `src/context/query_transform.py` (config field, signature, bundle field + ordering, `_build_jobs`, `_assemble`, `from_env`)
- Modify: `src/context/query_enhancer.py` (add `_REWRITE_PROMPT` + `QueryEnhancer.rewrite`)
- Test: `tests/unit/test_query_enhancer.py`, `tests/unit/test_query_transform.py`

**Interfaces:**
- Produces: `QueryEnhancer.rewrite(query: str) -> str | None`; `QueryTransformConfig.rewrite: bool`; `TransformedQueryBundle.rewrite: str | None`; env flag `QT_REWRITE`.
- Consumes: existing `_llm_text`, `ChatMessage`, `QueryEnhancer` fallback pattern.

- [ ] **Step 1: Write the failing enhancer test**

In `tests/unit/test_query_enhancer.py`:

```python
def test_rewrite_returns_cleaned_query():
    llm = _llm("What is FAISS?")
    enhancer = QueryEnhancer(llm)
    result = enhancer.rewrite("uhh wht is  faiss???")
    assert result == "What is FAISS?"


def test_rewrite_falls_back_to_none_without_llm():
    assert QueryEnhancer(None).rewrite("anything") is None
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_query_enhancer.py::test_rewrite_returns_cleaned_query -v`
Expected: FAIL — `AttributeError: 'QueryEnhancer' object has no attribute 'rewrite'`

- [ ] **Step 3: Implement `rewrite` in `query_enhancer.py`**

Add the prompt constant after `_STEP_BACK_PROMPT` (line ~33):

```python

_[Section compacted.]_

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

Replace the body to bias toward retrieval-friendly, non-overlapping sub-questions:

```python
_DECOMPOSE_PROMPT = """Break the question into 2-4 focused, non-overlapping sub-questions that together cover its full intent.
Each sub-question must be self-contained and keyword-rich so it retrieves well on its own.
Return one sub-question per line. Do not number them. Do not repeat the original question.
If the question is already atomic, return it unchanged on a single line.

Question: {query}""".strip()
```

- [ ] **Step 3: Tighten `_STEP_BACK_PROMPT`**

```python
_STEP_BACK_PROMPT = """Write one broader background question whose answer provides the concepts needed to answer the original.
Keep the same domain and key entities; generalise only the specifics.
Return only the rewritten question, no explanation.

Question: {query}
Broader question:""".strip()
```

_[Section compacted.]_

### Task 3: Refine adaptive MMR λ + fix docstring/code mismatch

`adaptive_mmr_lambda` claims "≤3 → 0.8" but checks `n <= 2`. Fix the mismatch and add a graded mid-range so medium queries lean slightly toward relevance.

**Files:**
- Modify: `src/internal/retrieval/fusion_learner.py` (`adaptive_mmr_lambda`)
- Test: `tests/unit/retrieval/test_fusion_learner.py`

**Interfaces:**
- Produces: `adaptive_mmr_lambda(query: str) -> float` with thresholds: ≤3 tokens → 0.8; 4–6 → 0.6; 7–9 → 0.5; ≥10 → 0.3.

- [ ] **Step 1: Write the failing test**

In `tests/unit/retrieval/test_fusion_learner.py`:

```python
def test_adaptive_mmr_lambda_tiers():
    from src.internal.retrieval.fusion_learner import adaptive_mmr_lambda

    assert adaptive_mmr_lambda("faiss") == 0.8            # 1 token (≤3)
    assert adaptive_mmr_lambda("a b c") == 0.8            # 3 tokens (≤3, was 0.5 before)
    assert adaptive_mmr_lambda("one two three four") == 0.6   # 4-6
    assert adaptive_mmr_lambda("one two three four five six seven") == 0.5  # 7-9
    assert adaptive_mmr_lambda(" ".join(["w"] * 12)) == 0.3  # ≥10
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/retrieval/test_fusion_learner.py::test_adaptive_mmr_lambda_tiers -v`
Expected: FAIL — `assert 0.5 == 0.8` for the 3-token case

- [ ] **Step 3: Update `adaptive_mmr_lambda`**

```python
def adaptive_mmr_lambda(query: str) -> float:
    """Return MMR lambda based on query length.

    Short  (≤3 tokens)  → 0.8  (prioritise relevance).
    Medium (4-6 tokens) → 0.6.
    Medium (7-9 tokens) → 0.5  (balanced).
    Long   (≥10 tokens) → 0.3  (prioritise diversity).
    """

_[Section compacted.]_

### Task 4: Router — add `rewrite` label, improve heuristic, expand & retrain seed

Makes routing the cost lever: easy/keyword queries skip expensive LLM legs; noisy/long queries get `rewrite`. Adds the `rewrite` label end-to-end and retrains the joblib model.

**Files:**
- Modify: `src/internal/retrieval/query_router.py` (`ROUTER_LABELS`, `_heuristic`)
- Modify: `src/training/train_query_router.py` (`SEED_DATA` widened to 7 columns + comment)
- Test: `tests/unit/retrieval/test_query_router.py`

**Interfaces:**
- Consumes: `QueryTransformConfig.rewrite` (Task 1).
- Produces: `ROUTER_LABELS` with `"rewrite"` appended (index 6); `_heuristic` returns `rewrite=True` for long noisy queries.

- [ ] **Step 1: Write failing router tests**

In `tests/unit/retrieval/test_query_router.py`:

```python
def test_router_labels_include_rewrite_last():
    from src.internal.retrieval.query_router import ROUTER_LABELS
    assert ROUTER_LABELS[-1] == "rewrite"
    assert len(ROUTER_LABELS) == 7


def test_heuristic_routes_rewrite_for_long_noisy_query():
    from src.internal.retrieval.query_router import QueryRouter
    cfg = QueryRouter().predict(
        "uhh so basically what is the deal with faiss vs scann and which one is faster i think"
    )
    assert cfg.rewrite is True


def test_heuristic_short_keyword_skips_expensive_legs():
    from src.internal.retrieval.query_router import QueryRouter
    cfg = QueryRouter().predict("faiss index")
    assert cfg.hyde is False and cfg.decompose is False and cfg.multi_query is False
    assert cfg.keywords is True
```

- [ ] **Step 2: Run to verify failure**

_[Section compacted.]_

### Task 5: Acceptance — benchmark grid, eval gate, full suite, PR

Proves all three gates and ships.

**Files:**
- Run-only + a short results note appended to this plan.

- [ ] **Step 1: Full test suite green**

Run: `pytest`
Expected: PASS, count ≥ prior 2011, zero failures.

- [ ] **Step 2: Benchmark grid over candidate configs**

Run the existing benchmark harness (via its test driver or a one-off script that calls `run_query_transform_benchmark`) over configs that toggle `rewrite`, `multi_query`, `decompose`, `hyde`. Capture the sorted recall@10 table.

Run: `pytest tests/unit/retrieval/test_query_transform_benchmark.py -v`
Expected: PASS; record the winning config signature.

- [ ] **Step 3: Eval gate vs baseline**

Start a retrieval server (`python3 -m src.internal.servers.retrieval.demo --corpus_path data/corpus.jsonl`) and run:

Run: `python -m src.internal.retrieval.eval_runner --dataset data/eval/qa_pairs.jsonl --top_k 10 --retrieval_url http://localhost:8001/retrieve`
Expected: recall@10 and nDCG@10 ≥ values in `data/eval/baseline_metrics.json`, with a measurable gain on ≥1 metric; p99 transform latency does not trip `qt_slo_exceeded`.

- [ ] **Step 4: Append a results note to this plan**

Add a short `## Results (M9)` section: baseline vs tuned recall@10/nDCG@10/MRR, the winning benchmark config, and the p99 latency. Commit.

```bash
git add docs/superpowers/plans/2026-06-23-query-transformations-optimization-tuning.md
git commit -m "docs(plan): record M9 eval + benchmark results"
```

- [ ] **Step 5: Push and open PR**

```bash
git push -u origin feat/query-transform-optimization-tuning

_[Section compacted.]_

### Acceptance

- eval_runner recall@10/nDCG@10 ≥ baseline (see plan Results note); p99 SLO held.
- query_transform_benchmark grid justifies tuned defaults.
- Full pytest green, no regressions.

Spec: docs/superpowers/specs/2026-06-23-query-transformations-optimization-tuning-design.md
Plan: docs/superpowers/plans/2026-06-23-query-transformations-optimization-tuning.md

🤖 Generated with Claude Code
EOF
)"
```

---

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
