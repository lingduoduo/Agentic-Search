# Spec: Query Transformations — Optimization & Tuning (M9)

- **Date:** 2026-06-23
- **Status:** Draft (awaiting approval)
- **Branch:** `feat/query-transform-optimization-tuning` (to be created; never commit to `main`)
- **Plan:** `docs/superpowers/plans/2026-06-23-query-transformations-optimization-tuning.md` (committed on the same branch)

## 1. Objective

The query-transform stack (M1–M8) already implements every technique in scope —
Multi-Query Retrieval, RAG-Fusion (RRF), Query Decomposition, Step-Back / Query
Rewriting, HyDE, Query Routing, and Query Construction. This milestone **tunes
and optimizes the existing implementation** to improve retrieval quality, cut
latency/cost, and make per-query routing smarter — **without adding new
modules/files**. Any technique "gap fill" (e.g. a true canonical query-rewrite
leg distinct from step-back) is added as a *mode/method inside an existing
module*, never a new file.

Four goals, all in scope:

1. **Retrieval quality** — raise recall@k / nDCG@k / MRR on the labeled eval set
   via better transform prompts, fusion weights, MMR λ, semantic dedup, and
   variant ordering/selection.
2. **Latency & cost** — lower p99 transform latency and LLM token spend (async
   timeout/worker tuning, cache TTL/embedding cache, fewer/cheaper LLM calls,
   routing that *skips* transforms for queries that don't need them).
3. **Smarter routing** — retrain/recalibrate `QueryRouter` so it picks the right
   transforms per query instead of running all enabled legs blindly.
4. **Fill technique gaps in place** — add a canonical query-rewrite mode and tune
   RAG-Fusion where the current behavior is thin, inside existing modules.

**Target users:** Operators of the retrieval stack (us) who toggle `QT_*` flags
in production, plus downstream RAG quality. No end-user-facing UI change.

### Non-goals
- No new files/modules. No new wrapper layers. No public API/interface changes
  to `transform(...)` / `retrieval_variants(...)` signatures.
- No frontend changes.
- No swap of the retrieval backend, reranker, or embedding model.
- No changes to default behavior when `QT_*` flags are unset (must stay zero-overhead).

## 2. Acceptance Criteria (all three gates required)

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

## 3. Scope — Tuning Levers (by existing file)

Each lever stays inside the listed file. No new files.

### Quality
- `src/context/query_enhancer.py` — sharpen decompose / HyDE / step-back prompts
  for higher-recall, less-redundant variants; **add a `rewrite(query)` canonical
  query-rewrite mode** (typo/verbosity normalization, distinct from step-back),
  wired through `QueryTransformConfig` + `_build_jobs` in `query_transform.py`.
- `src/internal/retrieval/multi_query.py` — tighten paraphrase prompt; avoid
  near-duplicate paraphrases at generation time.
- `src/internal/retrieval/fusion.py` — tune `rrf_k`, weighted-RRF weights, and
  adaptive MMR λ; refine `dedup_variants` threshold behavior.
- `src/context/query_transform.py` — tune `retrieval_variants` ordering/selection
  (which variants survive truncation to `max_variants`); original always last (unchanged contract).
- `src/internal/retrieval/fusion_learner.py` — re-run weight/λ grid search to
  produce the new defaults.

### Latency & cost
- `src/internal/retrieval/async_query_transform.py` — tune `QT_TRANSFORM_TIMEOUT_MS`,
  `QT_MAX_WORKERS` defaults; ensure independent legs truly overlap.
- `src/internal/retrieval/cached_query_transform.py` — tune `QT_CACHE_TTL_SECONDS`;
  confirm filter re-merge stays correct.
- Reduce LLM calls where a single prompt can cover multiple legs without quality loss.

### Routing
- `src/internal/retrieval/query_router.py` + `src/training/train_query_router.py` —
  expand the labeled `SEED_DATA`, recalibrate decision thresholds, and improve the
  heuristic fallback so easy queries skip expensive legs. Retrain the joblib artifact.

### Config defaults
- `QT_*` env defaults may be re-tuned (e.g. timeout, dedup threshold, multi-query N,
  fusion weighting) but every flag **must keep defaulting to disabled / zero-overhead**.

## 4. Commands

```bash
# Setup
pip install -e . && pip install -r requirements.txt

# Eval gate (baseline vs tuned)
python -m src.internal.retrieval.eval_runner --dataset data/eval/qa_pairs.jsonl --top_k 10
python -m src.internal.retrieval.eval_runner --dataset data/eval/qa_pairs.jsonl --top_k 10 \
  --retrieval_url http://localhost:8001/retrieve

# Benchmark grid (config selection)
python -m src.internal.retrieval.query_transform_benchmark  # via test/driver harness

# Router retrain
python -m src.training.train_query_router

# Tests + lint
pytest
pytest tests/unit/retrieval -v
ruff check . --fix && ruff format .
```

## 5. Project Structure (files touched — all pre-existing)

```
src/context/query_transform.py            # variant ordering, wire rewrite mode + config flag
src/context/query_enhancer.py             # prompts; new rewrite() mode
src/internal/retrieval/multi_query.py     # paraphrase prompt tuning
src/internal/retrieval/fusion.py          # rrf_k / weights / MMR λ / dedup
src/internal/retrieval/fusion_learner.py  # regenerate tuned weights
src/internal/retrieval/async_query_transform.py   # timeout/worker defaults
src/internal/retrieval/cached_query_transform.py  # TTL tuning
src/internal/retrieval/query_router.py    # thresholds + heuristic fallback
src/training/train_query_router.py        # expanded SEED_DATA, retrain
tests/unit/** , tests/unit/retrieval/**   # coverage for each lever
docs/superpowers/specs|plans/2026-06-23-*  # this spec + plan
```

## 6. Code Style

- Match existing module idioms: frozen dataclasses for config, `from __future__
  import annotations`, env reads via the local `_bool` helper, fallback-safe
  transformers (any LLM failure degrades that leg to empty/None, pipeline continues).
- Surgical edits only — change only lines tied to a tuning lever; no drive-by refactors.
- `ruff check`/`ruff format` clean.

## 7. Testing Strategy

- **Unit (primary):** deterministic tests with a stub LLM for every prompt/mode
  change; numeric tests for fusion weights, MMR λ, dedup threshold, variant
  ordering, router thresholds/fallback. New `rewrite()` mode gets its own tests.
- **Eval (gate):** `eval_runner` recall@10 / nDCG@10 / MRR vs
  `data/eval/baseline_metrics.json`; p99 SLO via `qt_slo_exceeded`.
- **Benchmark (selection):** `run_query_transform_benchmark` grid table justifies
  chosen defaults; results summarized in the plan.
- No new integration dependencies; integration suite untouched.

## 8. Boundaries

**Always**
- Work on a feature branch; create it before the first commit.
- Keep `QT_*` flags defaulting to disabled (zero overhead when unset).
- Run `pytest` + `ruff` green before opening the PR; commit spec **and** plan on the branch.
- Open a PR after the work with a unique, specific title.

**Ask first**
- Any change to a public signature (`transform`, `retrieval_variants`, router labels)
  or to default-on behavior.
- Any net-new file/module (default answer is "no" for this milestone).
- Adding a heavyweight dependency or changing the embedding/rerank model.

**Never**
- Commit directly to `main`.
- Delete/rewrite pre-existing query-transform modules or unrelated code.
- Change frontend, retrieval backend, or eval dataset contents to "make numbers move".
```
