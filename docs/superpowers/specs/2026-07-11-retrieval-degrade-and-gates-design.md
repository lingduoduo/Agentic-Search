# Spec: Retrieval degrade-on-rerank-timeout + QT_REWRITE gate + weighted-RRF weight fix

Date: 2026-07-11
Component: `src/internal/retrieval/service.py`

## Problem

Three verified defects in `RetrievalService`:

### R1 — Async reranker timeout crashes the whole request (HIGH)
`search()` calls `self._reranker.rerank(query, fused, top_k)` with no error
handling. When `RERANKER_ASYNC=1`, the reranker is an `AsyncReranker` whose
`.rerank()` raises `RerankerTimeoutError` on timeout
(`src/internal/retrieval/async_reranker.py`). Nothing catches it, so a slow
reranker fails the ENTIRE search request instead of returning the already-fused
pre-rerank ordering.

### R2 — `QT_REWRITE` missing from the pipeline gate (MEDIUM)
`from_env()` gates pipeline construction on a `_qt_flags` tuple that omits
`QT_REWRITE`, even though `QueryTransformPipeline.from_env`
(`src/context/query_transform.py`) treats `QT_REWRITE` as a first-class enabling
flag. Enabling ONLY `QT_REWRITE=1` leaves `pipeline=None`, so rewriting never
runs.

### R3 — Weighted-RRF weight desync when the original variant fails (MEDIUM)
Variants are retrieved in parallel; failed variants are silently skipped, so the
surviving result-set list can be shorter than `variants`. For
`QT_FUSION_WEIGHTED=1`, the code built `weights = [0.3]*(n-1) + [1.0]`, assuming
the original query (which `retrieval_variants()` places LAST) is still the last
surviving set. If the original's retrieval fails and is dropped, the `1.0` weight
lands on a paraphrase, letting a paraphrase dominate the fused ranking instead of
the user's real query.

## Fix

- **R1**: Wrap the rerank call in `try/except`. On `RerankerTimeoutError` (and, to
  be safe, any `Exception`), log a warning and keep `fused` unchanged; do NOT
  append `+reranked` to `mode` when it falls back. Exact success behavior is
  preserved.
- **R2**: Add `"QT_REWRITE"` to the `_qt_flags` tuple.
- **R3**: Track which surviving result set belongs to the original query by
  identity (the `variants[-1]` future). Build weights so the original set gets
  `1.0` and the rest `0.3`. If the original didn't survive, fall back to
  unweighted `rrf_fuse` rather than faking a 1.0. Non-weighted and single-variant
  paths are unchanged.

## Non-goals

No change to reranker internals, fusion math, or the query-transform pipeline.
Behavior on the success path is byte-for-byte identical.

## Verification

Regression tests in `tests/unit/retrieval/test_service.py`:
- R1: fake reranker raising `RerankerTimeoutError` (and a generic `RuntimeError`)
  → `search()` returns pre-rerank order, mode without `+reranked`, no raise.
- R2: only `QT_REWRITE=1` set → `from_env()` builds a non-None pipeline.
- R3: 3 variants where the LAST (original) fails under `QT_FUSION_WEIGHTED=1` →
  weighted fuse is NOT used (falls back). Plus: when an earlier paraphrase fails
  but the original survives, the `1.0` weight tracks the original set by identity.
