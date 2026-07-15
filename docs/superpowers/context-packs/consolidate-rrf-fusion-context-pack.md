# Generated Context Pack

# Consolidate Rrf Fusion

## Sources

- [Specification: 2026-07-03-consolidate-rrf-fusion-design.md](../specs/2026-07-03-consolidate-rrf-fusion-design.md)
- [Plan: 2026-07-03-consolidate-rrf-fusion.md](../plans/2026-07-03-consolidate-rrf-fusion.md)

## Specification Context

### Non-goals

- No change to the max-dedup `combine_retrieval_results` in `search_runner.py`.
- No change to MMR (`mmr_rerank`), query transform, caching, or the agent loop.
- No new fusion behavior — pure de-duplication of existing logic.

## Implementation Plan Context

### Global Constraints

- **Behavior-preserving.** Same formula/keys/tie-breaks; public signatures unchanged; no existing test modified beyond deleting the dead module's test.
- **In scope:** the 4 RRF sites + `index_optimizer.py`. Out: the max-dedup `combine_retrieval_results`, MMR, query transform, caching, agent loop.

---

### Task 1: Generic core + fusion.py adapters

- [x] **Step 1:** Add `rrf_rank(result_sets, key_fn, *, weights=None, rrf_k=_RRF_K)` to `fusion.py`.
- [x] **Step 2:** Rewrite `rrf_fuse`, `weighted_rrf_fuse`, `variant_weighted_rrf_fuse` as adapters over `rrf_rank` (key=doc_id, first-seen, rebuild RetrievalResult); keep signatures + fallbacks.
- [x] **Verify:** `pytest tests/**/test*fusion*` (and fusion-touching tests) green.

### Task 2: Route the three external sites

- [x] **Step 1:** `document_index/hybrid.py::_rrf_merge` → `rrf_rank` with `(document_id, chunk_ind)` key + prefer-keyword map + top_k.
- [x] **Step 2:** `document_index/hybrid_retriever.py::combine_retrieval_results` → `rrf_rank` with `["id"]` key, rebuild dicts.
- [x] **Step 3:** `process_search_query.py::weighted_reciprocal_rank_fusion` → `rrf_rank` with `(url, contents[:100])` key + weights + best-score tie-break.
- [x] **Verify:** each site's tests pass unchanged.

### Task 3: Dead-code + full verification

- [x] **Step 1:** Delete the dead modules + tests (`index_optimizer`, `chunk_config`, `onnx_reranker`, `embedding_cache`); confirm zero non-test references for each first.
- [x] **Step 2:** `ruff check` clean; the broad retrieval/fusion/hybrid/search unit suites green.
- [x] **Step 3:** Grep: RRF `1/(k+rank)` formula appears only inside `rrf_rank`.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
