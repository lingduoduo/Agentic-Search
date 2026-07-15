# Consolidate the duplicated RRF fusion — design

**Date:** 2026-07-03
**Status:** Approved (design).
**Scope:** The four copies of Reciprocal Rank Fusion in the retrieval layer.
Extract one generic scorer; route the copies through it. Also delete the dead
retrieval modules (`index_optimizer.py`, `chunk_config.py`, `onnx_reranker.py`,
`embedding_cache.py`).

## Problem

RRF (`score = Σ 1/(k+rank)`) is copy-pasted across four call sites that differ
only in item type, key, and which-object-wins tie-break:

| Site | file | item / key | tie-break |
|------|------|-----------|-----------|
| `rrf_fuse` / `weighted_rrf_fuse` / `variant_weighted_rrf_fuse` | `internal/retrieval/fusion.py` | `RetrievalResult` / `doc_id` | first-seen |
| `_rrf_merge` | `internal/document_index/hybrid.py` | `InferenceChunk` / `(document_id, chunk_ind)` | prefer keyword chunk |
| `combine_retrieval_results` | `internal/document_index/hybrid_retriever.py` | `dict` / `["id"]` | first-seen |
| `weighted_reciprocal_rank_fusion` | `internal/search/process_search_query.py` | `SearchResult` / `(url, contents[:100])` | best original score |

The RRF math is identical; only the glue differs. (The similarly-named
`combine_retrieval_results` in `context/search/retrieval/search_runner.py` is a
**max-score dedup, not RRF** — out of scope.)

Four retrieval modules have **zero non-test references** — dead scaffolding:

| Module | LOC | Symbol | Why dead |
|--------|-----|--------|----------|
| `index_optimizer.py` | 102 | `FaissIndexBuilder`/`HNSWTuner` | nothing imports it |
| `chunk_config.py` | 35 | `ChunkConfig` | nothing imports it |
| `onnx_reranker.py` | 60 | `ONNXReranker` | not in the reranker factory chain (`TwoStage→Cached→Async→Reranker`); never instantiated |
| `embedding_cache.py` | 119 | `CachedEmbedder`/`EmbeddingBatcher` | orphaned duplicate — the live embedding cache is `document_index/embedding_cache.py` (`EmbeddingCache`/`OpenAIEmbedder`) |

## Design

### Generic core — `fusion.py::rrf_rank`

```python
def rrf_rank(
    result_sets: Sequence[Sequence[T]],
    key_fn: Callable[[T], K],
    *,
    weights: Sequence[float] | None = None,
    rrf_k: int = _RRF_K,
) -> list[tuple[K, float]]:
    """Reciprocal Rank Fusion scoring.

    score(key) = Σ_i wᵢ · 1/(rrf_k + rank)   (rank is 1-based within each set)
    Returns (key, score) pairs sorted by score descending. weights defaults to
    all-1.0 (standard RRF); a shorter/omitted weights list falls back to uniform.
    """
```

Pure scoring — no object reconstruction, so it is type-agnostic and reused by
every site.

### Each site becomes an adapter

Each call site keeps only its own key function and its own "which object to keep
per key" rule, then reconstructs its native return type from `rrf_rank`'s
ordered keys:

- **fusion.py** — `rrf_fuse`/`weighted_rrf_fuse`/`variant_weighted_rrf_fuse` call
  `rrf_rank` with `key_fn=lambda r: r.doc_id` (+ the appropriate `weights`),
  keep `first_seen`, rebuild `RetrievalResult`. The three public signatures and
  their fallbacks (uniform when weights is None / lengths mismatch) are
  unchanged.
- **document_index/hybrid.py** — `_rrf_merge` uses
  `key_fn=lambda c: (c.document_id, c.chunk_ind)`, keeps the prefer-keyword map
  (`chunk_map[key]=…` for keyword, `setdefault` for semantic), truncates to
  `top_k`.
- **document_index/hybrid_retriever.py** — `combine_retrieval_results` uses
  `key_fn=lambda d: d["id"]`, first-seen, rebuilds the dict list with fused
  `score`.
- **process_search_query.py** — `weighted_reciprocal_rank_fusion` uses
  `key_fn=lambda r: (r.url, r.contents[:100])`, weights, best-original-score
  tie-break. Its previous `enumerate(...)` (rank from 0) + `k+rank+1` equals the
  core's 1-based `k+rank`, so scores are identical.

### Dead-code removal

Delete the four dead modules and their tests: `index_optimizer.py`,
`chunk_config.py`, `onnx_reranker.py`, `embedding_cache.py` (the latter is a
duplicate of `document_index/embedding_cache.py`, which stays). Each is verified
to have zero non-test references before removal.

## Behavior-preserving

- Same formula, same keys, same per-site tie-breaks → identical outputs.
- Public signatures unchanged at every site.
- Proof: each site's existing unit tests pass unchanged, plus the broad
  retrieval/fusion/hybrid suites.

## Non-goals

- No change to the max-dedup `combine_retrieval_results` in `search_runner.py`.
- No change to MMR (`mmr_rerank`), query transform, caching, or the agent loop.
- No new fusion behavior — pure de-duplication of existing logic.

## Files touched

- **Modify:** `src/internal/retrieval/fusion.py`,
  `src/internal/document_index/hybrid.py`,
  `src/internal/document_index/hybrid_retriever.py`,
  `src/internal/search/process_search_query.py`.
- **Delete:** `src/internal/retrieval/{index_optimizer,chunk_config,onnx_reranker,embedding_cache}.py` + their tests.
