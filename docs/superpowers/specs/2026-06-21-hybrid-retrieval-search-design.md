# Hybrid Retrieval Search — Design Spec

**Date:** 2026-06-21
**Status:** Approved

## Problem

The web search path's internal "Local Retrieval" provider runs `demo.py`, a TF-IDF-only
`/retrieve` server. The repo already contains hybrid-retrieval machinery — `DenseRetriever`
(FAISS), `SparseRetriever`, and `combine_retrieval_results` (Reciprocal Rank Fusion) — but
it is not on the search path: it lives behind `server.py`'s Pyserini-based
`RetrievalService`, which exposes a different `/search` contract and requires Java. So the
product gets keyword-only internal retrieval despite having embedding + fusion code
available.

## Goal

Serve **RRF-fused dense + sparse** results from the internal retrieval provider, reusing the
existing hybrid library, without adding a Java dependency and without changing the web/agent
layers. Dense uses a local e5 embedding model on Apple Silicon (MPS).

## Scope

- New: `src/internal/servers/retrieval/hybrid.py` (`/retrieve` server)
- Reuse: `DenseRetriever` / `DenseRetrieverConfig.for_e5_base_v2`
  (`src/internal/document_index/retrieval.py`), `combine_retrieval_results`
  (`src/internal/document_index/hybrid_retriever.py`), `write_dense_faiss_index`
  (`src/internal/document_index/index_builder.py`), `TfidfRetriever`
  (`src/internal/servers/retrieval/demo.py`)
- Tests under `tests/unit/`

Out of scope: Pyserini/Java BM25; cross-encoder reranking (web backend already supports
`rerank_url` separately); changing the web backend, agent loops, or frontend.

## Design

### Components

- **`hybrid.py`** — a `/retrieve` FastAPI server with the **same request/response contract
  as `demo.py`**, so `SearchClient` and the whole web stack are unchanged.
- **Dense half (reuse):** `DenseRetriever` via `DenseRetrieverConfig.for_e5_base_v2`
  (`intfloat/e5-base-v2`), `device="mps"`, over a FAISS index built from the corpus.
- **Sparse half (reuse):** `TfidfRetriever` from `demo.py` (sklearn, no Java).
- **Fusion (reuse):** `combine_retrieval_results([dense, sparse], rrf_k=60)`.

### Startup

1. Load corpus from `--corpus_path`.
2. Build `TfidfRetriever` in-memory (instant).
3. Ensure the dense FAISS index exists: if absent, embed the corpus with e5 (MPS) via
   `write_dense_faiss_index` and cache it under `data/index/e5-<corpus-hash>/`; then load
   `DenseRetriever`.

### Per-request data flow

```
query → dense: e5(MPS) → FAISS → top(2·topk)
      → sparse: TF-IDF → top(2·topk)
      → combine_retrieval_results([dense, sparse], rrf_k=60) → topk
      → {"results": [...]}   (demo /retrieve shape; list-of-lists for batch)
```

Each leg over-fetches `2·topk` candidates so RRF has overlap to reward documents both
legs surface, before truncating the fused list to `topk`.

The fused result set is formatted to the exact `demo.py` `/retrieve` response shape,
honoring `query` vs batch `queries` and `return_scores`.

### Graceful degradation

If the dense half cannot initialize (torch/sentence-transformers missing, model download
fails, MPS unavailable), the server logs a warning and falls back to **TF-IDF-only**, still
serving the `/retrieve` contract. A `--no-dense` flag forces sparse-only.

### Wiring / operations

Run `hybrid.py` on port 8001 in place of `demo.py`; no web-backend change. The
3-process startup docs (CLAUDE.md) gain the hybrid-server command as the internal
retrieval option.

## Testing

- **Fusion:** a doc ranked high by *both* legs outranks a doc ranked high by only one
  (through `combine_retrieval_results`).
- **Contract parity:** hybrid `/retrieve` returns the same shape as `demo.py` (single-query
  dict, batch list-of-lists, `return_scores` honored).
- **Degradation:** dense forced to fail / `--no-dense` → server serves TF-IDF-only, 200.
- **Dense contributes:** with a stub embedder (no e5 download in CI), a query returns a
  dense-only hit the sparse leg misses.

The real e5/MPS path is excluded from CI (stub the embedder); exercised via manual E2E.

## Success criteria

1. `hybrid.py` serves RRF-fused dense+sparse results on the `demo.py` `/retrieve` contract.
2. Web backend "Local Retrieval" returns hybrid results with no web/agent changes.
3. Dense-init failure degrades to TF-IDF-only rather than crashing.
4. `pytest` green; no Java dependency added.
