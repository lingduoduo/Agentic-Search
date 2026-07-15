# Generated Context Pack

# Reranking Optimization

## Sources

- [Specification: 2026-06-19-reranking-optimization-design.md](../specs/2026-06-19-reranking-optimization-design.md)
- [Plan: 2026-06-19-reranking-optimization.md](../plans/2026-06-19-reranking-optimization.md)

## Specification Context

### Overview

Extends the existing `Reranker` class (BGE local + Cohere remote) with latency optimizations and quality improvements through layered wrapper composition. The `Reranker` leaf is unchanged; wrappers compose on top of it.

### Architecture

All wrappers share the same interface as `Reranker`:

`AsyncReranker` additionally exposes an `async` variant used by `RetrievalService`.

## Implementation Plan Context

### Task 1: PassageTruncator + Reranker integration

**Files:**
- Create: `src/internal/retrieval/passage_truncator.py`
- Modify: `src/internal/retrieval/reranker.py` (add truncation call in `_rerank_local`)
- Test: `tests/unit/retrieval/test_passage_truncator.py`

**Interfaces:**
- Produces: `PassageTruncator(max_tokens=512)`, `PassageTruncator.truncate(text: str) -> str`, `PassageTruncator.from_env() -> PassageTruncator`

- [ ] **Step 1: Write the failing tests**

- [ ] **Step 2: Run tests to verify they fail**

Expected: `ModuleNotFoundError: No module named 'src.internal.retrieval.passage_truncator'`

- [ ] **Step 3: Implement PassageTruncator**

- [ ] **Step 4: Wire truncator into Reranker._rerank_local**

…

### Task 2: AsyncReranker

**Files:**
- Create: `src/internal/retrieval/async_reranker.py`
- Test: `tests/unit/retrieval/test_async_reranker.py`

**Interfaces:**
- Consumes: any object with `rerank(query, results, top_k) -> list[RetrievalResult]`
- Produces:
  - `RerankerTimeoutError(RuntimeError)`
  - `AsyncReranker(base_reranker, *, timeout_ms=500, max_workers=4)`
  - `AsyncReranker.rerank(query, results, top_k) -> list[RetrievalResult]` — sync shim with thread timeout
  - `AsyncReranker.arerank(query, results, top_k) -> list[RetrievalResult]` — async entry point
  - `AsyncReranker.from_env(base_reranker) -> AsyncReranker`

- [ ] **Step 1: Write the failing tests**

- [ ] **Step 2: Run tests to verify they fail**

…

### Task 3: CachedReranker

**Files:**
- Create: `src/internal/retrieval/cached_reranker.py`
- Test: `tests/unit/retrieval/test_cached_reranker.py`

**Interfaces:**
- Consumes: any object with `rerank(query, results, top_k) -> list[RetrievalResult]`; `ResultCache` serialization pattern (json + `asdict`)
- Produces:
  - `CachedReranker(base_reranker, redis_client=None, *, ttl_seconds=300)`
  - `CachedReranker.rerank(query, results, top_k) -> list[RetrievalResult]`
  - `CachedReranker.stats() -> dict` — `{hits, misses, hit_rate}`
  - `CachedReranker.from_env(base_reranker)` — returns `base_reranker` unchanged if `RERANKER_CACHE_REDIS_URL` not set

- [ ] **Step 1: Write the failing tests**

…

### Final verification

- [ ] **Run full test suite**

```
pytest tests/unit/retrieval/ -v
```
Expected: all PASS (onnx tests skipped without `optimum`)

- [ ] **Run ruff**

```
ruff check src/internal/retrieval/async_reranker.py \
           src/internal/retrieval/cached_reranker.py \
           src/internal/retrieval/passage_truncator.py \
           src/internal/retrieval/onnx_reranker.py \
           src/internal/retrieval/two_stage_reranker.py \
           src/internal/retrieval/reranker_benchmark.py \
           --fix && ruff format src/internal/retrieval/
```
Expected: no errors after fix

- [ ] **Verify spec coverage**

| Spec requirement | Task |
|---|---|

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
