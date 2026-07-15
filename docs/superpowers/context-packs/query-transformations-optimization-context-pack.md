# Generated Context Pack

# Query Transformations Optimization

## Sources

- [Specification: 2026-06-19-query-transformations-optimization-design.md](../specs/2026-06-19-query-transformations-optimization-design.md)
- [Plan: 2026-06-19-query-transformations-optimization.md](../plans/2026-06-19-query-transformations-optimization.md)

## Specification Context

### Overview

Extends the existing `QueryTransformPipeline` (decompose, HyDE, step-back, keyword
expansion, filter construction) with latency optimizations, net-new techniques, and
offline tooling — following the same layered-wrapper pattern as the Retrieval and
Reranking optimization series.

The base pipeline is the unchanged **leaf**: wrappers compose on top of it, net-new
techniques attach as parallel tools, and a learned router selects transforms per query.
Every addition is gated by a `QT_*` env var defaulting to off. With all flags unset,
behaviour is byte-identical to today.

### Architecture

All wrappers share the leaf's interface so `RetrievalService` consumes any layer
transparently:

`AsyncQueryTransformPipeline` additionally exposes an `async` variant for async callers.

`RetrievalService.from_env()` composes the active layers in order
(`Routed(Cached(Async(leaf)))`), skipping any layer whose flag is unset. When no
`QT_*` flag is set, `QueryTransformPipeline.from_env()` already returns `None` and the
service runs the single-query path unchanged.

---

### Testing

Unit tests per component with a stub LLM (no network):
- **async**: parallelism (calls dispatched concurrently), per-transform timeout/degrade,
  bundle assembled from survivors.
- **cache**: hit/miss, key includes config signature, TTL/serialization round-trip.
- **multi-query**: parse numbered/bulleted output, N respected, `[]` on LLM failure.
- **fusion**: `weighted_rrf_fuse` ordering vs unweighted; `dedup_variants` drops
  near-duplicates and keeps the original.
- **router**: heuristic fallback path with no artifact; predicted config shape;
  serialized-artifact load path with a tiny fixture model.
- **benchmark**: runs on a tiny fixture corpus, produces ranked rows; `--qt-slo-ms`

…

### Out of scope

- Replacing the leaf `QueryTransformPipeline` or `QueryEnhancer` (they continue to work).
- HTTP endpoint for query transformation; streaming transformed results.
- Online / RL training of the router (offline supervised only).
- Routing to different HTTP retrieval servers (only transform-set routing).
- New retrieval backends or reranker changes.

## Implementation Plan Context

### Global Constraints

- Every new behaviour is gated by a `QT_*` env var that defaults to **off**. With all `QT_*` unset, `RetrievalService.from_env()` must produce `pipeline is None` and search behaviour must be byte-identical to today.
- Wrappers share the leaf interface: `transform(query, filters=None, *, config_override=None) -> TransformedQueryBundle` and a `max_variants` property and a `base_config` property.
- Every transformer is fallback-safe: an LLM failure or timeout in one transform degrades that field to its empty/None default; the bundle is still returned. Never raise out of `transform()`.

…

### Task 1: Refactor leaf to job-based transform

Make `QueryTransformPipeline` expose its transform orchestration as reusable jobs so wrappers can run them in parallel, and accept a per-query `config_override`. Behaviour-preserving — guarded by a regression test.

**Files:**
- Modify: `src/context/query_transform.py`
- Test: `tests/unit/test_query_transform.py`

**Interfaces:**
- Consumes: `QueryEnhancer` (`decompose`, `hyde`, `step_back`), `expand_keywords`, `QueryConstructor.extract_filters`.
- Produces:
  - `QueryTransformPipeline._build_jobs(query: str, config: QueryTransformConfig) -> dict[str, Callable[[], object]]`

…

### Task 2: AsyncQueryTransformPipeline

Run the leaf's transform jobs concurrently with a per-transform timeout.

**Files:**
- Create: `src/internal/retrieval/async_query_transform.py`
- Test: `tests/unit/retrieval/test_async_query_transform.py`

**Interfaces:**
- Consumes: leaf `_build_jobs`, `_assemble`, `base_config`, `max_variants`.
- Produces: `AsyncQueryTransformPipeline(base, *, timeout_ms=400, max_workers=5)` with `transform(query, filters=None, *, config_override=None)`, `max_variants`, `base_config`, and classmethod `from_env(base)`.

- [ ] **Step 1: Write the failing test**

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/retrieval/test_async_query_transform.py -v`

…

### Task 3: CachedQueryTransformPipeline

Cache the computed bundle in Redis, keyed by query + config signature.

**Files:**
- Create: `src/internal/retrieval/cached_query_transform.py`
- Test: `tests/unit/retrieval/test_cached_query_transform.py`

**Interfaces:**
- Consumes: base pipeline `transform`, `max_variants`, `base_config`; `config_signature`; `TransformedQueryBundle`.
- Produces: `CachedQueryTransformPipeline(base, redis_client=None, *, ttl_seconds=600)` with `transform(...)`, `max_variants`, `base_config`, `stats()`, classmethod `from_env(base)`.

- [ ] **Step 1: Write the failing test**

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/retrieval/test_cached_query_transform.py -v`

…

### Final verification

- [ ] **Run the full suite**

Run: `pytest -q`
Expected: all pass.

- [ ] **Lint**

Run: `ruff check . --fix && ruff format .`
Expected: clean.

- [ ] **Regression guarantee (manual)**

Confirm with `QT_*` unset that `build_query_transform_pipeline_from_env(...)` returns `None` (covered by `test_returns_none_when_all_flags_unset`) and `RetrievalService` runs the single-query path.

---

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
