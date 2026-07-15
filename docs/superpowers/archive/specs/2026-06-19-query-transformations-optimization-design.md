# Query Transformations Optimization Design

**Date:** 2026-06-19
**Status:** Approved

## Overview

Extends the existing `QueryTransformPipeline` (decompose, HyDE, step-back, keyword
expansion, filter construction) with latency optimizations, net-new techniques, and
offline tooling — following the same layered-wrapper pattern as the Retrieval and
Reranking optimization series.

The base pipeline is the unchanged **leaf**: wrappers compose on top of it, net-new
techniques attach as parallel tools, and a learned router selects transforms per query.
Every addition is gated by a `QT_*` env var defaulting to off. With all flags unset,
behaviour is byte-identical to today.

### Why these milestones

The base pipeline already parallelizes per-variant **retrieval** (`ThreadPoolExecutor`
in `RetrievalService.search`) and already fuses with `rrf_fuse`. The remaining
bottleneck is `QueryTransformPipeline.transform()` itself, which runs up to five
**sequential** LLM calls (decompose → hyde → step_back → keywords → construct_filters)
on every request. M5 attacks that directly; M6–M8 add quality and tooling.

## Architecture

```
RetrievalService.search()
  └── RoutedQueryTransformPipeline (M7)        ← per-query: learned router picks transforms
        └── CachedQueryTransformPipeline (M5)  ← Redis bundle cache
              └── AsyncQueryTransformPipeline (M5) ← parallel transform LLM calls + timeout
                    └── QueryTransformPipeline  ← existing leaf (unchanged)

Parallel tools (not wrappers):
  MultiQueryGenerator (M6)              — N paraphrases in one LLM call (true Multi-Query)
  weighted_rrf_fuse + semantic dedup (M6) — extends src/internal/retrieval/fusion.py
  QueryRouter + train_query_router.py (M7) — sklearn artifact, heuristic fallback
  QueryTransformBenchmark (M8)          — offline grid: technique combos × dataset
  Richer QueryConstructor (M8)          — operators/ranges in NL → filter
```

All wrappers share the leaf's interface so `RetrievalService` consumes any layer
transparently:

```python
def transform(self, query: str, filters: dict | None = None) -> TransformedQueryBundle: ...

@property
def max_variants(self) -> int: ...
```

`AsyncQueryTransformPipeline` additionally exposes an `async` variant for async callers.

`RetrievalService.from_env()` composes the active layers in order
(`Routed(Cached(Async(leaf)))`), skipping any layer whose flag is unset. When no
`QT_*` flag is set, `QueryTransformPipeline.from_env()` already returns `None` and the
service runs the single-query path unchanged.

---

## Milestones

### M5 — Latency Foundations

**`AsyncQueryTransformPipeline`** (`src/internal/retrieval/async_query_transform.py`)

Wraps any pipeline exposing `transform()` / `max_variants`. Runs the independent
transform calls (decompose, hyde, step_back, keywords, construct_filters) concurrently
in a `ThreadPoolExecutor` instead of sequentially.

```python
AsyncQueryTransformPipeline(base_pipeline, *, timeout_ms: int = 400, max_workers: int = 5)
```

- `transform(query, filters) -> TransformedQueryBundle` — sync entry point; submits each
  enabled transform as a future, gathers results.
- `async atransform(query, filters)` — awaitable variant for async callers.
- Per-transform timeout: a transform exceeding `timeout_ms` (or raising) is dropped to
  its empty/None default and the bundle is still assembled from the rest. No request
  fails because one transform was slow.
- Because the leaf only exposes a single `transform()`, M5 introduces a small internal
  seam: the async wrapper calls the leaf's underlying transformer methods
  (`_enhancer.decompose`, `_enhancer.hyde`, `_enhancer.step_back`, `expand_keywords`,
  `_constructor.extract_filters`) as independent units. These are read off the leaf's
  config so disabled transforms are never scheduled.
- `from_env(base_pipeline)` reads `QT_TRANSFORM_TIMEOUT_MS`, `QT_MAX_WORKERS`.

**`CachedQueryTransformPipeline`** (`src/internal/retrieval/cached_query_transform.py`)

Wraps any pipeline. Redis-backed cache of the computed bundle.

- Cache key: `"qt:" + sha256(f"{query}|{config_signature}").hexdigest()[:20]`, where
  `config_signature` captures which transforms are enabled + `max_variants` so a config
  change invalidates stale entries.
- On hit: deserialize and return the `TransformedQueryBundle` without calling the base.
- On miss: call through, serialize (JSON, mirroring `ResultCache`), write with
  `ttl_seconds`.
- `stats() -> dict` — hits, misses, hit_rate.
- `from_env(base_pipeline)` reads `QT_CACHE_REDIS_URL`, `QT_CACHE_TTL_SECONDS`; returns
  `base_pipeline` unchanged when no URL is set.

**M5 environment variables**

| Variable | Default | Description |
|---|---|---|
| `QT_ASYNC` | `false` | Run transform LLM calls in parallel |
| `QT_TRANSFORM_TIMEOUT_MS` | `400` | Per-transform timeout (degrade on exceed) |
| `QT_MAX_WORKERS` | `5` | Thread pool size for transforms |
| `QT_CACHE_REDIS_URL` | _(unset)_ | Enable Redis bundle cache |
| `QT_CACHE_TTL_SECONDS` | `600` | Cache TTL |

---

### M6 — Multi-Query Generation + Smarter Fusion

**`MultiQueryGenerator`** (`src/internal/retrieval/multi_query.py`)

True Multi-Query Retrieval: one LLM call produces N paraphrased reformulations of the
query (semantically equivalent, lexically diverse) — distinct from decompose's
sub-questions. Parse-tolerant (numbered/bulleted lines), fallback-safe (returns `[]` on
LLM failure).

```python
class MultiQueryGenerator:
    def __init__(self, llm, *, n: int = 3): ...
    def generate(self, query: str) -> list[str]: ...
    @staticmethod
    def from_env(llm) -> "MultiQueryGenerator | None": ...  # QT_MULTI_QUERY, QT_MULTI_QUERY_N
```

Integration: a new `multi_query: list[str]` field on `TransformedQueryBundle`, surfaced
in `retrieval_variants()` ordering (after `sub_queries`, before `hyde_text`). The leaf's
`transform()` gains a `multi_query` branch behind the new config flag. `max_variants`
still caps the total, original still always last.

**Weighted RRF + semantic dedup** (`src/internal/retrieval/fusion.py`)

- `weighted_rrf_fuse(result_sets: list[list[RetrievalResult]], weights: list[float])` —
  RRF where each result set contributes `weight / (k + rank)`. The original query's
  result set is weighted highest; expansions contribute less. Existing `rrf_fuse` stays
  as the unweighted default.
- `dedup_variants(variants: list[str], embed_fn, threshold: float) -> list[str]` —
  drops near-duplicate variants (cosine ≥ threshold) **before** retrieval to avoid
  spending retrieval calls on redundant reformulations. Applied in `RetrievalService`
  between `retrieval_variants()` and the retrieval fan-out when `QT_SEMANTIC_DEDUP=true`.

`RetrievalService.search` selects `weighted_rrf_fuse` over `rrf_fuse` when
`QT_FUSION_WEIGHTED=true`; weights derive from variant provenance (original vs
expansion). Mode string becomes `...+rag_fusion` as today (no new suffix needed).

**M6 environment variables**

| Variable | Default | Description |
|---|---|---|
| `QT_MULTI_QUERY` | `false` | Generate N paraphrase variants |
| `QT_MULTI_QUERY_N` | `3` | Number of paraphrases |
| `QT_FUSION_WEIGHTED` | `false` | Use weighted RRF (original weighted highest) |
| `QT_SEMANTIC_DEDUP` | `false` | Drop near-duplicate variants before retrieval |
| `QT_SEMANTIC_DEDUP_THRESHOLD` | `0.95` | Cosine cutoff for dedup |

---

### M7 — Learned Query Router

**`QueryRouter`** (`src/internal/retrieval/query_router.py`)

Predicts, per query, which transforms to enable — replacing the static, apply-everything
`QueryTransformConfig`. Cheap queries skip expensive transforms; multi-hop queries get
decompose; filterable queries get construction.

- Loads a serialized scikit-learn classifier from `QT_ROUTER_MODEL_PATH`.
- Features: sentence embedding of the query concatenated with cheap lexical cues
  (token length, presence of question words, multi-clause/conjunction cues, date/number
  cues, capitalized-entity count).
- Output: a `QueryTransformConfig` (per-transform booleans) for this query.
- **Heuristic fallback**: when the artifact is missing or unset, `predict()` uses a
  rule-based mapping (e.g. long multi-clause → decompose; date/number cues →
  construct_filters; short keyword-like → keywords) so serving never hard-depends on a
  trained model.

```python
class QueryRouter:
    def __init__(self, model_path: str | None = None): ...
    def predict(self, query: str) -> QueryTransformConfig: ...
    @staticmethod
    def from_env() -> "QueryRouter | None": ...  # QT_ROUTER, QT_ROUTER_MODEL_PATH
```

**`RoutedQueryTransformPipeline`** (`src/internal/retrieval/routed_query_transform.py`)

Outermost wrapper. On each `transform()`, calls `router.predict(query)` to get a
per-query config, then threads that config down the stack via an **optional
`config_override` parameter** added to each layer's internal transform path
(`transform(query, filters, *, config_override=None)`). The override is additive and
backward-compatible: when `None` (every existing caller), each layer uses its own
constructed config and behaviour is unchanged; when set, the leaf runs exactly the
transforms the router selected for this query. `max_variants` delegates to the base.
Composes over the cached/async layers unchanged — the override flows Router → Cache →
Async → leaf, and the cache key incorporates the routed config signature so different
routings of the same query cache separately.

**`train_query_router.py`** (`src/training/train_query_router.py`)

Offline-only. Reads a labeled dataset (query → ideal transform set), extracts the same
features, fits a multilabel classifier, serializes the artifact to `QT_ROUTER_MODEL_PATH`.
A small seed dataset (derivable from existing eval query sets with weak labels) ships in
`data/` so the script is runnable. Training infra never enters the serving path.

**M7 environment variables**

| Variable | Default | Description |
|---|---|---|
| `QT_ROUTER` | `false` | Route transforms per query |
| `QT_ROUTER_MODEL_PATH` | _(unset)_ | Path to serialized router; falls back to heuristic if unset |

---

### M8 — Benchmark CLI + Eval Integration + Richer Construction

**`QueryTransformBenchmark`** (`src/internal/retrieval/query_transform_benchmark.py`)

Offline grid search over technique combinations × a labeled dataset (mirrors
`RerankerBenchmark`). For each config (e.g. `{decompose}`, `{multi_query}`,
`{multi_query, hyde, weighted_fusion}`, …) reports recall@k / NDCG@k and per-query
transform + end-to-end latency. CLI entry point prints a ranked table and writes JSONL.

```python
class QueryTransformBenchmark:
    def __init__(self, dataset, retrieval_service_factory): ...
    def run(self, configs: list[QueryTransformConfig]) -> list[BenchmarkRow]: ...
```

**`eval_runner` additions** (`src/internal/retrieval/eval_runner.py`)

- Per-query transform latency stored in the output JSONL alongside existing metrics.
- `--qt-slo-ms INT` flag: non-zero exit if P99 transform latency across queries exceeds
  the value.

**Richer `QueryConstructor`** (`src/internal/retrieval/query_constructor.py`)

Extend NL → filter extraction beyond equality to operators and ranges:
- comparison operators (`>=`, `<=`, `>`, `<`) for numeric/date fields
  ("papers after 2023", "rating above 4")
- date ranges ("between 2020 and 2022")
- Emits a structured filter dict the retrieval backends already accept; new operator
  keys are documented and additive (equality extraction unchanged).

**M8 environment variables**

| Variable | Default | Description |
|---|---|---|
| `QT_CONSTRUCT_OPERATORS` | `false` | Extract range/comparison operators, not just equality |

(`QueryTransformBenchmark` and the `--qt-slo-ms` eval flag are CLI/offline; no serving
env var.)

---

## Cross-cutting

### Composition order (RetrievalService.from_env)

```
leaf = QueryTransformPipeline.from_env(llm)      # None if no transform flags set
if leaf and QT_ASYNC:  leaf = AsyncQueryTransformPipeline.from_env(leaf)
if leaf and QT_CACHE_REDIS_URL: leaf = CachedQueryTransformPipeline.from_env(leaf)
if leaf and QT_ROUTER: leaf = RoutedQueryTransformPipeline.from_env(leaf)
pipeline = leaf
```

Router outermost (decides config) → cache (memoizes the routed bundle) → async (parallel
execution of whatever survived routing) → leaf. Each `from_env` returns its input
unchanged when its own flag is unset.

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
  exit-code behaviour.
- **construction**: operator/range extraction for representative phrasings; equality
  path unchanged.

**Regression guarantee:** a test asserting that with all `QT_*` unset,
`RetrievalService.from_env()` produces `pipeline is None` and the single-query search
path is byte-identical to the pre-change behaviour.

### Out of scope

- Replacing the leaf `QueryTransformPipeline` or `QueryEnhancer` (they continue to work).
- HTTP endpoint for query transformation; streaming transformed results.
- Online / RL training of the router (offline supervised only).
- Routing to different HTTP retrieval servers (only transform-set routing).
- New retrieval backends or reranker changes.

## Technique coverage

| Technique | Where |
|---|---|
| Multi-Query Retrieval | M6 `MultiQueryGenerator` |
| RAG-Fusion | M6 `weighted_rrf_fuse` (extends existing `rrf_fuse`) |
| Query Decomposition | existing leaf (now parallelized M5, cached M5) |
| Query Rewriting / Rewrite / step-back | existing leaf (parallelized M5) |
| HyDE | existing leaf (parallelized M5) |
| Query Routing | M7 learned `QueryRouter` + heuristic fallback |
| Query Construction | M8 richer `QueryConstructor` (operators/ranges) |
