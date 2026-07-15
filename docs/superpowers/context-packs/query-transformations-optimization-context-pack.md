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

## Implementation Plan Context

### Global Constraints

- Every new behaviour is gated by a `QT_*` env var that defaults to **off**. With all `QT_*` unset, `RetrievalService.from_env()` must produce `pipeline is None` and search behaviour must be byte-identical to today.
- Wrappers share the leaf interface: `transform(query, filters=None, *, config_override=None) -> TransformedQueryBundle` and a `max_variants` property and a `base_config` property.
- Every transformer is fallback-safe: an LLM failure or timeout in one transform degrades that field to its empty/None default; the bundle is still returned. Never raise out of `transform()`.
- Match existing patterns: mirror `async_reranker.py`, `cached_reranker.py`, `reranker_factory.py`, `reranker_benchmark.py`. Use `MagicMock`/fake LLMs in tests — no network.
- `RetrievalResult` fields are `doc_id, title, text, url, score, metadata` (from `src/internal/retrieval/backends/base.py`).
- The LLM interface is `LLMClient.complete(messages: list[ChatMessage]) -> LLMResponse | str` (`src/context/models.py`); `LLMResponse.text` holds the string.

---

### Task 1: Refactor leaf to job-based transform

Make `QueryTransformPipeline` expose its transform orchestration as reusable jobs so wrappers can run them in parallel, and accept a per-query `config_override`. Behaviour-preserving — guarded by a regression test.

**Files:**
- Modify: `src/context/query_transform.py`
- Test: `tests/unit/test_query_transform.py`

**Interfaces:**
- Consumes: `QueryEnhancer` (`decompose`, `hyde`, `step_back`), `expand_keywords`, `QueryConstructor.extract_filters`.
- Produces:
  - `QueryTransformPipeline._build_jobs(query: str, config: QueryTransformConfig) -> dict[str, Callable[[], object]]`
  - `QueryTransformPipeline._assemble(query: str, results: dict, caller_filters: dict | None) -> TransformedQueryBundle`
  - `QueryTransformPipeline.transform(query, filters=None, *, config_override=None) -> TransformedQueryBundle`
  - `QueryTransformPipeline.base_config -> QueryTransformConfig` (property)
  - `config_signature(config: QueryTransformConfig) -> str` (module function)

- [ ] **Step 1: Write the failing test**

```python

### tests/unit/test_query_transform.py — append

from unittest.mock import MagicMock
from src.context.query_transform import (
    QueryTransformConfig,
    QueryTransformPipeline,
    config_signature,
)


def _fake_llm(text: str = "") -> MagicMock:
    llm = MagicMock()
    llm.complete.return_value = type("R", (), {"text": text})()
    return llm


def test_transform_config_override_runs_only_overridden_transforms():
    # Leaf built with everything OFF; override turns step_back ON.
    pipe = QueryTransformPipeline(QueryTransformConfig(), _fake_llm("broader query"))
    override = QueryTransformConfig(step_back=True)
    bundle = pipe.transform("specific q", config_override=override)
    assert bundle.step_back == "broader query"
    assert bundle.sub_queries == []  # decompose stayed off


def test_base_config_exposed():
    cfg = QueryTransformConfig(hyde=True)
    pipe = QueryTransformPipeline(cfg, _fake_llm())
    assert pipe.base_config is cfg


def test_config_signature_changes_with_flags():
    a = config_signature(QueryTransformConfig(hyde=True))
    b = config_signature(QueryTransformConfig(hyde=False))
    assert a != b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_query_transform.py::test_transform_config_override_runs_only_overridden_transforms -v`
Expected: FAIL — `transform()` has no `config_override` kwarg / `config_signature` not importable.

- [ ] **Step 3: Refactor the implementation**

Replace the body of `QueryTransformPipeline` and add the module function. In `src/context/query_transform.py`:

```python
from typing import TYPE_CHECKING, Callable

### Task 2: AsyncQueryTransformPipeline

Run the leaf's transform jobs concurrently with a per-transform timeout.

**Files:**
- Create: `src/internal/retrieval/async_query_transform.py`
- Test: `tests/unit/retrieval/test_async_query_transform.py`

**Interfaces:**
- Consumes: leaf `_build_jobs`, `_assemble`, `base_config`, `max_variants`.
- Produces: `AsyncQueryTransformPipeline(base, *, timeout_ms=400, max_workers=5)` with `transform(query, filters=None, *, config_override=None)`, `max_variants`, `base_config`, and classmethod `from_env(base)`.

- [ ] **Step 1: Write the failing test**

```python

### tests/unit/retrieval/test_async_query_transform.py

from __future__ import annotations

import time

from src.context.query_transform import QueryTransformConfig, QueryTransformPipeline
from src.internal.retrieval.async_query_transform import AsyncQueryTransformPipeline
from unittest.mock import MagicMock


def _fake_llm(text: str = "x") -> MagicMock:
    llm = MagicMock()
    llm.complete.return_value = type("R", (), {"text": text})()
    return llm


def test_slow_transform_times_out_and_degrades():
    leaf = QueryTransformPipeline(
        QueryTransformConfig(step_back=True), _fake_llm("broad")
    )
    # Make step_back sleep past the timeout.
    leaf._enhancer.step_back = lambda q: (time.sleep(0.5) or "broad")  # type: ignore
    pipe = AsyncQueryTransformPipeline(leaf, timeout_ms=50, max_workers=2)
    bundle = pipe.transform("q")
    assert bundle.step_back is None  # degraded, no raise
    assert bundle.original == "q"


def test_runs_transforms_and_assembles():
    leaf = QueryTransformPipeline(
        QueryTransformConfig(step_back=True), _fake_llm("broad")
    )
    pipe = AsyncQueryTransformPipeline(leaf, timeout_ms=2000)
    bundle = pipe.transform("q")
    assert bundle.step_back == "broad"


def test_max_variants_and_base_config_delegate():
    leaf = QueryTransformPipeline(QueryTransformConfig(max_variants=7), _fake_llm())
    pipe = AsyncQueryTransformPipeline(leaf)
    assert pipe.max_variants == 7
    assert pipe.base_config is leaf.base_config
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/retrieval/test_async_query_transform.py -v`
Expected: FAIL — module does not exist.

_[Section compacted.]_

### Task 3: CachedQueryTransformPipeline

Cache the computed bundle in Redis, keyed by query + config signature.

**Files:**
- Create: `src/internal/retrieval/cached_query_transform.py`
- Test: `tests/unit/retrieval/test_cached_query_transform.py`

**Interfaces:**
- Consumes: base pipeline `transform`, `max_variants`, `base_config`; `config_signature`; `TransformedQueryBundle`.
- Produces: `CachedQueryTransformPipeline(base, redis_client=None, *, ttl_seconds=600)` with `transform(...)`, `max_variants`, `base_config`, `stats()`, classmethod `from_env(base)`.

- [ ] **Step 1: Write the failing test**

```python

### tests/unit/retrieval/test_cached_query_transform.py

from __future__ import annotations

import json
from unittest.mock import MagicMock

from src.context.query_transform import QueryTransformConfig, QueryTransformPipeline
from src.internal.retrieval.cached_query_transform import CachedQueryTransformPipeline


class FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, k):
        return self.store.get(k)

    def setex(self, k, ttl, v):
        self.store[k] = v


def _fake_llm(text="broad"):
    llm = MagicMock()
    llm.complete.return_value = type("R", (), {"text": text})()
    return llm


def test_second_call_is_cache_hit():
    leaf = QueryTransformPipeline(
        QueryTransformConfig(step_back=True), _fake_llm("broad")
    )
    redis = FakeRedis()
    pipe = CachedQueryTransformPipeline(leaf, redis)
    b1 = pipe.transform("q")
    b2 = pipe.transform("q")
    assert b1.step_back == b2.step_back == "broad"
    assert pipe.stats() == {"hits": 1, "misses": 1, "hit_rate": 0.5}


def test_disabled_without_redis_passes_through():
    leaf = QueryTransformPipeline(QueryTransformConfig(step_back=True), _fake_llm())
    pipe = CachedQueryTransformPipeline(leaf, None)
    assert pipe.transform("q").original == "q"
    assert pipe.stats()["hits"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/retrieval/test_cached_query_transform.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Write the implementation**

```python

### Task 4: Factory + RetrievalService wiring

Compose the wrapper chain from env and have `RetrievalService` use it. Verify the all-unset regression guarantee.

**Files:**
- Create: `src/internal/retrieval/query_transform_factory.py`
- Modify: `src/internal/retrieval/service.py:133-135` (the `from_env` block that builds the pipeline)
- Test: `tests/unit/retrieval/test_query_transform_factory.py`

**Interfaces:**
- Consumes: `QueryTransformPipeline.from_env`, `AsyncQueryTransformPipeline.from_env`, `CachedQueryTransformPipeline.from_env`.
- Produces: `build_query_transform_pipeline_from_env(llm) -> object | None`.

- [ ] **Step 1: Write the failing test**

```python

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
