# Generated Context Pack

# Reranking Optimization

## Sources

- [Specification: 2026-06-19-reranking-optimization-design.md](../specs/2026-06-19-reranking-optimization-design.md)
- [Plan: 2026-06-19-reranking-optimization.md](../plans/2026-06-19-reranking-optimization.md)

## Specification Context

### Overview

Extends the existing `Reranker` class (BGE local + Cohere remote) with latency optimizations and quality improvements through layered wrapper composition. The `Reranker` leaf is unchanged; wrappers compose on top of it.

### Architecture

```
RetrievalService.search()
  └── TwoStageReranker (M7)          ← quality: pre-filter → heavy scorer
        └── AsyncReranker (M5)        ← latency: thread-offloaded + timeout
              └── CachedReranker (M5) ← latency: Redis score cache
                    └── Reranker      ← existing leaf (unchanged)

Parallel tools (no wrapping):
  PassageTruncator (M5)    — trims passages before any scorer call
  ONNXReranker (M6)        — drop-in Reranker replacement for ONNX runtime
  RerankerBenchmark (M8)   — offline model × config grid search CLI
  Cohere v3 adapter (M8)   — document format update inside Reranker
```

All wrappers share the same interface as `Reranker`:

```python
def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
    ...
```

`AsyncReranker` additionally exposes an `async` variant used by `RetrievalService`.

### Testing Strategy

Each wrapper tested in isolation with mocked inner reranker:

- `test_async_reranker.py` — timeout fires correctly, thread offload returns same results as sync
- `test_cached_reranker.py` — cache hit skips scorer, key includes sorted doc IDs, TTL respected, stats tracking
- `test_passage_truncator.py` — truncation at exact boundary, zero-length, env factory
- `test_onnx_reranker.py` — skipped when `optimum` absent (`pytest.importorskip`), interface parity with `Reranker`
- `test_two_stage_reranker.py` — fast scorer called with all N, heavy scorer called with top M only, over-fetch multiplier applied correctly
- `test_reranker_benchmark.py` — runs against mock reranker, output JSONL has correct fields
- `test_eval_metrics.py` — append `map_at_k` and `reranker_improvement_ratio` tests
- `test_eval_runner.py` — `--slo-ms` exits non-zero on violation, `--compare-baseline` computes ratio

Integration tests (skipped without Redis): `CachedReranker` round-trip against real Redis instance.

---

## Implementation Plan Context

### Global Constraints

- All new files: `from __future__ import annotations` first line, then stdlib, then third-party, then local imports (ruff-enforced)
- All tests: `from src.internal.retrieval.backends.base import RetrievalResult` for the dataclass
- Helper: `def _result(doc_id, score=1.0)` factory in every test file
- `RetrievalResult(doc_id, title, text, url, score)` — all five fields required
- No `import optimum` at module level — lazy inside methods, skip with `pytest.importorskip` in tests
- Run `pytest tests/unit/retrieval/ -v` after each task; all must pass
- Branch: `feat/reranking-optimization` (create before Task 1; never commit to main)
- Commit after every task

---

### Task 1: PassageTruncator + Reranker integration

**Files:**
- Create: `src/internal/retrieval/passage_truncator.py`
- Modify: `src/internal/retrieval/reranker.py` (add truncation call in `_rerank_local`)
- Test: `tests/unit/retrieval/test_passage_truncator.py`

**Interfaces:**
- Produces: `PassageTruncator(max_tokens=512)`, `PassageTruncator.truncate(text: str) -> str`, `PassageTruncator.from_env() -> PassageTruncator`

- [ ] **Step 1: Write the failing tests**

```python

### tests/unit/retrieval/test_passage_truncator.py

from __future__ import annotations
import os
import pytest
from src.internal.retrieval.passage_truncator import PassageTruncator


def test_truncate_below_limit_unchanged():
    t = PassageTruncator(max_tokens=10)
    assert t.truncate("hello world") == "hello world"


def test_truncate_above_limit():
    t = PassageTruncator(max_tokens=3)
    result = t.truncate("one two three four five")
    assert result == "one two three"


def test_truncate_exactly_at_limit():
    t = PassageTruncator(max_tokens=3)
    assert t.truncate("a b c") == "a b c"


def test_truncate_zero_disabled():
    t = PassageTruncator(max_tokens=0)
    long = " ".join(str(i) for i in range(1000))
    assert t.truncate(long) == long


def test_truncate_empty_string():
    t = PassageTruncator(max_tokens=5)
    assert t.truncate("") == ""


def test_from_env_reads_max_tokens(monkeypatch):
    monkeypatch.setenv("RERANKER_MAX_TOKENS", "100")
    t = PassageTruncator.from_env()
    assert t._max == 100


def test_from_env_default(monkeypatch):
    monkeypatch.delenv("RERANKER_MAX_TOKENS", raising=False)
    t = PassageTruncator.from_env()
    assert t._max == 512
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/unit/retrieval/test_passage_truncator.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.internal.retrieval.passage_truncator'`

- [ ] **Step 3: Implement PassageTruncator**

```python

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

```python

### tests/unit/retrieval/test_async_reranker.py

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from src.internal.retrieval.async_reranker import AsyncReranker, RerankerTimeoutError
from src.internal.retrieval.backends.base import RetrievalResult


def _result(doc_id: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(doc_id=doc_id, title="t", text="c", url=None, score=score)


def _make_base(return_val=None):
    base = MagicMock()
    base.rerank.return_value = return_val or [_result("d1")]
    return base


def test_sync_rerank_returns_results():
    base = _make_base([_result("d1"), _result("d2")])
    ar = AsyncReranker(base, timeout_ms=1000)
    results = ar.rerank("query", [_result("d1"), _result("d2")], top_k=2)
    assert [r.doc_id for r in results] == ["d1", "d2"]


def test_sync_rerank_delegates_to_base():
    base = _make_base()
    ar = AsyncReranker(base, timeout_ms=1000)
    ar.rerank("q", [_result("x")], top_k=1)
    base.rerank.assert_called_once_with("q", [_result("x")], 1)


def test_sync_rerank_timeout_raises():
    import time as _time

    base = MagicMock()

    def slow(*_):
        _time.sleep(0.3)
        return [_result("d1")]

    base.rerank.side_effect = slow
    ar = AsyncReranker(base, timeout_ms=50)
    with pytest.raises(RerankerTimeoutError):
        ar.rerank("q", [_result("d1")], top_k=1)


def test_async_rerank_returns_results():
    base = _make_base([_result("d1")])
    ar = AsyncReranker(base, timeout_ms=1000)
    results = asyncio.run(ar.arerank("q", [_result("d1")], top_k=1))

_[Section compacted.]_

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

```python

### tests/unit/retrieval/test_cached_reranker.py

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.cached_reranker import CachedReranker


def _result(doc_id: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(doc_id=doc_id, title="t", text="c", url=None, score=score)


def _make_redis():
    store: dict = {}
    redis = MagicMock()
    redis.get.side_effect = lambda k: store.get(k)
    redis.setex.side_effect = lambda k, ttl, v: store.update({k: v})
    return redis


def test_cache_miss_calls_base():
    base = MagicMock()
    base.rerank.return_value = [_result("d1")]
    cr = CachedReranker(base, _make_redis(), ttl_seconds=60)
    cr.rerank("q", [_result("d1")], top_k=1)
    base.rerank.assert_called_once()


def test_cache_hit_skips_base():
    base = MagicMock()
    base.rerank.return_value = [_result("d1")]
    redis = _make_redis()
    cr = CachedReranker(base, redis, ttl_seconds=60)
    # Populate cache on first call
    cr.rerank("q", [_result("d1")], top_k=1)
    # Second call — base should NOT be called again
    result = cr.rerank("q", [_result("d1")], top_k=1)
    assert base.rerank.call_count == 1
    assert result[0].doc_id == "d1"


def test_cache_key_includes_doc_order_invariant():
    base = MagicMock()
    base.rerank.return_value = [_result("d1")]
    redis = _make_redis()
    cr = CachedReranker(base, redis, ttl_seconds=60)
    cr.rerank("q", [_result("a"), _result("b")], top_k=1)
    # Reversed order of inputs — same doc_ids, should hit cache

_[Section compacted.]_

### Task 4: M5 service.py wiring

**Files:**
- Modify: `src/internal/retrieval/service.py`
- Test: `tests/unit/retrieval/test_service.py` (append tests)

**Interfaces:**
- Consumes: `AsyncReranker.from_env(base)`, `CachedReranker.from_env(base)` from Tasks 2–3
- Produces: `RetrievalService.from_env()` builds `CachedReranker(AsyncReranker(base))` when `RERANKER_ASYNC=true`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/retrieval/test_service.py`. First read the file to see existing structure, then add:

```python

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
