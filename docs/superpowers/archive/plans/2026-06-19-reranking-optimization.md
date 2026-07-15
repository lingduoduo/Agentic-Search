# Reranking Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add latency (async thread offload, Redis score cache, passage truncation, ONNX) and quality (two-stage pipeline, over-fetch, model benchmarking) optimizations to the existing `Reranker` via layered wrapper composition.

**Architecture:** `Reranker` leaf is unchanged; `AsyncReranker`, `CachedReranker`, and `TwoStageReranker` are wrappers that share the same `rerank(query, results, top_k)` interface. `RetrievalService.from_env()` composes the chain from env vars; callers never change.

**Tech Stack:** Python stdlib (`concurrent.futures`, `asyncio`, `hashlib`, `json`), Redis (`redis-py`), ONNX (`optimum.onnxruntime` — optional), `pytest`, existing `src.internal.retrieval.*` patterns.

## Global Constraints

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
# tests/unit/retrieval/test_passage_truncator.py
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
# src/internal/retrieval/passage_truncator.py
from __future__ import annotations

import os


class PassageTruncator:
    """Whitespace-token truncation for reranker passages."""

    def __init__(self, max_tokens: int = 512) -> None:
        self._max = max_tokens

    def truncate(self, text: str) -> str:
        if self._max == 0 or not text:
            return text
        tokens = text.split()
        if len(tokens) <= self._max:
            return text
        return " ".join(tokens[: self._max])

    @staticmethod
    def from_env() -> "PassageTruncator":
        return PassageTruncator(
            max_tokens=int(os.environ.get("RERANKER_MAX_TOKENS", "512"))
        )
```

- [ ] **Step 4: Wire truncator into Reranker._rerank_local**

Read `src/internal/retrieval/reranker.py` first. Add `truncator` param to `__init__` and call it in `_rerank_local`:

```python
# In reranker.py — add import at top (after existing imports)
from src.internal.retrieval.passage_truncator import PassageTruncator

# Modify __init__ signature:
def __init__(self, config: RerankerConfig, truncator: PassageTruncator | None = None) -> None:
    config.validate()
    self._config = config
    self._truncator = truncator or PassageTruncator.from_env()
    if config.provider == "local":
        self._local = SentenceTransformerReranker.load(
            config.model,
            batch_size=config.batch_size,
            device=config.device,
        )

# Modify _rerank_local — change the docs list comprehension:
docs = [
    {
        "contents": f"{r.title}\n{self._truncator.truncate(r.text)}",
        "doc_id": r.doc_id,
    }
    for r in results
]
```

- [ ] **Step 5: Verify tests pass**

```
pytest tests/unit/retrieval/test_passage_truncator.py tests/unit/retrieval/test_reranker.py -v
```
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git checkout -b feat/reranking-optimization
git add src/internal/retrieval/passage_truncator.py \
        src/internal/retrieval/reranker.py \
        tests/unit/retrieval/test_passage_truncator.py
git commit -m "feat(reranker): PassageTruncator with RERANKER_MAX_TOKENS env var"
```

---

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
# tests/unit/retrieval/test_async_reranker.py
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
    assert results[0].doc_id == "d1"


def test_async_rerank_timeout_raises():
    import time as _time

    base = MagicMock()

    def slow(*_):
        _time.sleep(0.3)
        return [_result("d1")]

    base.rerank.side_effect = slow
    ar = AsyncReranker(base, timeout_ms=50)
    with pytest.raises(RerankerTimeoutError):
        asyncio.run(ar.arerank("q", [_result("d1")], top_k=1))


def test_from_env_reads_timeout(monkeypatch):
    monkeypatch.setenv("RERANKER_TIMEOUT_MS", "250")
    base = _make_base()
    ar = AsyncReranker.from_env(base)
    assert ar._timeout_ms == 250
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/unit/retrieval/test_async_reranker.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement AsyncReranker**

```python
# src/internal/retrieval/async_reranker.py
from __future__ import annotations

import asyncio
import concurrent.futures
import os

from src.internal.retrieval.backends.base import RetrievalResult


class RerankerTimeoutError(RuntimeError):
    pass


class AsyncReranker:
    """Wraps any reranker, offloads scoring to a thread pool with a timeout."""

    def __init__(
        self,
        base_reranker,
        *,
        timeout_ms: int = 500,
        max_workers: int = 4,
    ) -> None:
        self._base = base_reranker
        self._timeout_ms = timeout_ms
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        """Sync shim: submits to thread pool, blocks with timeout."""
        future = self._executor.submit(self._base.rerank, query, results, top_k)
        try:
            return future.result(timeout=self._timeout_ms / 1000)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise RerankerTimeoutError(
                f"Reranker exceeded {self._timeout_ms}ms timeout"
            )

    async def arerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        """Async entry point: runs scorer in thread pool, awaits with timeout."""
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(self._executor, self._base.rerank, query, results, top_k)
        try:
            return await asyncio.wait_for(future, timeout=self._timeout_ms / 1000)
        except asyncio.TimeoutError:
            raise RerankerTimeoutError(
                f"Reranker exceeded {self._timeout_ms}ms timeout"
            )

    @classmethod
    def from_env(cls, base_reranker) -> "AsyncReranker":
        return cls(
            base_reranker,
            timeout_ms=int(os.environ.get("RERANKER_TIMEOUT_MS", "500")),
            max_workers=int(os.environ.get("RERANKER_MAX_WORKERS", "4")),
        )
```

- [ ] **Step 4: Verify tests pass**

```
pytest tests/unit/retrieval/test_async_reranker.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/async_reranker.py \
        tests/unit/retrieval/test_async_reranker.py
git commit -m "feat(reranker): AsyncReranker with thread-pool offload and configurable timeout"
```

---

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
# tests/unit/retrieval/test_cached_reranker.py
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
    result = cr.rerank("q", [_result("b"), _result("a")], top_k=1)
    assert base.rerank.call_count == 1


def test_cache_key_normalises_query_case():
    base = MagicMock()
    base.rerank.return_value = [_result("d1")]
    redis = _make_redis()
    cr = CachedReranker(base, redis, ttl_seconds=60)
    cr.rerank("Hello World", [_result("d1")], top_k=1)
    cr.rerank("hello world", [_result("d1")], top_k=1)
    assert base.rerank.call_count == 1


def test_none_redis_disables_cache():
    base = MagicMock()
    base.rerank.return_value = [_result("d1")]
    cr = CachedReranker(base, None, ttl_seconds=60)
    cr.rerank("q", [_result("d1")], top_k=1)
    cr.rerank("q", [_result("d1")], top_k=1)
    assert base.rerank.call_count == 2


def test_stats_tracks_hits_and_misses():
    base = MagicMock()
    base.rerank.return_value = [_result("d1")]
    redis = _make_redis()
    cr = CachedReranker(base, redis, ttl_seconds=60)
    cr.rerank("q", [_result("d1")], top_k=1)  # miss
    cr.rerank("q", [_result("d1")], top_k=1)  # hit
    stats = cr.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(0.5)


def test_from_env_returns_base_when_no_redis_url(monkeypatch):
    monkeypatch.delenv("RERANKER_CACHE_REDIS_URL", raising=False)
    base = MagicMock()
    result = CachedReranker.from_env(base)
    assert result is base
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/unit/retrieval/test_cached_reranker.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement CachedReranker**

```python
# src/internal/retrieval/cached_reranker.py
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from typing import Any

from src.internal.retrieval.backends.base import RetrievalResult

logger = logging.getLogger(__name__)


def _cache_key(query: str, doc_ids: list[str]) -> str:
    canonical = query.lower().strip()
    sorted_ids = json.dumps(sorted(doc_ids))
    raw = f"{canonical}:{sorted_ids}"
    return "rrk:" + hashlib.sha256(raw.encode()).hexdigest()[:20]


class CachedReranker:
    """Redis-backed cache for reranker scores. Cache key: query + sorted doc_ids."""

    def __init__(
        self,
        base_reranker,
        redis_client: Any | None = None,
        *,
        ttl_seconds: int = 300,
    ) -> None:
        self._base = base_reranker
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        doc_ids = [r.doc_id for r in results]
        key = _cache_key(query, doc_ids)

        if self._redis is not None:
            raw = self._redis.get(key)
            if raw is not None:
                self._hits += 1
                return [RetrievalResult(**row) for row in json.loads(raw)]

        self._misses += 1
        reranked = self._base.rerank(query, results, top_k)

        if self._redis is not None:
            payload = json.dumps([dataclasses.asdict(r) for r in reranked])
            self._redis.setex(key, self._ttl, payload)

        return reranked

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total else 0.0,
        }

    @classmethod
    def from_env(cls, base_reranker):
        """Returns base_reranker unchanged if RERANKER_CACHE_REDIS_URL is not set."""
        redis_url = os.environ.get("RERANKER_CACHE_REDIS_URL")
        if not redis_url:
            return base_reranker
        try:
            import redis as _redis

            rc = _redis.from_url(redis_url)
            return cls(
                base_reranker,
                rc,
                ttl_seconds=int(os.environ.get("RERANKER_CACHE_TTL_SECONDS", "300")),
            )
        except Exception as exc:
            logger.warning("Reranker cache disabled: %s", exc)
            return base_reranker
```

- [ ] **Step 4: Verify tests pass**

```
pytest tests/unit/retrieval/test_cached_reranker.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/cached_reranker.py \
        tests/unit/retrieval/test_cached_reranker.py
git commit -m "feat(reranker): CachedReranker with Redis score cache and hit/miss stats"
```

---

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
# Append to tests/unit/retrieval/test_service.py

def test_from_env_builds_async_reranker_chain(monkeypatch):
    """RERANKER_ASYNC=true wraps base reranker in AsyncReranker."""
    monkeypatch.setenv("RERANKER_ASYNC", "true")
    monkeypatch.setenv("RERANKER_PROVIDER", "local")
    monkeypatch.setenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

    from src.internal.retrieval.async_reranker import AsyncReranker
    from src.internal.retrieval.cached_reranker import CachedReranker

    with (
        patch("src.internal.retrieval.service._build_backend", return_value=MagicMock()),
        patch("src.internal.retrieval.reranker.SentenceTransformerReranker.load", return_value=MagicMock()),
    ):
        svc = RetrievalService.from_env()

    assert isinstance(svc._reranker, (AsyncReranker, CachedReranker))


def test_from_env_no_async_flag_leaves_base_reranker(monkeypatch):
    """Without RERANKER_ASYNC, reranker is the bare Reranker instance."""
    monkeypatch.delenv("RERANKER_ASYNC", raising=False)
    monkeypatch.delenv("RERANKER_PROVIDER", raising=False)

    with patch("src.internal.retrieval.service._build_backend", return_value=MagicMock()):
        svc = RetrievalService.from_env()

    assert svc._reranker is None
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/unit/retrieval/test_service.py -v -k "async_reranker"
```
Expected: FAIL (new tests not passing yet)

- [ ] **Step 3: Update service.py from_env to build async reranker chain**

Read `src/internal/retrieval/service.py` first. Then modify the `from_env()` method. Replace the `reranker=Reranker.from_env()` line with:

```python
    # In RetrievalService.from_env(), replace:
    #     reranker=Reranker.from_env(),
    # with:

    base_reranker = Reranker.from_env()
    if base_reranker is not None and os.environ.get("RERANKER_ASYNC", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        from src.internal.retrieval.async_reranker import AsyncReranker
        from src.internal.retrieval.cached_reranker import CachedReranker

        base_reranker = AsyncReranker.from_env(base_reranker)
        base_reranker = CachedReranker.from_env(base_reranker)

    return cls(
        _build_backend(),
        reranker=base_reranker,
        pipeline=pipeline,
        optimizer=optimizer,
        result_cache=result_cache,
    )
```

- [ ] **Step 4: Verify tests pass**

```
pytest tests/unit/retrieval/test_service.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/service.py \
        tests/unit/retrieval/test_service.py
git commit -m "feat(service): wire AsyncReranker+CachedReranker chain via RERANKER_ASYNC"
```

---

### Task 5: ONNXReranker + M6 eval_runner additions

**Files:**
- Create: `src/internal/retrieval/onnx_reranker.py`
- Modify: `src/internal/retrieval/eval_runner.py` (add `--slo-ms`, mean latency, latency table)
- Test: `tests/unit/retrieval/test_onnx_reranker.py`
- Test: `tests/unit/retrieval/test_eval_runner.py` (append tests)

**Interfaces:**
- Consumes: `Reranker` interface; `_percentile()` from eval_runner
- Produces:
  - `ONNXReranker(model_name, *, device="cpu")` — lazy `optimum` import
  - `ONNXReranker.rerank(query, results, top_k) -> list[RetrievalResult]`
  - `ONNXReranker.from_env() -> ONNXReranker | Reranker`
  - `run_eval(... , slo_ms=None)` — raises `SLOViolationError` if p99 > slo_ms
  - `latency_ms` dict gains `"mean"` field

- [ ] **Step 1: Write failing tests for ONNXReranker**

```python
# tests/unit/retrieval/test_onnx_reranker.py
from __future__ import annotations

import pytest

optimum = pytest.importorskip("optimum")  # noqa: E402

from unittest.mock import MagicMock, patch  # noqa: E402

from src.internal.retrieval.backends.base import RetrievalResult  # noqa: E402
from src.internal.retrieval.onnx_reranker import ONNXReranker  # noqa: E402


def _result(doc_id: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(doc_id=doc_id, title="t", text="c", url=None, score=score)


def _make_onnx_reranker():
    with (
        patch("optimum.onnxruntime.ORTModelForSequenceClassification.from_pretrained", return_value=MagicMock()),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()),
    ):
        return ONNXReranker("BAAI/bge-reranker-base")


def test_onnx_reranker_rerank_returns_results():
    reranker = _make_onnx_reranker()
    # Patch the model to return fake logits
    import torch
    reranker._model.return_value.logits = torch.tensor([[0.8], [0.3]])
    results = [_result("d1"), _result("d2")]
    out = reranker.rerank("query", results, top_k=2)
    assert len(out) == 2
    assert out[0].doc_id == "d1"  # d1 has higher score


def test_onnx_reranker_respects_top_k():
    reranker = _make_onnx_reranker()
    import torch
    reranker._model.return_value.logits = torch.tensor([[0.9], [0.5], [0.1]])
    results = [_result("d1"), _result("d2"), _result("d3")]
    out = reranker.rerank("query", results, top_k=2)
    assert len(out) == 2


def test_from_env_returns_reranker_when_onnx_disabled(monkeypatch):
    monkeypatch.delenv("RERANKER_USE_ONNX", raising=False)
    from src.internal.retrieval.onnx_reranker import ONNXReranker
    from src.internal.retrieval.reranker import Reranker
    monkeypatch.delenv("RERANKER_PROVIDER", raising=False)
    result = ONNXReranker.from_env()
    assert result is None or isinstance(result, Reranker)
```

- [ ] **Step 2: Write failing tests for eval_runner M6**

Append to `tests/unit/retrieval/test_eval_runner.py`. First read the file, then append:

```python
# Append to tests/unit/retrieval/test_eval_runner.py

def test_run_eval_latency_includes_mean():
    """latency_ms dict now includes 'mean' field."""
    qa_path = _write_qa([{"query": "q", "relevant_doc_ids": ["d1"]}])
    svc = _make_service([["d1"]])
    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = [
        RetrievalResult(doc_id="d1", title="", text="", url=None, score=0.9)
    ]
    metrics = run_eval(qa_path, service=svc, top_k=1, reranker=fake_reranker)
    assert "mean" in metrics["latency_ms"]


def test_run_eval_slo_passes_when_fast(monkeypatch):
    """No error when latency is within slo_ms."""
    qa_path = _write_qa([{"query": "q", "relevant_doc_ids": ["d1"]}])
    svc = _make_service([["d1"]])
    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = [
        RetrievalResult(doc_id="d1", title="", text="", url=None, score=0.9)
    ]
    # slo_ms=10000 should never be exceeded by a mock
    metrics = run_eval(qa_path, service=svc, top_k=1, reranker=fake_reranker, slo_ms=10000)
    assert "latency_ms" in metrics


def test_run_eval_slo_raises_when_exceeded():
    """SLOViolationError raised when p99 latency exceeds slo_ms."""
    import time as _time
    qa_path = _write_qa([{"query": "q", "relevant_doc_ids": ["d1"]}])
    svc = _make_service([["d1"]])
    fake_reranker = MagicMock()

    def slow_rerank(*_):
        _time.sleep(0.1)
        return [RetrievalResult(doc_id="d1", title="", text="", url=None, score=0.9)]

    fake_reranker.rerank.side_effect = slow_rerank

    from src.internal.retrieval.eval_runner import SLOViolationError
    with pytest.raises(SLOViolationError):
        run_eval(qa_path, service=svc, top_k=1, reranker=fake_reranker, slo_ms=1)
```

- [ ] **Step 3: Run tests to verify they fail**

```
pytest tests/unit/retrieval/test_onnx_reranker.py tests/unit/retrieval/test_eval_runner.py -v -k "mean or slo"
```
Expected: FAIL (missing files/attributes)

- [ ] **Step 4: Implement ONNXReranker**

```python
# src/internal/retrieval/onnx_reranker.py
from __future__ import annotations

import dataclasses
import logging
import os

from src.internal.retrieval.backends.base import RetrievalResult

logger = logging.getLogger(__name__)


class ONNXReranker:
    """Drop-in Reranker replacement using ONNX runtime (requires optimum)."""

    def __init__(self, model_name: str, *, device: str = "cpu") -> None:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = ORTModelForSequenceClassification.from_pretrained(
            model_name, export=True
        )
        self._device = device

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        import torch

        pairs = [[query, f"{r.title}\n{r.text}"] for r in results]
        inputs = self._tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits.squeeze(-1)
        scores = logits.tolist() if hasattr(logits, "tolist") else list(logits)
        if isinstance(scores, float):
            scores = [scores]
        scored = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
        return [dataclasses.replace(r, score=float(s)) for s, r in scored[:top_k]]

    @staticmethod
    def from_env():
        """Returns ONNXReranker or falls back to Reranker. Returns None if no provider set."""
        from src.internal.retrieval.reranker import Reranker

        if os.environ.get("RERANKER_USE_ONNX", "").lower() not in ("1", "true", "yes"):
            return Reranker.from_env()
        try:
            model = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
            return ONNXReranker(model, device=os.environ.get("RERANKER_DEVICE", "cpu"))
        except ImportError:
            logger.warning("optimum not installed; falling back to PyTorch Reranker")
            return Reranker.from_env()
```

- [ ] **Step 5: Add SLOViolationError and mean latency to eval_runner.py**

Read `src/internal/retrieval/eval_runner.py` first, then apply these changes:

```python
# Add after imports in eval_runner.py:
class SLOViolationError(RuntimeError):
    pass

# Modify run_eval signature:
def run_eval(
    dataset_path: str,
    *,
    service: RetrievalService | None = None,
    top_k: int = 10,
    reranker=None,
    slo_ms: int | None = None,
) -> dict:

# Add mean to _avg calls — add this helper inside run_eval before the return:
    def _round(v):
        return round(v, 1)

# In the latency_ms dict returned when reranker is not None, add mean:
    "latency_ms": {
        "mean": _round(sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0,
        "p50": _percentile(latencies_ms, 50),
        "p95": _percentile(latencies_ms, 95),
        "p99": _percentile(latencies_ms, 99),
        "n": n,
    },

# After building the return dict (before the return statement), add SLO check:
    if slo_ms is not None and latencies_ms:
        p99 = _percentile(latencies_ms, 99)
        if p99 > slo_ms:
            raise SLOViolationError(
                f"P99 reranker latency {p99}ms exceeds SLO {slo_ms}ms"
            )
```

Also add `--slo-ms` to the CLI `__main__` block:

```python
# In the argparse section at bottom of eval_runner.py, add:
parser.add_argument(
    "--slo-ms",
    type=int,
    default=None,
    help="P99 latency SLO in ms. Exits non-zero if exceeded.",
)

# Pass to run_eval:
metrics = run_eval(
    args.dataset,
    top_k=args.top_k,
    service=service,
    reranker=reranker,
    slo_ms=args.slo_ms,
)
```

- [ ] **Step 6: Verify tests pass**

```
pytest tests/unit/retrieval/test_onnx_reranker.py tests/unit/retrieval/test_eval_runner.py -v
```
Expected: onnx tests skipped (no optimum), eval_runner tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/internal/retrieval/onnx_reranker.py \
        src/internal/retrieval/eval_runner.py \
        tests/unit/retrieval/test_onnx_reranker.py \
        tests/unit/retrieval/test_eval_runner.py
git commit -m "feat(reranker): ONNXReranker drop-in + eval_runner --slo-ms and mean latency"
```

---

### Task 6: TwoStageReranker

**Files:**
- Create: `src/internal/retrieval/two_stage_reranker.py`
- Test: `tests/unit/retrieval/test_two_stage_reranker.py`

**Interfaces:**
- Consumes: any objects with `rerank(query, results, top_k) -> list[RetrievalResult]`
- Produces:
  - `TwoStageReranker(fast_reranker, heavy_reranker, *, pre_filter_top_n=50)`
  - `TwoStageReranker.rerank(query, results, top_k) -> list[RetrievalResult]`
  - `TwoStageReranker.arerank(query, results, top_k) -> list[RetrievalResult]` — async; calls `fast.arerank` then `heavy.arerank` when available
  - `TwoStageReranker.from_env(fast_reranker, heavy_reranker) -> TwoStageReranker`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/retrieval/test_two_stage_reranker.py
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, call

import pytest

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.two_stage_reranker import TwoStageReranker


def _result(doc_id: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(doc_id=doc_id, title="t", text="c", url=None, score=score)


def _make_fast(return_val):
    m = MagicMock()
    m.rerank.return_value = return_val
    return m


def _make_heavy(return_val):
    m = MagicMock()
    m.rerank.return_value = return_val
    return m


def test_fast_gets_all_candidates():
    """Fast reranker receives all input results."""
    fast = _make_fast([_result("d1"), _result("d2")])
    heavy = _make_heavy([_result("d1")])
    tsr = TwoStageReranker(fast, heavy, pre_filter_top_n=2)
    inputs = [_result("d1"), _result("d2"), _result("d3")]
    tsr.rerank("q", inputs, top_k=1)
    fast.rerank.assert_called_once_with("q", inputs, 2)


def test_heavy_gets_fast_output():
    """Heavy reranker receives fast reranker's output."""
    fast_out = [_result("d2"), _result("d1")]
    fast = _make_fast(fast_out)
    heavy = _make_heavy([_result("d2")])
    tsr = TwoStageReranker(fast, heavy, pre_filter_top_n=2)
    tsr.rerank("q", [_result("d1"), _result("d2")], top_k=1)
    heavy.rerank.assert_called_once_with("q", fast_out, 1)


def test_returns_heavy_output():
    fast = _make_fast([_result("d1"), _result("d2")])
    heavy = _make_heavy([_result("d1")])
    tsr = TwoStageReranker(fast, heavy, pre_filter_top_n=2)
    result = tsr.rerank("q", [_result("d1"), _result("d2")], top_k=1)
    assert [r.doc_id for r in result] == ["d1"]


def test_empty_inputs_returns_empty():
    fast = _make_fast([])
    heavy = _make_heavy([])
    tsr = TwoStageReranker(fast, heavy, pre_filter_top_n=5)
    assert tsr.rerank("q", [], top_k=3) == []


def test_async_rerank_uses_arerank_when_available():
    fast = MagicMock()
    fast.arerank = asyncio.coroutine(lambda q, r, k: [_result("d1")])
    heavy = MagicMock()
    heavy.arerank = asyncio.coroutine(lambda q, r, k: [_result("d1")])

    async def _run():
        tsr = TwoStageReranker(fast, heavy, pre_filter_top_n=2)
        return await tsr.arerank("q", [_result("d1")], top_k=1)

    result = asyncio.run(_run())
    assert result[0].doc_id == "d1"


def test_from_env_reads_pre_filter_top_n(monkeypatch):
    monkeypatch.setenv("RERANKER_PRE_FILTER_TOP_N", "25")
    fast = MagicMock()
    heavy = MagicMock()
    tsr = TwoStageReranker.from_env(fast, heavy)
    assert tsr._pre_n == 25
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/unit/retrieval/test_two_stage_reranker.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement TwoStageReranker**

```python
# src/internal/retrieval/two_stage_reranker.py
from __future__ import annotations

import os

from src.internal.retrieval.backends.base import RetrievalResult


class TwoStageReranker:
    """Chains a fast pre-filter on all candidates, then a heavy scorer on top-N."""

    def __init__(
        self,
        fast_reranker,
        heavy_reranker,
        *,
        pre_filter_top_n: int = 50,
    ) -> None:
        self._fast = fast_reranker
        self._heavy = heavy_reranker
        self._pre_n = pre_filter_top_n

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        if not results:
            return results
        candidates = self._fast.rerank(query, results, self._pre_n)
        return self._heavy.rerank(query, candidates, top_k)

    async def arerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        if not results:
            return results
        if hasattr(self._fast, "arerank"):
            candidates = await self._fast.arerank(query, results, self._pre_n)
        else:
            candidates = self._fast.rerank(query, results, self._pre_n)
        if hasattr(self._heavy, "arerank"):
            return await self._heavy.arerank(query, candidates, top_k)
        return self._heavy.rerank(query, candidates, top_k)

    @classmethod
    def from_env(cls, fast_reranker, heavy_reranker) -> "TwoStageReranker":
        return cls(
            fast_reranker,
            heavy_reranker,
            pre_filter_top_n=int(os.environ.get("RERANKER_PRE_FILTER_TOP_N", "50")),
        )
```

- [ ] **Step 4: Verify tests pass**

```
pytest tests/unit/retrieval/test_two_stage_reranker.py -v
```
Expected: all PASS (async_rerank test may be skipped if `asyncio.coroutine` deprecated — use `async def` lambda alternative)

Note: if `asyncio.coroutine` is unavailable (Python 3.11+), replace the async test with:

```python
def test_async_rerank_falls_back_to_sync_when_no_arerank():
    fast = _make_fast([_result("d1")])
    heavy = _make_heavy([_result("d1")])
    tsr = TwoStageReranker(fast, heavy, pre_filter_top_n=2)
    result = asyncio.run(tsr.arerank("q", [_result("d1")], top_k=1))
    assert result[0].doc_id == "d1"
```

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/two_stage_reranker.py \
        tests/unit/retrieval/test_two_stage_reranker.py
git commit -m "feat(reranker): TwoStageReranker fast pre-filter then heavy scorer"
```

---

### Task 7: M7 service.py wiring — over-fetch + TwoStageReranker

**Files:**
- Modify: `src/internal/retrieval/service.py`
- Test: `tests/unit/retrieval/test_service.py` (append tests)

**Interfaces:**
- Consumes: `TwoStageReranker.from_env(fast, heavy)` from Task 6
- Produces: `RetrievalService.search()` passes pre-trim candidates to reranker using `RERANKER_OVER_FETCH_MULTIPLIER`; `from_env()` builds `TwoStageReranker` when `RERANKER_TWO_STAGE=true`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/retrieval/test_service.py`:

```python
# Append to tests/unit/retrieval/test_service.py

def test_search_passes_over_fetch_candidates_to_reranker():
    """When reranker active, fused results are NOT pre-trimmed to top_k before reranking."""
    from src.internal.retrieval.backends.base import RetrievalResult

    results_6 = [
        RetrievalResult(doc_id=f"d{i}", title="", text="", url=None, score=float(i))
        for i in range(6)
    ]

    backend = MagicMock()
    backend.search_sparse.return_value = results_6
    backend.search_dense.side_effect = NotImplementedError

    reranker = MagicMock()
    reranker.rerank.return_value = results_6[:3]

    svc = RetrievalService(backend, reranker=reranker)

    import os
    with patch.dict(os.environ, {"OVER_FETCH_MULTIPLIER": "2", "RERANKER_OVER_FETCH_MULTIPLIER": "3"}):
        svc.search("q", top_k=2)

    # reranker.rerank should have received more than top_k=2 candidates
    called_results = reranker.rerank.call_args[0][1]
    assert len(called_results) > 2


def test_from_env_builds_two_stage_reranker(monkeypatch):
    monkeypatch.setenv("RERANKER_TWO_STAGE", "true")
    monkeypatch.setenv("RERANKER_PROVIDER", "local")
    monkeypatch.setenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    monkeypatch.setenv("RERANKER_FAST_MODEL", "BAAI/bge-reranker-base")

    from src.internal.retrieval.two_stage_reranker import TwoStageReranker

    with (
        patch("src.internal.retrieval.service._build_backend", return_value=MagicMock()),
        patch("src.internal.retrieval.reranker.SentenceTransformerReranker.load", return_value=MagicMock()),
    ):
        svc = RetrievalService.from_env()

    assert isinstance(svc._reranker, TwoStageReranker)
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/unit/retrieval/test_service.py -v -k "over_fetch or two_stage"
```
Expected: FAIL

- [ ] **Step 3: Update service.py search() to use RERANKER_OVER_FETCH_MULTIPLIER**

Read `src/internal/retrieval/service.py`. In `search()`, replace the `over_fetch` line and the fusing section:

```python
# Replace existing over_fetch line:
import math
reranker_multiplier = (
    float(os.environ.get("RERANKER_OVER_FETCH_MULTIPLIER", "2.0"))
    if self._reranker else 1.0
)
over_fetch = math.ceil(top_k * int(os.environ.get("OVER_FETCH_MULTIPLIER", "2")) * reranker_multiplier)

# Replace fusing section (currently trims to top_k before reranking):
reranker_fetch = math.ceil(top_k * reranker_multiplier)

if len(all_result_sets) > 1:
    fused = rrf_fuse(all_result_sets)
    fused = mmr_rerank(fused, top_k=reranker_fetch)
    mode = f"{base_mode}+rag_fusion"
else:
    raw = all_result_sets[0] if all_result_sets else []
    if base_mode == "hybrid":
        fused = mmr_rerank(raw, top_k=reranker_fetch)
    else:
        fused = raw[:reranker_fetch]
    mode = base_mode
```

- [ ] **Step 4: Replace the reranker-building block in service.py from_env()**

This task replaces the RERANKER_ASYNC block added in Task 4 with a unified block that handles both two-stage and async cases. The key correctness point: when two-stage is enabled, EACH leg is wrapped in async independently (not the outer TwoStageReranker).

Replace the entire reranker-building section (from `base_reranker = Reranker.from_env()` through the RERANKER_ASYNC block) with:

```python
    base_reranker = Reranker.from_env()

    _async = os.environ.get("RERANKER_ASYNC", "").lower() in ("1", "true", "yes")
    _two_stage = os.environ.get("RERANKER_TWO_STAGE", "").lower() in ("1", "true", "yes")

    def _wrap_async(r):
        if not _async:
            return r
        from src.internal.retrieval.async_reranker import AsyncReranker
        from src.internal.retrieval.cached_reranker import CachedReranker
        return CachedReranker.from_env(AsyncReranker.from_env(r))

    if base_reranker is not None:
        if _two_stage:
            from src.internal.retrieval.two_stage_reranker import TwoStageReranker
            fast_model = os.environ.get("RERANKER_FAST_MODEL")
            if fast_model:
                fast_cfg = RerankerConfig(
                    provider=os.environ.get("RERANKER_PROVIDER", "local"),  # type: ignore[arg-type]
                    model=fast_model,
                    batch_size=int(os.environ.get("RERANKER_BATCH_SIZE", "32")),
                    device=os.environ.get("RERANKER_DEVICE", "cpu"),
                    api_key=os.environ.get("COHERE_API_KEY"),
                )
                fast_base = Reranker(fast_cfg)
            else:
                fast_base = base_reranker
            base_reranker = TwoStageReranker.from_env(
                _wrap_async(fast_base),
                _wrap_async(base_reranker),
            )
        else:
            base_reranker = _wrap_async(base_reranker)
```

Also add `from src.internal.retrieval.reranker import RerankerConfig` to the `from_env` imports block (it's already imported via `Reranker` but `RerankerConfig` needs an explicit import).

Add `import math` after the existing stdlib imports at the top of `service.py` if not present.

- [ ] **Step 5: Add `import math` to service.py if not already present**

Check the top of `service.py` — add `import math` after existing stdlib imports if missing.

- [ ] **Step 6: Verify tests pass**

```
pytest tests/unit/retrieval/test_service.py -v
```
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add src/internal/retrieval/service.py \
        tests/unit/retrieval/test_service.py
git commit -m "feat(service): TwoStageReranker wiring and RERANKER_OVER_FETCH_MULTIPLIER"
```

---

### Task 8: eval_metrics extensions + Cohere v3 adapter

**Files:**
- Modify: `src/internal/retrieval/eval_metrics.py` (append `map_at_k`, `reranker_improvement_ratio`)
- Modify: `src/internal/retrieval/reranker.py` (add `_cohere_documents` helper)
- Test: `tests/unit/retrieval/test_eval_metrics.py` (append tests)
- Test: `tests/unit/retrieval/test_reranker.py` (append Cohere v3 test)

**Interfaces:**
- Produces:
  - `map_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float`
  - `reranker_improvement_ratio(pre_ndcg: float, post_ndcg: float) -> float`
  - `_cohere_documents(texts: list[str]) -> list[dict] | list[str]` (private helper in reranker.py)

- [ ] **Step 1: Write failing tests for new metrics**

Append to `tests/unit/retrieval/test_eval_metrics.py`:

```python
# Append to tests/unit/retrieval/test_eval_metrics.py

from src.internal.retrieval.eval_metrics import map_at_k, reranker_improvement_ratio


def test_map_at_k_perfect():
    assert map_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)


def test_map_at_k_partial():
    # Only "a" is relevant, found at rank 1 → AP = 1/1 / 1 = 1.0
    assert map_at_k(["a", "x", "y"], {"a"}, k=3) == pytest.approx(1.0)


def test_map_at_k_second_rank():
    # "a" at rank 2, "b" at rank 1 (not relevant) → precision when "a" found = 1/2
    assert map_at_k(["x", "a"], {"a"}, k=2) == pytest.approx(0.5)


def test_map_at_k_none_found():
    assert map_at_k(["x", "y"], {"a", "b"}, k=2) == pytest.approx(0.0)


def test_map_at_k_empty_relevant():
    assert map_at_k(["a", "b"], set(), k=5) == 0.0


def test_reranker_improvement_positive():
    ratio = reranker_improvement_ratio(pre_ndcg=0.5, post_ndcg=0.6)
    assert ratio == pytest.approx(0.2)


def test_reranker_improvement_negative():
    ratio = reranker_improvement_ratio(pre_ndcg=0.6, post_ndcg=0.5)
    assert ratio == pytest.approx(-1 / 6, rel=1e-4)


def test_reranker_improvement_zero_pre():
    assert reranker_improvement_ratio(pre_ndcg=0.0, post_ndcg=0.5) == 0.0
```

- [ ] **Step 2: Write failing test for Cohere v3**

Append to `tests/unit/retrieval/test_reranker.py`:

```python
# Append to tests/unit/retrieval/test_reranker.py

def test_cohere_documents_v4_returns_dicts():
    from unittest.mock import patch, MagicMock
    import sys

    fake_cohere = MagicMock()
    fake_cohere.__version__ = "4.0.0"
    with patch.dict(sys.modules, {"cohere": fake_cohere}):
        from importlib import reload
        import src.internal.retrieval.reranker as rmod
        reload(rmod)
        result = rmod._cohere_documents(["text1", "text2"])
    assert result == [{"text": "text1"}, {"text": "text2"}]


def test_cohere_documents_v3_returns_strings():
    from unittest.mock import patch, MagicMock
    import sys

    fake_cohere = MagicMock()
    fake_cohere.__version__ = "3.9.0"
    with patch.dict(sys.modules, {"cohere": fake_cohere}):
        from importlib import reload
        import src.internal.retrieval.reranker as rmod
        reload(rmod)
        result = rmod._cohere_documents(["text1", "text2"])
    assert result == ["text1", "text2"]
```

- [ ] **Step 3: Run tests to verify they fail**

```
pytest tests/unit/retrieval/test_eval_metrics.py tests/unit/retrieval/test_reranker.py -v -k "map_at_k or improvement or cohere_doc"
```
Expected: FAIL

- [ ] **Step 4: Append new metrics to eval_metrics.py**

Read `src/internal/retrieval/eval_metrics.py` first, then append:

```python
def map_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Mean average precision at k (binary relevance)."""
    if not relevant_ids:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], 1):
        if doc_id in relevant_ids:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant_ids)


def reranker_improvement_ratio(pre_ndcg: float, post_ndcg: float) -> float:
    """(post / pre) - 1; negative means reranker hurt quality. Returns 0.0 if pre is 0."""
    if pre_ndcg == 0.0:
        return 0.0
    return post_ndcg / pre_ndcg - 1.0
```

- [ ] **Step 5: Add _cohere_documents to reranker.py**

Read `src/internal/retrieval/reranker.py`. Add after the logger line, and update `_rerank_cohere`:

```python
# Add module-level helper (after logger = ...):
def _cohere_documents(texts: list[str]) -> "list[dict] | list[str]":
    """Return Cohere v4+ document dicts or raw strings for older clients."""
    try:
        import cohere
        major = int(cohere.__version__.split(".")[0])
        if major >= 4:
            return [{"text": t} for t in texts]
    except Exception:
        pass
    return texts

# In _rerank_cohere, replace:
#     coro = cohere_rerank_api(query, passages, ...)
# with:
#     coro = cohere_rerank_api(query, _cohere_documents(passages), ...)
```

- [ ] **Step 6: Verify tests pass**

```
pytest tests/unit/retrieval/test_eval_metrics.py tests/unit/retrieval/test_reranker.py -v
```
Expected: all PASS (cohere_doc tests may be fragile due to module reload — mark `xfail` if reload is flaky, and verify the logic is correct manually)

- [ ] **Step 7: Commit**

```bash
git add src/internal/retrieval/eval_metrics.py \
        src/internal/retrieval/reranker.py \
        tests/unit/retrieval/test_eval_metrics.py \
        tests/unit/retrieval/test_reranker.py
git commit -m "feat(eval): map_at_k, reranker_improvement_ratio; Cohere v3/v4 document format"
```

---

### Task 9: RerankerBenchmark CLI + eval_runner --compare-baseline

**Files:**
- Create: `src/internal/retrieval/reranker_benchmark.py`
- Modify: `src/internal/retrieval/eval_runner.py` (add `--compare-baseline`, `map_at_k` in output)
- Test: `tests/unit/retrieval/test_reranker_benchmark.py`
- Test: `tests/unit/retrieval/test_eval_runner.py` (append compare-baseline test)

**Interfaces:**
- Consumes: `Reranker`, `RerankerConfig`, `ndcg_at_k`, `mrr`, `map_at_k`, `_percentile` from eval_runner
- Produces:
  - `run_benchmark(qa_pairs_path, *, models, batch_sizes, max_tokens_list, top_k=10, output_path=None) -> list[dict]`
  - CLI: `python -m src.internal.retrieval.reranker_benchmark --qa-pairs ... --models ... --batch-sizes ... --max-tokens ...`
  - `run_eval(..., compare_baseline=False)` — when True, adds `reranker_improvement_ratio` to output

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/retrieval/test_reranker_benchmark.py
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.reranker_benchmark import run_benchmark


def _result(doc_id: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(doc_id=doc_id, title="t", text="c", url=None, score=score)


def _write_qa(path, *, with_candidates=True):
    entry = {
        "query": "test query",
        "relevant_doc_ids": ["d1"],
    }
    if with_candidates:
        entry["candidates"] = [
            {"doc_id": "d1", "title": "t", "text": "c", "url": None, "score": 0.9},
            {"doc_id": "d2", "title": "t", "text": "c", "url": None, "score": 0.5},
        ]
    path.write_text(json.dumps(entry) + "\n")


def test_run_benchmark_returns_results(tmp_path):
    qa = tmp_path / "qa.jsonl"
    _write_qa(qa)

    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = [_result("d1"), _result("d2")]

    with patch(
        "src.internal.retrieval.reranker_benchmark.Reranker",
        return_value=fake_reranker,
    ):
        results = run_benchmark(
            str(qa),
            models=["BAAI/bge-reranker-base"],
            batch_sizes=[8],
            max_tokens_list=[256],
        )

    assert len(results) == 1
    row = results[0]
    assert "model" in row
    assert "batch_size" in row
    assert "max_tokens" in row
    assert "ndcg@10" in row
    assert "mrr" in row
    assert "p99_ms" in row
    assert "mean_ms" in row


def test_run_benchmark_output_jsonl(tmp_path):
    qa = tmp_path / "qa.jsonl"
    _write_qa(qa)
    out = tmp_path / "bench.jsonl"

    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = [_result("d1")]

    with patch("src.internal.retrieval.reranker_benchmark.Reranker", return_value=fake_reranker):
        run_benchmark(
            str(qa),
            models=["m1"],
            batch_sizes=[4],
            max_tokens_list=[128],
            output_path=str(out),
        )

    assert out.exists()
    rows = [json.loads(line) for line in out.read_text().strip().splitlines()]
    assert len(rows) == 1
```

Append to `tests/unit/retrieval/test_eval_runner.py`:

```python
# Append to tests/unit/retrieval/test_eval_runner.py

def test_run_eval_compare_baseline_includes_improvement_ratio():
    from src.internal.retrieval.eval_runner import run_eval
    qa_path = _write_qa([{"query": "q", "relevant_doc_ids": ["d1"]}])
    svc = _make_service([["d1"]])
    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = [
        RetrievalResult(doc_id="d1", title="", text="", url=None, score=0.9)
    ]
    metrics = run_eval(
        qa_path, service=svc, top_k=1, reranker=fake_reranker, compare_baseline=True
    )
    assert "reranker_improvement_ratio" in metrics
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/unit/retrieval/test_reranker_benchmark.py tests/unit/retrieval/test_eval_runner.py -v -k "benchmark or compare"
```
Expected: FAIL

- [ ] **Step 3: Implement RerankerBenchmark CLI**

```python
# src/internal/retrieval/reranker_benchmark.py
from __future__ import annotations

import json
import math
import time

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.eval_metrics import map_at_k
from src.internal.retrieval.eval_metrics import mrr as mrr_score
from src.internal.retrieval.eval_metrics import ndcg_at_k
from src.internal.retrieval.passage_truncator import PassageTruncator
from src.internal.retrieval.reranker import Reranker, RerankerConfig


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(len(sorted_vals) * p / 100) - 1))
    return round(sorted_vals[idx], 1)


def run_benchmark(
    qa_pairs_path: str,
    *,
    models: list[str],
    batch_sizes: list[int],
    max_tokens_list: list[int],
    top_k: int = 10,
    output_path: str | None = None,
) -> list[dict]:
    """Grid search over model × batch_size × max_tokens. QA pairs must include 'candidates'."""
    with open(qa_pairs_path) as f:
        qa_pairs = [json.loads(line) for line in f if line.strip()]

    rows = []
    for model in models:
        for batch_size in batch_sizes:
            for max_tokens in max_tokens_list:
                config = RerankerConfig(provider="local", model=model, batch_size=batch_size)
                reranker = Reranker(config)
                truncator = PassageTruncator(max_tokens=max_tokens)

                ndcgs, mrrs, maps, latencies = [], [], [], []
                for item in qa_pairs:
                    query: str = item["query"]
                    relevant: set[str] = set(item["relevant_doc_ids"])
                    raw_candidates = item.get("candidates", [])
                    candidates = [RetrievalResult(**c) for c in raw_candidates]
                    # Apply truncation before reranking
                    truncated = [
                        RetrievalResult(
                            doc_id=c.doc_id,
                            title=c.title,
                            text=truncator.truncate(c.text),
                            url=c.url,
                            score=c.score,
                        )
                        for c in candidates
                    ]

                    t0 = time.monotonic()
                    reranked = reranker.rerank(query, truncated, top_k)
                    latencies.append((time.monotonic() - t0) * 1000)

                    retrieved = [r.doc_id for r in reranked]
                    ndcgs.append(ndcg_at_k(retrieved, relevant, top_k))
                    mrrs.append(mrr_score(retrieved, relevant))
                    maps.append(map_at_k(retrieved, relevant, top_k))

                n = len(qa_pairs)

                def _avg(lst):
                    return round(sum(lst) / n, 4) if n else 0.0

                row = {
                    "model": model,
                    "batch_size": batch_size,
                    "max_tokens": max_tokens,
                    f"ndcg@{top_k}": _avg(ndcgs),
                    "mrr": _avg(mrrs),
                    f"map@{top_k}": _avg(maps),
                    "mean_ms": round(sum(latencies) / n, 1) if n else 0.0,
                    "p50_ms": _percentile(latencies, 50),
                    "p99_ms": _percentile(latencies, 99),
                }
                rows.append(row)

    if output_path:
        with open(output_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    # Print ranked table
    sorted_rows = sorted(rows, key=lambda r: r.get(f"ndcg@{top_k}", 0), reverse=True)
    header = f"{'model':<35} {'batch':>6} {'tok':>5} {f'ndcg@{top_k}':>8} {'mrr':>6} {'p99ms':>7}"
    print(header)
    print("-" * len(header))
    for r in sorted_rows:
        print(
            f"{r['model']:<35} {r['batch_size']:>6} {r['max_tokens']:>5} "
            f"{r[f'ndcg@{top_k}']:>8.4f} {r['mrr']:>6.4f} {r['p99_ms']:>7.1f}"
        )

    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reranker model/config benchmark")
    parser.add_argument("--qa-pairs", required=True, help="Path to qa_pairs.jsonl (with 'candidates' field)")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32])
    parser.add_argument("--max-tokens", type=int, nargs="+", default=[512], dest="max_tokens")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_benchmark(
        args.qa_pairs,
        models=args.models,
        batch_sizes=args.batch_sizes,
        max_tokens_list=args.max_tokens,
        top_k=args.top_k,
        output_path=args.output,
    )
```

- [ ] **Step 4: Add compare_baseline to eval_runner.py**

Read `src/internal/retrieval/eval_runner.py`. Update `run_eval` signature and return:

```python
# Update run_eval signature:
def run_eval(
    dataset_path: str,
    *,
    service: RetrievalService | None = None,
    top_k: int = 10,
    reranker=None,
    slo_ms: int | None = None,
    compare_baseline: bool = False,
) -> dict:

# Add import at top:
from .eval_metrics import map_at_k, reranker_improvement_ratio

# In the return dict when reranker is not None, add reranker_improvement_ratio when compare_baseline=True:
    result = {
        "retrieval": retrieval_metrics,
        "reranked": {
            f"recall@{top_k}": _avg(r_recalls),
            f"ndcg@{top_k}": _avg(r_ndcgs),
            f"map@{top_k}": round(sum(r_maps) / n, 4) if n else 0.0,
            "mrr": _avg(r_mrrs),
            "num_queries": n,
        },
        "latency_ms": {
            "mean": round(sum(latencies_ms) / len(latencies_ms), 1) if latencies_ms else 0.0,
            "p50": _percentile(latencies_ms, 50),
            "p95": _percentile(latencies_ms, 95),
            "p99": _percentile(latencies_ms, 99),
            "n": n,
        },
    }
    if compare_baseline:
        result["reranker_improvement_ratio"] = reranker_improvement_ratio(
            retrieval_metrics[f"ndcg@{top_k}"],
            result["reranked"][f"ndcg@{top_k}"],
        )
```

Also add `r_maps` list tracking — add `r_maps = []` alongside `r_recalls` at the top of the loop, and `r_maps.append(map_at_k(r_retrieved, relevant, top_k))` inside the reranker block.

Add `--compare-baseline` to the CLI argparse block:

```python
parser.add_argument(
    "--compare-baseline",
    action="store_true",
    help="Print reranker_improvement_ratio vs retrieval-only NDCG",
)
# Pass to run_eval:
metrics = run_eval(
    args.dataset,
    top_k=args.top_k,
    service=service,
    reranker=reranker,
    slo_ms=args.slo_ms,
    compare_baseline=args.compare_baseline,
)
```

- [ ] **Step 5: Verify all tests pass**

```
pytest tests/unit/retrieval/ -v
```
Expected: all PASS (onnx tests skipped)

- [ ] **Step 6: Commit**

```bash
git add src/internal/retrieval/reranker_benchmark.py \
        src/internal/retrieval/eval_runner.py \
        tests/unit/retrieval/test_reranker_benchmark.py \
        tests/unit/retrieval/test_eval_runner.py
git commit -m "feat(reranker): RerankerBenchmark CLI and eval_runner --compare-baseline with MAP@k"
```

---

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
| PassageTruncator + RERANKER_MAX_TOKENS | Task 1 |
| AsyncReranker + RerankerTimeoutError | Task 2 |
| CachedReranker + Redis cache + stats | Task 3 |
| M5 service.py wiring | Task 4 |
| ONNXReranker + optimum fallback | Task 5 |
| eval_runner --slo-ms + mean latency | Task 5 |
| TwoStageReranker + arerank | Task 6 |
| M7 service.py over-fetch + two-stage wiring | Task 7 |
| map_at_k + reranker_improvement_ratio | Task 8 |
| Cohere v3 _cohere_documents | Task 8 |
| RerankerBenchmark CLI | Task 9 |
| eval_runner --compare-baseline | Task 9 |
