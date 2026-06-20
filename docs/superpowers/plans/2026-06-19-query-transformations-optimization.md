# Query Transformations Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add latency wrappers, true Multi-Query generation, a learned query router, and offline benchmarking on top of the existing `QueryTransformPipeline`, all behind `QT_*` flags defaulting off.

**Architecture:** The leaf `QueryTransformPipeline` is refactored to expose per-transform *jobs* and bundle *assembly* so wrappers can reuse its orchestration. Latency wrappers (`AsyncQueryTransformPipeline`, `CachedQueryTransformPipeline`) and a routing wrapper (`RoutedQueryTransformPipeline`) compose around it via a factory, exactly like the existing reranker wrapper chain. A per-query `config_override` threads through every layer so the router can pick transforms per query.

**Tech Stack:** Python 3.10+, `concurrent.futures`, `redis`, `scikit-learn` + `joblib` (M7), `pytest`.

## Global Constraints

- Every new behaviour is gated by a `QT_*` env var that defaults to **off**. With all `QT_*` unset, `RetrievalService.from_env()` must produce `pipeline is None` and search behaviour must be byte-identical to today.
- Wrappers share the leaf interface: `transform(query, filters=None, *, config_override=None) -> TransformedQueryBundle` and a `max_variants` property and a `base_config` property.
- Every transformer is fallback-safe: an LLM failure or timeout in one transform degrades that field to its empty/None default; the bundle is still returned. Never raise out of `transform()`.
- Match existing patterns: mirror `async_reranker.py`, `cached_reranker.py`, `reranker_factory.py`, `reranker_benchmark.py`. Use `MagicMock`/fake LLMs in tests — no network.
- `RetrievalResult` fields are `doc_id, title, text, url, score, metadata` (from `src/internal/retrieval/backends/base.py`).
- The LLM interface is `LLMClient.complete(messages: list[ChatMessage]) -> LLMResponse | str` (`src/context/models.py`); `LLMResponse.text` holds the string.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `src/context/query_transform.py` (modify) | Leaf: job-based transform, `config_override`, `base_config`, config signature, `multi_query` field/flag | 1, 5 |
| `src/internal/retrieval/async_query_transform.py` (create) | Parallel transform execution + per-transform timeout | 2 |
| `src/internal/retrieval/cached_query_transform.py` (create) | Redis bundle cache keyed by query + config signature | 3 |
| `src/internal/retrieval/query_transform_factory.py` (create) | Compose the wrapper chain from env | 4, 9 |
| `src/internal/retrieval/service.py` (modify) | Build pipeline via factory; weighted fusion + dedup wiring | 4, 6, 7 |
| `src/internal/retrieval/multi_query.py` (create) | `MultiQueryGenerator` — N paraphrases in one LLM call | 5 |
| `src/internal/retrieval/fusion.py` (modify) | `variant_weighted_rrf_fuse`, `dedup_variants` | 6, 7 |
| `src/internal/retrieval/query_router.py` (create) | Learned router + heuristic fallback | 8, 10 |
| `src/internal/retrieval/routed_query_transform.py` (create) | Per-query routing wrapper | 9 |
| `src/training/train_query_router.py` (create) | Offline training script + seed dataset | 10 |
| `src/internal/retrieval/query_transform_benchmark.py` (create) | Offline technique-combo grid benchmark + CLI | 11 |
| `src/internal/retrieval/eval_runner.py` (modify) | Transform latency in output + `--qt-slo-ms` | 12 |
| `src/internal/retrieval/query_constructor.py` (modify) | Operator/range filter extraction | 13 |
| `requirements.txt` (modify) | Add `scikit-learn`, `joblib` | 10 |

---

## Task 1: Refactor leaf to job-based transform

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
# tests/unit/test_query_transform.py — append
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

# ... existing imports, dataclasses unchanged ...


def config_signature(config: QueryTransformConfig) -> str:
    """Stable string identifying which transforms are enabled (for cache keys)."""
    return "|".join(
        [
            f"d={int(config.decompose)}",
            f"h={int(config.hyde)}",
            f"s={int(config.step_back)}",
            f"k={int(config.keywords)}",
            f"c={int(config.construct_filters)}",
            f"m={int(getattr(config, 'multi_query', False))}",
            f"mv={config.max_variants}",
        ]
    )
```

In `__init__`, always build the constructor so an override can enable it:

```python
    def __init__(self, config: QueryTransformConfig, llm: object) -> None:
        from src.context.query_enhancer import QueryEnhancer
        from src.internal.retrieval.query_constructor import QueryConstructor as QC

        self._config = config
        self._llm = llm
        self._enhancer = QueryEnhancer(llm)  # type: ignore[arg-type]
        self._constructor = QC(llm)  # type: ignore[arg-type]
```

Add `base_config` and the job/assemble/transform methods (replace the existing `transform`):

```python
    @property
    def base_config(self) -> QueryTransformConfig:
        return self._config

    def _build_jobs(
        self, query: str, config: QueryTransformConfig
    ) -> dict[str, Callable[[], object]]:
        """Map each enabled transform to a zero-arg callable producing its field value."""
        jobs: dict[str, Callable[[], object]] = {}
        if config.decompose:
            jobs["sub_queries"] = lambda: self._enhancer.decompose(query)
        if config.hyde:
            jobs["hyde_text"] = lambda: self._enhancer.hyde(query)
        if config.step_back:
            jobs["step_back"] = lambda: self._enhancer.step_back(query)
        if config.keywords:

            def _keywords() -> object:
                from src.internal.servers.secondary_llm_flows.query_expansion import (
                    expand_keywords,
                )

                return expand_keywords(query, self._llm)  # type: ignore[arg-type]

            jobs["keywords"] = _keywords
        if config.construct_filters:
            jobs["_filters"] = lambda: self._constructor.extract_filters(query)[1]
        return jobs

    def _assemble(
        self, query: str, results: dict, caller_filters: dict | None
    ) -> TransformedQueryBundle:
        extracted = results.get("_filters") or {}
        return TransformedQueryBundle(
            original=query,
            sub_queries=results.get("sub_queries") or [],
            hyde_text=results.get("hyde_text"),
            step_back=results.get("step_back"),
            keywords=results.get("keywords") or [],
            merged_filters={**extracted, **(caller_filters or {})},
        )

    def transform(
        self,
        query: str,
        filters: dict | None = None,
        *,
        config_override: QueryTransformConfig | None = None,
    ) -> TransformedQueryBundle:
        config = config_override or self._config
        jobs = self._build_jobs(query, config)
        results = {field: fn() for field, fn in jobs.items()}
        return self._assemble(query, results, filters)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_query_transform.py -v`
Expected: PASS (new tests + all pre-existing tests in the file).

- [ ] **Step 5: Commit**

```bash
git add src/context/query_transform.py tests/unit/test_query_transform.py
git commit -m "refactor(query-transform): job-based transform + config_override + signature"
```

---

## Task 2: AsyncQueryTransformPipeline

Run the leaf's transform jobs concurrently with a per-transform timeout.

**Files:**
- Create: `src/internal/retrieval/async_query_transform.py`
- Test: `tests/unit/retrieval/test_async_query_transform.py`

**Interfaces:**
- Consumes: leaf `_build_jobs`, `_assemble`, `base_config`, `max_variants`.
- Produces: `AsyncQueryTransformPipeline(base, *, timeout_ms=400, max_workers=5)` with `transform(query, filters=None, *, config_override=None)`, `max_variants`, `base_config`, and classmethod `from_env(base)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/retrieval/test_async_query_transform.py
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

- [ ] **Step 3: Write the implementation**

```python
# src/internal/retrieval/async_query_transform.py
from __future__ import annotations

import concurrent.futures
import logging
import os

from src.context.query_transform import QueryTransformConfig

logger = logging.getLogger(__name__)


class AsyncQueryTransformPipeline:
    """Wraps a leaf QueryTransformPipeline; runs transform jobs in parallel."""

    def __init__(self, base, *, timeout_ms: int = 400, max_workers: int = 5) -> None:
        self._base = base
        self._timeout_ms = timeout_ms
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    @property
    def max_variants(self) -> int:
        return self._base.max_variants

    @property
    def base_config(self) -> QueryTransformConfig:
        return self._base.base_config

    def transform(self, query, filters=None, *, config_override=None):
        config = config_override or self._base.base_config
        jobs = self._base._build_jobs(query, config)
        futures = {field: self._executor.submit(fn) for field, fn in jobs.items()}
        results: dict = {}
        for field, fut in futures.items():
            try:
                results[field] = fut.result(timeout=self._timeout_ms / 1000)
            except Exception as exc:  # timeout or transform error → degrade field
                logger.warning("transform %s failed/timed out: %s", field, exc)
                fut.cancel()
        return self._base._assemble(query, results, filters)

    @classmethod
    def from_env(cls, base) -> "AsyncQueryTransformPipeline":
        return cls(
            base,
            timeout_ms=int(os.environ.get("QT_TRANSFORM_TIMEOUT_MS", "400")),
            max_workers=int(os.environ.get("QT_MAX_WORKERS", "5")),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/retrieval/test_async_query_transform.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/async_query_transform.py tests/unit/retrieval/test_async_query_transform.py
git commit -m "feat(query-transform): AsyncQueryTransformPipeline parallel transforms + timeout"
```

---

## Task 3: CachedQueryTransformPipeline

Cache the computed bundle in Redis, keyed by query + config signature.

**Files:**
- Create: `src/internal/retrieval/cached_query_transform.py`
- Test: `tests/unit/retrieval/test_cached_query_transform.py`

**Interfaces:**
- Consumes: base pipeline `transform`, `max_variants`, `base_config`; `config_signature`; `TransformedQueryBundle`.
- Produces: `CachedQueryTransformPipeline(base, redis_client=None, *, ttl_seconds=600)` with `transform(...)`, `max_variants`, `base_config`, `stats()`, classmethod `from_env(base)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/retrieval/test_cached_query_transform.py
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
# src/internal/retrieval/cached_query_transform.py
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from typing import Any

from src.context.query_transform import (
    QueryTransformConfig,
    TransformedQueryBundle,
    config_signature,
)

logger = logging.getLogger(__name__)


def _key(query: str, sig: str) -> str:
    raw = f"{query.lower().strip()}|{sig}"
    return "qt:" + hashlib.sha256(raw.encode()).hexdigest()[:20]


class CachedQueryTransformPipeline:
    """Redis cache of TransformedQueryBundle keyed by query + config signature."""

    def __init__(self, base, redis_client: Any | None = None, *, ttl_seconds: int = 600):
        self._base = base
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    @property
    def max_variants(self) -> int:
        return self._base.max_variants

    @property
    def base_config(self) -> QueryTransformConfig:
        return self._base.base_config

    def transform(self, query, filters=None, *, config_override=None):
        config = config_override or self._base.base_config
        key = _key(query, config_signature(config))

        if self._redis is not None:
            raw = self._redis.get(key)
            if raw is not None:
                self._hits += 1
                data = json.loads(raw)
                bundle = TransformedQueryBundle(**data)
                # Re-merge caller filters (not part of the cached transform output).
                if filters:
                    return dataclasses.replace(
                        bundle, merged_filters={**bundle.merged_filters, **filters}
                    )
                return bundle

        self._misses += 1
        bundle = self._base.transform(query, filters, config_override=config_override)

        if self._redis is not None:
            self._redis.setex(key, self._ttl, json.dumps(dataclasses.asdict(bundle)))
        return bundle

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total else 0.0,
        }

    @classmethod
    def from_env(cls, base):
        url = os.environ.get("QT_CACHE_REDIS_URL")
        if not url:
            return base
        try:
            import redis

            client = redis.Redis.from_url(url)
        except Exception as exc:  # pragma: no cover - infra dependent
            logger.warning("QT cache disabled, redis unavailable: %s", exc)
            return base
        return cls(
            base, client, ttl_seconds=int(os.environ.get("QT_CACHE_TTL_SECONDS", "600"))
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/retrieval/test_cached_query_transform.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/cached_query_transform.py tests/unit/retrieval/test_cached_query_transform.py
git commit -m "feat(query-transform): CachedQueryTransformPipeline Redis bundle cache"
```

---

## Task 4: Factory + RetrievalService wiring

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
# tests/unit/retrieval/test_query_transform_factory.py
from __future__ import annotations

from unittest.mock import MagicMock

from src.internal.retrieval.query_transform_factory import (
    build_query_transform_pipeline_from_env,
)


def test_returns_none_when_all_flags_unset(monkeypatch):
    for v in ("QT_DECOMPOSE", "QT_HYDE", "QT_STEP_BACK", "QT_KEYWORDS",
              "QT_CONSTRUCT_FILTERS", "QT_ASYNC", "QT_ROUTER"):
        monkeypatch.delenv(v, raising=False)
    assert build_query_transform_pipeline_from_env(MagicMock()) is None


def test_async_wraps_leaf(monkeypatch):
    monkeypatch.setenv("QT_STEP_BACK", "true")
    monkeypatch.setenv("QT_ASYNC", "true")
    monkeypatch.delenv("QT_CACHE_REDIS_URL", raising=False)
    monkeypatch.delenv("QT_ROUTER", raising=False)
    pipe = build_query_transform_pipeline_from_env(MagicMock())
    assert type(pipe).__name__ == "AsyncQueryTransformPipeline"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/retrieval/test_query_transform_factory.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Write the implementation**

```python
# src/internal/retrieval/query_transform_factory.py
"""Assemble the query-transform wrapper chain from environment variables.

Chain (outermost → innermost):
    RoutedQueryTransformPipeline → CachedQueryTransformPipeline
        → AsyncQueryTransformPipeline → QueryTransformPipeline
Each layer is optional; unset env vars leave the chain unchanged.
RoutedQueryTransformPipeline is wired in Task 9.
"""

from __future__ import annotations

import os


def _flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def build_query_transform_pipeline_from_env(llm: object) -> object | None:
    from src.context.query_transform import QueryTransformPipeline

    leaf = QueryTransformPipeline.from_env(llm)
    if leaf is None:
        return None

    pipe: object = leaf
    if _flag("QT_ASYNC"):
        from src.internal.retrieval.async_query_transform import (
            AsyncQueryTransformPipeline,
        )

        pipe = AsyncQueryTransformPipeline.from_env(pipe)

    from src.internal.retrieval.cached_query_transform import (
        CachedQueryTransformPipeline,
    )

    pipe = CachedQueryTransformPipeline.from_env(pipe)  # returns pipe unchanged if no URL
    return pipe
```

Then in `src/internal/retrieval/service.py`, replace the pipeline-building lines (currently `from src.context.query_transform import QueryTransformPipeline` / `pipeline = QueryTransformPipeline.from_env(_build_llm())` around line 133):

```python
            from src.internal.retrieval.query_transform_factory import (
                build_query_transform_pipeline_from_env,
            )

            pipeline = build_query_transform_pipeline_from_env(_build_llm())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/retrieval/test_query_transform_factory.py tests/unit/retrieval/test_service.py -v`
Expected: PASS (factory tests + existing service tests still green).

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/query_transform_factory.py src/internal/retrieval/service.py tests/unit/retrieval/test_query_transform_factory.py
git commit -m "feat(query-transform): env-driven wrapper-chain factory + service wiring"
```

---

## Task 5: MultiQueryGenerator + bundle wiring

True Multi-Query: one LLM call → N paraphrases. Add a `multi_query` field/flag and surface variants.

**Files:**
- Create: `src/internal/retrieval/multi_query.py`
- Modify: `src/context/query_transform.py` (`QueryTransformConfig.multi_query`, `TransformedQueryBundle.multi_query`, `retrieval_variants` ordering, `_build_jobs`/`_assemble`, `from_env`)
- Test: `tests/unit/retrieval/test_multi_query.py`, `tests/unit/test_query_transform.py`

**Interfaces:**
- Consumes: `LLMClient`, `ChatMessage`.
- Produces: `MultiQueryGenerator(llm, *, n=3)` with `generate(query) -> list[str]`, classmethod `from_env(llm) -> MultiQueryGenerator | None`. New field `TransformedQueryBundle.multi_query: list[str]`; new flag `QueryTransformConfig.multi_query: bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/retrieval/test_multi_query.py
from __future__ import annotations

from unittest.mock import MagicMock

from src.internal.retrieval.multi_query import MultiQueryGenerator


def _llm(text):
    llm = MagicMock()
    llm.complete.return_value = type("R", (), {"text": text})()
    return llm


def test_parses_numbered_lines_and_caps_n():
    gen = MultiQueryGenerator(_llm("1. a\n2. b\n3. c\n4. d"), n=3)
    assert gen.generate("orig") == ["a", "b", "c"]


def test_empty_on_llm_failure():
    bad = MagicMock()
    bad.complete.side_effect = RuntimeError("boom")
    assert MultiQueryGenerator(bad).generate("q") == []
```

```python
# tests/unit/test_query_transform.py — append
def test_multi_query_variants_surface():
    from src.context.query_transform import TransformedQueryBundle

    b = TransformedQueryBundle(original="orig", multi_query=["p1", "p2"])
    variants = b.retrieval_variants(max_variants=5)
    assert "p1" in variants and "p2" in variants and variants[-1] == "orig"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/retrieval/test_multi_query.py tests/unit/test_query_transform.py::test_multi_query_variants_surface -v`
Expected: FAIL — module/field missing.

- [ ] **Step 3: Write the implementation**

```python
# src/internal/retrieval/multi_query.py
from __future__ import annotations

import logging
import os
import re

from src.context.models import ChatMessage

logger = logging.getLogger(__name__)

_PROMPT = (
    "Generate {n} alternative phrasings of the user's search query. "
    "Keep the meaning identical but vary the wording. "
    "Return each on its own line, no numbering needed.\n\nQuery: {query}"
)

_STRIP = re.compile(r"^\s*(?:\d+[.)]|[-*])\s*")


def _text(resp: object) -> str:
    if isinstance(resp, str):
        return resp
    return getattr(resp, "text", None) or getattr(resp, "content", None) or str(resp)


class MultiQueryGenerator:
    def __init__(self, llm, *, n: int = 3) -> None:
        self._llm = llm
        self._n = n

    def generate(self, query: str) -> list[str]:
        try:
            raw = _text(
                self._llm.complete(
                    [ChatMessage(role="user",
                                 content=_PROMPT.format(n=self._n, query=query))]
                )
            )
        except Exception as exc:
            logger.warning("multi-query generation failed: %s", exc)
            return []
        out: list[str] = []
        seen = {query.lower()}
        for line in raw.splitlines():
            cleaned = _STRIP.sub("", line).strip()
            if cleaned and cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                out.append(cleaned)
        return out[: self._n]

    @classmethod
    def from_env(cls, llm) -> "MultiQueryGenerator | None":
        if os.environ.get("QT_MULTI_QUERY", "").lower() not in ("1", "true", "yes"):
            return None
        return cls(llm, n=int(os.environ.get("QT_MULTI_QUERY_N", "3")))
```

In `src/context/query_transform.py`:

1. Add to `QueryTransformConfig`: `multi_query: bool = False`.
2. Add to `TransformedQueryBundle`: `multi_query: list[str] = field(default_factory=list)`.
3. In `retrieval_variants`, add multi-query right after the sub_queries loop:

```python
        for q in self.sub_queries:
            _add(q)
        for q in self.multi_query:
            _add(q)
        _add(self.hyde_text)
```

4. In `_build_jobs`, add a multi-query job (lazy import avoids an import cycle, since `multi_query.py` imports from `src.context`):

```python
        if config.multi_query:
            def _mq() -> object:
                from src.internal.retrieval.multi_query import MultiQueryGenerator

                return MultiQueryGenerator(
                    self._llm, n=int(os.environ.get("QT_MULTI_QUERY_N", "3"))
                ).generate(query)

            jobs["multi_query"] = _mq
```

5. In `_assemble`, add `multi_query=results.get("multi_query") or []` to the `TransformedQueryBundle(...)` call.
6. In `from_env`, add `multi_query=_bool("QT_MULTI_QUERY")` to the config and include it in the `any([...])` enable check.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/retrieval/test_multi_query.py tests/unit/test_query_transform.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/multi_query.py src/context/query_transform.py tests/unit/retrieval/test_multi_query.py tests/unit/test_query_transform.py
git commit -m "feat(query-transform): MultiQueryGenerator + multi_query bundle field/flag"
```

---

## Task 6: Variant-weighted RRF fusion

Weight the original query's result set above expansion variants when fusing N sets.

**Files:**
- Modify: `src/internal/retrieval/fusion.py` (new `variant_weighted_rrf_fuse`)
- Modify: `src/internal/retrieval/service.py:311-314` (fusion selection)
- Test: `tests/unit/retrieval/test_fusion.py`

**Interfaces:**
- Produces: `variant_weighted_rrf_fuse(result_sets: list[list[RetrievalResult]], weights: list[float], *, rrf_k: int = 60) -> list[RetrievalResult]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/retrieval/test_fusion.py — append (create file if absent with imports)
from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.fusion import variant_weighted_rrf_fuse


def _r(doc_id):
    return RetrievalResult(doc_id=doc_id, title="", text="", url=None, score=1.0)


def test_variant_weight_favours_heavier_set():
    # doc A only in the heavy (original) set, doc B only in a light set, same rank.
    original = [_r("A")]
    expansion = [_r("B")]
    fused = variant_weighted_rrf_fuse([original, expansion], weights=[1.0, 0.1])
    assert fused[0].doc_id == "A"


def test_uniform_weights_match_rank_order():
    fused = variant_weighted_rrf_fuse([[_r("A"), _r("B")]], weights=[1.0])
    assert [r.doc_id for r in fused] == ["A", "B"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/retrieval/test_fusion.py::test_variant_weight_favours_heavier_set -v`
Expected: FAIL — `variant_weighted_rrf_fuse` not defined.

- [ ] **Step 3: Write the implementation**

Append to `src/internal/retrieval/fusion.py`:

```python
def variant_weighted_rrf_fuse(
    result_sets: list[list[RetrievalResult]],
    weights: list[float],
    *,
    rrf_k: int = _RRF_K,
) -> list[RetrievalResult]:
    """RRF across N variant result sets, each contributing weight / (k + rank).

    weights[i] applies to result_sets[i]. Falls back to uniform when lengths
    mismatch.
    """
    if len(weights) != len(result_sets):
        weights = [1.0] * len(result_sets)

    rrf_scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, RetrievalResult] = {}
    for w, result_set in zip(weights, result_sets):
        for rank, result in enumerate(result_set, 1):
            rrf_scores[result.doc_id] += w * (1.0 / (rrf_k + rank))
            if result.doc_id not in first_seen:
                first_seen[result.doc_id] = result

    return sorted(
        [
            RetrievalResult(
                doc_id=doc_id,
                title=first_seen[doc_id].title,
                text=first_seen[doc_id].text,
                url=first_seen[doc_id].url,
                score=rrf_scores[doc_id],
                metadata=first_seen[doc_id].metadata,
            )
            for doc_id in rrf_scores
        ],
        key=lambda r: r.score,
        reverse=True,
    )
```

In `src/internal/retrieval/service.py`, where `len(all_result_sets) > 1` currently calls `rrf_fuse(all_result_sets)`, select weighted fusion when enabled. The variants list order from `retrieval_variants()` ends with the original (always last), so the original's result set is the last one:

```python
        if len(all_result_sets) > 1:
            if os.environ.get("QT_FUSION_WEIGHTED", "").lower() in ("1", "true", "yes"):
                # original is the last variant → heaviest weight
                weights = [0.3] * (len(all_result_sets) - 1) + [1.0]
                fused = variant_weighted_rrf_fuse(all_result_sets, weights)
            else:
                fused = rrf_fuse(all_result_sets)
            fused = mmr_rerank(fused, top_k=reranker_fetch)
            mode = f"{base_mode}+rag_fusion"
```

Add the import: `from .fusion import mmr_rerank, rrf_fuse, variant_weighted_rrf_fuse` and ensure `import os` is present in the module.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/retrieval/test_fusion.py tests/unit/retrieval/test_service.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/fusion.py src/internal/retrieval/service.py tests/unit/retrieval/test_fusion.py
git commit -m "feat(query-transform): variant-weighted RRF fusion (original weighted highest)"
```

---

## Task 7: Semantic variant dedup

Drop near-duplicate variants before retrieval to avoid wasted retrieval calls.

**Files:**
- Modify: `src/internal/retrieval/fusion.py` (`dedup_variants`)
- Modify: `src/internal/retrieval/service.py` (apply between `retrieval_variants()` and fan-out)
- Test: `tests/unit/retrieval/test_fusion.py`

**Interfaces:**
- Produces: `dedup_variants(variants: list[str], embed_fn: Callable[[list[str]], list[list[float]]], *, threshold: float = 0.95) -> list[str]`. Order-preserving; first occurrence wins; never drops the last element (the original).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/retrieval/test_fusion.py — append
from src.internal.retrieval.fusion import dedup_variants


def test_dedup_drops_near_duplicate():
    # Identical embeddings for the first two → second dropped; original (last) kept.
    embs = {"a": [1.0, 0.0], "a2": [1.0, 0.0], "orig": [0.0, 1.0]}
    out = dedup_variants(
        ["a", "a2", "orig"], lambda xs: [embs[x] for x in xs], threshold=0.99
    )
    assert out == ["a", "orig"]


def test_dedup_keeps_distinct():
    embs = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    out = dedup_variants(["a", "b"], lambda xs: [embs[x] for x in xs], threshold=0.99)
    assert out == ["a", "b"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/retrieval/test_fusion.py::test_dedup_drops_near_duplicate -v`
Expected: FAIL — `dedup_variants` not defined.

- [ ] **Step 3: Write the implementation**

Append to `src/internal/retrieval/fusion.py` (add `from typing import Callable` and `import math` at top):

```python
def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def dedup_variants(
    variants: list[str],
    embed_fn: Callable[[list[str]], list[list[float]]],
    *,
    threshold: float = 0.95,
) -> list[str]:
    """Drop variants whose embedding cosine ≥ threshold to an earlier kept variant.

    Order-preserving; the last variant (the original) is always kept.
    """
    if len(variants) <= 1:
        return variants
    embs = embed_fn(variants)
    kept: list[str] = []
    kept_embs: list[list[float]] = []
    for i, (v, e) in enumerate(zip(variants, embs)):
        is_last = i == len(variants) - 1
        if is_last or all(_cosine(e, ke) < threshold for ke in kept_embs):
            kept.append(v)
            kept_embs.append(e)
    return kept
```

In `src/internal/retrieval/service.py`, after computing `variants = bundle.retrieval_variants(...)` and before the `ThreadPoolExecutor` fan-out, apply dedup when enabled. The backend (`self._backend`, a `RetrievalBackend`) does not currently expose a standalone embed method — only `search_sparse`/`search_dense`. So pull an optional batch-embed callable off the backend via `getattr`; when absent, dedup stays dormant (no backend change forced here):

```python
            embed_fn = getattr(self._backend, "embed", None)
            if (
                os.environ.get("QT_SEMANTIC_DEDUP", "").lower() in ("1", "true", "yes")
                and embed_fn is not None
                and len(variants) > 1
            ):
                from .fusion import dedup_variants

                threshold = float(os.environ.get("QT_SEMANTIC_DEDUP_THRESHOLD", "0.95"))
                variants = dedup_variants(variants, embed_fn, threshold=threshold)
```

`dedup_variants` is the tested deliverable here; the service wiring is a conservative, non-breaking hook. A future task can add `embed(texts: list[str]) -> list[list[float]]` to a dense backend to activate it — that is out of scope for this plan.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/retrieval/test_fusion.py tests/unit/retrieval/test_service.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/fusion.py src/internal/retrieval/service.py tests/unit/retrieval/test_fusion.py
git commit -m "feat(query-transform): semantic dedup of variants before retrieval"
```

---

## Task 8: QueryRouter (heuristic fallback)

Predict which transforms to enable per query. This task ships the heuristic path and the artifact-load scaffold (artifact training is Task 10).

**Files:**
- Create: `src/internal/retrieval/query_router.py`
- Test: `tests/unit/retrieval/test_query_router.py`

**Interfaces:**
- Consumes: `QueryTransformConfig`.
- Produces: `QueryRouter(model_path=None)` with `predict(query) -> QueryTransformConfig`, classmethod `from_env() -> QueryRouter | None`. Module constant `ROUTER_LABELS = ["decompose","hyde","step_back","keywords","construct_filters","multi_query"]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/retrieval/test_query_router.py
from __future__ import annotations

from src.internal.retrieval.query_router import QueryRouter


def test_heuristic_short_keyword_query():
    cfg = QueryRouter(model_path=None).predict("faiss index")
    assert cfg.keywords is True
    assert cfg.decompose is False


def test_heuristic_multi_clause_query():
    cfg = QueryRouter(model_path=None).predict(
        "Compare dense and sparse retrieval and explain when each wins"
    )
    assert cfg.decompose is True


def test_heuristic_date_query_constructs_filters():
    cfg = QueryRouter(model_path=None).predict("FAISS papers after 2023")
    assert cfg.construct_filters is True


def test_from_env_disabled(monkeypatch):
    monkeypatch.delenv("QT_ROUTER", raising=False)
    assert QueryRouter.from_env() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/retrieval/test_query_router.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Write the implementation**

```python
# src/internal/retrieval/query_router.py
from __future__ import annotations

import logging
import os
import re

from src.context.query_transform import QueryTransformConfig

logger = logging.getLogger(__name__)

ROUTER_LABELS = [
    "decompose",
    "hyde",
    "step_back",
    "keywords",
    "construct_filters",
    "multi_query",
]

_QUESTION_WORDS = ("what", "why", "how", "when", "where", "who", "which")
_DATE_RE = re.compile(r"\b(19|20)\d{2}\b")
_RANGE_WORDS = ("after", "before", "since", "between", "from")


def _heuristic(query: str) -> QueryTransformConfig:
    q = query.lower()
    tokens = query.split()
    n = len(tokens)
    has_question = any(w in q for w in _QUESTION_WORDS)
    multi_clause = (" and " in q) or (";" in q) or (", " in q) or n > 18
    has_date = bool(_DATE_RE.search(q)) or any(w in q for w in _RANGE_WORDS)
    short_keyword = n <= 3
    return QueryTransformConfig(
        decompose=multi_clause,
        hyde=has_question and not short_keyword,
        step_back=has_question and not multi_clause,
        keywords=short_keyword,
        construct_filters=has_date,
        multi_query=not short_keyword and not multi_clause,
    )


class QueryRouter:
    """Predict the per-query transform set. Learned model with heuristic fallback."""

    def __init__(self, model_path: str | None = None) -> None:
        self._model = None
        if model_path and os.path.exists(model_path):
            try:
                import joblib

                self._model = joblib.load(model_path)
            except Exception as exc:  # pragma: no cover - infra dependent
                logger.warning("router model load failed, using heuristic: %s", exc)

    def predict(self, query: str) -> QueryTransformConfig:
        if self._model is None:
            return _heuristic(query)
        try:
            row = self._model.predict([query])[0]
            flags = {label: bool(row[i]) for i, label in enumerate(ROUTER_LABELS)}
            return QueryTransformConfig(**flags)
        except Exception as exc:
            logger.warning("router predict failed, using heuristic: %s", exc)
            return _heuristic(query)

    @classmethod
    def from_env(cls) -> "QueryRouter | None":
        if os.environ.get("QT_ROUTER", "").lower() not in ("1", "true", "yes"):
            return None
        return cls(model_path=os.environ.get("QT_ROUTER_MODEL_PATH"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/retrieval/test_query_router.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/query_router.py tests/unit/retrieval/test_query_router.py
git commit -m "feat(query-transform): QueryRouter with heuristic fallback"
```

---

## Task 9: RoutedQueryTransformPipeline + factory wiring

Apply the router's per-query config by threading `config_override` through the chain.

**Files:**
- Create: `src/internal/retrieval/routed_query_transform.py`
- Modify: `src/internal/retrieval/query_transform_factory.py` (wrap with router when `QT_ROUTER`)
- Test: `tests/unit/retrieval/test_routed_query_transform.py`

**Interfaces:**
- Consumes: `QueryRouter.predict`, base pipeline `transform(..., config_override=...)`, `max_variants`, `base_config`.
- Produces: `RoutedQueryTransformPipeline(base, router)` with `transform(query, filters=None, *, config_override=None)`, `max_variants`, `base_config`, classmethod `from_env(base)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/retrieval/test_routed_query_transform.py
from __future__ import annotations

from unittest.mock import MagicMock

from src.context.query_transform import QueryTransformConfig, QueryTransformPipeline
from src.internal.retrieval.query_router import QueryRouter
from src.internal.retrieval.routed_query_transform import RoutedQueryTransformPipeline


def _llm(text="broad"):
    llm = MagicMock()
    llm.complete.return_value = type("R", (), {"text": text})()
    return llm


def test_router_config_applied_per_query():
    # Leaf built with all-off; router forces step_back on for a question query.
    leaf = QueryTransformPipeline(QueryTransformConfig(), _llm("broad"))
    routed = RoutedQueryTransformPipeline(leaf, QueryRouter(model_path=None))
    bundle = routed.transform("why does HNSW work")
    assert bundle.step_back == "broad"  # heuristic enabled step_back


def test_explicit_override_wins_over_router():
    leaf = QueryTransformPipeline(QueryTransformConfig(), _llm("broad"))
    routed = RoutedQueryTransformPipeline(leaf, QueryRouter(model_path=None))
    bundle = routed.transform("why x", config_override=QueryTransformConfig())
    assert bundle.step_back is None  # forced all-off override respected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/retrieval/test_routed_query_transform.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Write the implementation**

```python
# src/internal/retrieval/routed_query_transform.py
from __future__ import annotations

import os

from src.context.query_transform import QueryTransformConfig


class RoutedQueryTransformPipeline:
    """Outer wrapper: picks per-query transforms via a router, threads config down."""

    def __init__(self, base, router) -> None:
        self._base = base
        self._router = router

    @property
    def max_variants(self) -> int:
        return self._base.max_variants

    @property
    def base_config(self) -> QueryTransformConfig:
        return self._base.base_config

    def transform(self, query, filters=None, *, config_override=None):
        config = config_override or self._router.predict(query)
        # Preserve the configured max_variants from the base.
        from dataclasses import replace

        config = replace(config, max_variants=self._base.base_config.max_variants)
        return self._base.transform(query, filters, config_override=config)

    @classmethod
    def from_env(cls, base):
        from src.internal.retrieval.query_router import QueryRouter

        router = QueryRouter.from_env()
        if router is None:
            return base
        return cls(base, router)
```

In `query_transform_factory.py`, after the cache wrap, add the router as the outermost layer:

```python
    if _flag("QT_ROUTER"):
        from src.internal.retrieval.routed_query_transform import (
            RoutedQueryTransformPipeline,
        )

        pipe = RoutedQueryTransformPipeline.from_env(pipe)
    return pipe
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/retrieval/test_routed_query_transform.py tests/unit/retrieval/test_query_transform_factory.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/routed_query_transform.py src/internal/retrieval/query_transform_factory.py tests/unit/retrieval/test_routed_query_transform.py
git commit -m "feat(query-transform): RoutedQueryTransformPipeline + factory wiring"
```

---

## Task 10: Train script + seed dataset + artifact load path

Offline training that produces the joblib artifact the router loads.

**Files:**
- Create: `src/training/train_query_router.py`
- Modify: `requirements.txt` (add `scikit-learn==1.4.2`, `joblib`)
- Test: `tests/unit/retrieval/test_query_router.py` (artifact round-trip)

**Interfaces:**
- Consumes: `ROUTER_LABELS`, `QueryRouter`.
- Produces: `SEED_DATA: list[tuple[str, list[int]]]`, `build_model() -> sklearn Pipeline`, `train(output_path: str) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/retrieval/test_query_router.py — append
def test_trained_artifact_round_trips(tmp_path):
    from src.training.train_query_router import train
    from src.internal.retrieval.query_router import QueryRouter

    path = str(tmp_path / "router.joblib")
    train(path)
    cfg = QueryRouter(model_path=path).predict("faiss index")
    # A loaded model returns a valid config (booleans), not a crash.
    assert isinstance(cfg.decompose, bool)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/retrieval/test_query_router.py::test_trained_artifact_round_trips -v`
Expected: FAIL — `src.training.train_query_router` does not exist.

- [ ] **Step 3: Write the implementation**

```python
# src/training/train_query_router.py
"""Offline trainer for the QueryRouter. Produces a joblib sklearn Pipeline.

Usage: python -m src.training.train_query_router --out data/query_router.joblib
"""

from __future__ import annotations

import argparse

from src.internal.retrieval.query_router import ROUTER_LABELS

# Labels order: decompose, hyde, step_back, keywords, construct_filters, multi_query
SEED_DATA: list[tuple[str, list[int]]] = [
    ("faiss index", [0, 0, 0, 1, 0, 0]),
    ("bm25 tuning", [0, 0, 0, 1, 0, 0]),
    ("what is reciprocal rank fusion", [0, 1, 1, 0, 0, 1]),
    ("how does HNSW graph search work", [0, 1, 1, 0, 0, 1]),
    ("compare dense and sparse retrieval and when each wins", [1, 0, 0, 0, 0, 0]),
    ("explain reranking and decompose the tradeoffs and latency", [1, 0, 0, 0, 0, 0]),
    ("FAISS papers after 2023", [0, 0, 0, 0, 1, 0]),
    ("arxiv papers between 2020 and 2022 on retrieval", [0, 0, 0, 0, 1, 0]),
    ("best embedding model for semantic search", [0, 1, 1, 0, 0, 1]),
    ("vector database benchmarks", [0, 0, 0, 1, 0, 0]),
]


def build_model():
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("vec", HashingVectorizer(ngram_range=(1, 2), n_features=2**12)),
            ("clf", MultiOutputClassifier(LogisticRegression(max_iter=1000))),
        ]
    )


def train(output_path: str) -> None:
    import joblib

    assert len(SEED_DATA[0][1]) == len(ROUTER_LABELS)
    queries = [q for q, _ in SEED_DATA]
    labels = [y for _, y in SEED_DATA]
    model = build_model()
    model.fit(queries, labels)
    joblib.dump(model, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the QueryRouter model")
    parser.add_argument("--out", default="data/query_router.joblib")
    args = parser.parse_args()
    train(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

Add to `requirements.txt`:

```
scikit-learn==1.4.2
joblib>=1.3
```

Note: with only 10 seed rows, a label column may be single-valued and `LogisticRegression` will learn a constant — acceptable for the artifact-load path; quality comes from expanding `SEED_DATA` later. The test only asserts the round-trip produces a valid config.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/retrieval/test_query_router.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/training/train_query_router.py requirements.txt tests/unit/retrieval/test_query_router.py
git commit -m "feat(query-transform): offline QueryRouter trainer + seed dataset"
```

---

## Task 11: QueryTransformBenchmark CLI

Offline grid over technique combos × a labeled dataset, reporting recall/NDCG and latency.

**Files:**
- Create: `src/internal/retrieval/query_transform_benchmark.py`
- Test: `tests/unit/retrieval/test_query_transform_benchmark.py`

**Interfaces:**
- Consumes: `QueryTransformConfig`; existing `eval_metrics` helpers (`ndcg_at_k`, `recall_at_k` — verify exact names in `src/internal/retrieval/eval_metrics.py` and use them).
- Produces: `run_query_transform_benchmark(dataset, retrieve_fn, configs) -> list[dict]` where `dataset` is `list[tuple[str, set[str]]]` (query, relevant doc_ids), `retrieve_fn(query, config) -> list[str]` returns ranked doc_ids, and each result dict has keys `config_signature, recall, ndcg, mean_latency_ms`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/retrieval/test_query_transform_benchmark.py
from __future__ import annotations

from src.context.query_transform import QueryTransformConfig
from src.internal.retrieval.query_transform_benchmark import (
    run_query_transform_benchmark,
)


def test_benchmark_ranks_configs():
    dataset = [("q1", {"d1"}), ("q2", {"d2"})]

    def retrieve(query, config):
        # A config that decomposes "finds" the right doc; the other does not.
        if config.decompose:
            return {"q1": ["d1"], "q2": ["d2"]}[query]
        return ["dx"]

    configs = [QueryTransformConfig(), QueryTransformConfig(decompose=True)]
    rows = run_query_transform_benchmark(dataset, retrieve, configs, k=5)
    best = max(rows, key=lambda r: r["recall"])
    assert best["recall"] == 1.0
    assert "mean_latency_ms" in best and "config_signature" in best
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/retrieval/test_query_transform_benchmark.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Write the implementation**

```python
# src/internal/retrieval/query_transform_benchmark.py
"""Offline grid benchmark over query-transform technique combinations.

Usage: python -m src.internal.retrieval.query_transform_benchmark --dataset path.jsonl
"""

from __future__ import annotations

import time

from src.context.query_transform import QueryTransformConfig, config_signature


def _recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for d in ranked[:k] if d in relevant)
    return hits / len(relevant)


def _ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    import math

    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, d in enumerate(ranked[:k])
        if d in relevant
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def run_query_transform_benchmark(
    dataset: list[tuple[str, set[str]]],
    retrieve_fn,
    configs: list[QueryTransformConfig],
    *,
    k: int = 10,
) -> list[dict]:
    rows: list[dict] = []
    for config in configs:
        recalls, ndcgs, latencies = [], [], []
        for query, relevant in dataset:
            start = time.perf_counter()
            ranked = retrieve_fn(query, config)
            latencies.append((time.perf_counter() - start) * 1000)
            recalls.append(_recall_at_k(ranked, relevant, k))
            ndcgs.append(_ndcg_at_k(ranked, relevant, k))
        n = len(dataset) or 1
        rows.append(
            {
                "config_signature": config_signature(config),
                "recall": sum(recalls) / n,
                "ndcg": sum(ndcgs) / n,
                "mean_latency_ms": sum(latencies) / n,
            }
        )
    return sorted(rows, key=lambda r: r["recall"], reverse=True)
```

(If `src/internal/retrieval/eval_metrics.py` already exposes `recall_at_k`/`ndcg_at_k` with the same semantics, import and use them instead of the private helpers above to stay DRY.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/retrieval/test_query_transform_benchmark.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/query_transform_benchmark.py tests/unit/retrieval/test_query_transform_benchmark.py
git commit -m "feat(query-transform): offline technique-combo benchmark"
```

---

## Task 12: eval_runner transform-latency + --qt-slo-ms

Record per-query transform latency and fail the run if P99 exceeds a budget.

**Files:**
- Modify: `src/internal/retrieval/eval_runner.py`
- Test: `tests/unit/retrieval/test_eval_runner_qt_slo.py`

**Interfaces:**
- Produces: `qt_slo_exceeded(latencies_ms: list[float], slo_ms: int) -> bool` (P99 check helper), and a `--qt-slo-ms` CLI arg that exits non-zero when exceeded.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/retrieval/test_eval_runner_qt_slo.py
from src.internal.retrieval.eval_runner import qt_slo_exceeded


def test_p99_within_budget():
    assert qt_slo_exceeded([10.0] * 100, slo_ms=50) is False


def test_p99_exceeds_budget():
    lat = [10.0] * 99 + [500.0]
    assert qt_slo_exceeded(lat, slo_ms=50) is True


def test_empty_latencies_never_exceed():
    assert qt_slo_exceeded([], slo_ms=50) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/retrieval/test_eval_runner_qt_slo.py -v`
Expected: FAIL — `qt_slo_exceeded` not defined.

- [ ] **Step 3: Write the implementation**

Add to `src/internal/retrieval/eval_runner.py`:

```python
def qt_slo_exceeded(latencies_ms: list[float], slo_ms: int) -> bool:
    """True when the P99 transform latency exceeds slo_ms."""
    if not latencies_ms:
        return False
    ordered = sorted(latencies_ms)
    idx = max(0, math.ceil(0.99 * len(ordered)) - 1)
    return ordered[idx] > slo_ms
```

Ensure `import math` is present. In the CLI argument parser, add:

```python
    parser.add_argument("--qt-slo-ms", type=int, default=None,
                        help="Fail if P99 query-transform latency exceeds this budget")
```

Where the runner already times retrieval per query, also record transform latency into the per-query output dict under `qt_latency_ms` (wrap the `pipeline.transform(...)` call, or the `service.search` call when no separate transform hook exists, with `time.perf_counter()`), accumulate into a `qt_latencies` list, and after the loop:

```python
    if args.qt_slo_ms is not None and qt_slo_exceeded(qt_latencies, args.qt_slo_ms):
        import sys

        print(f"QT SLO breach: P99 > {args.qt_slo_ms}ms")
        sys.exit(1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/retrieval/test_eval_runner_qt_slo.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/eval_runner.py tests/unit/retrieval/test_eval_runner_qt_slo.py
git commit -m "feat(query-transform): eval_runner transform latency + --qt-slo-ms gate"
```

---

## Task 13: Richer QueryConstructor (operators/ranges)

Extend NL→filter extraction beyond equality to comparison operators and ranges.

**Files:**
- Modify: `src/internal/retrieval/query_constructor.py`
- Test: `tests/unit/retrieval/test_query_constructor.py`

**Interfaces:**
- Produces: `QueryConstructor.extract_filters` continues to return `(cleaned_query, filters)`; with `QT_CONSTRUCT_OPERATORS` enabled, `filters` may include range keys (`date_after`, `date_before`) already in `_KNOWN_FILTER_FIELDS`, plus numeric comparison keys `rating_gte`, `rating_lte`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/retrieval/test_query_constructor.py
from __future__ import annotations

from unittest.mock import MagicMock

from src.internal.retrieval.query_constructor import QueryConstructor


def _llm(json_text):
    llm = MagicMock()
    llm.complete.return_value = type("R", (), {"text": json_text})()
    return llm


def test_range_operators_extracted(monkeypatch):
    monkeypatch.setenv("QT_CONSTRUCT_OPERATORS", "true")
    llm = _llm(
        '{"query": "papers", "filters": '
        '{"date_after": "2023-01-01", "rating_gte": 4}}'
    )
    _, filters = QueryConstructor(llm).extract_filters("papers after 2023 rated 4+")
    assert filters["date_after"] == "2023-01-01"
    assert filters["rating_gte"] == 4


def test_operators_dropped_when_flag_off(monkeypatch):
    monkeypatch.delenv("QT_CONSTRUCT_OPERATORS", raising=False)
    llm = _llm('{"query": "papers", "filters": {"rating_gte": 4}}')
    _, filters = QueryConstructor(llm).extract_filters("papers rated 4+")
    assert "rating_gte" not in filters  # equality-only behaviour unchanged
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/retrieval/test_query_constructor.py -v`
Expected: FAIL — operator keys not in the allow-list.

- [ ] **Step 3: Write the implementation**

In `src/internal/retrieval/query_constructor.py`:

1. Add operator fields and a flag-gated allow-list:

```python
import os

_OPERATOR_FILTER_FIELDS = frozenset({"rating_gte", "rating_lte"})


def _operators_enabled() -> bool:
    return os.environ.get("QT_CONSTRUCT_OPERATORS", "").lower() in ("1", "true", "yes")
```

2. Extend the prompt (append to `_EXTRACT_PROMPT`, before the `Examples:` block) so operator fields are produced only conceptually; actual gating happens on parse:

```
  - "rating_gte": number (e.g. minimum rating like 4)
  - "rating_lte": number (maximum rating)
```

3. In `extract_filters`, widen the allowed keys when the flag is on:

```python
            allowed = set(_KNOWN_FILTER_FIELDS)
            if _operators_enabled():
                allowed |= _OPERATOR_FILTER_FIELDS
            filters: dict = {
                k: v
                for k, v in raw_filters.items()
                if k in allowed and v is not None
            }
```

(`date_after`/`date_before` are already in `_KNOWN_FILTER_FIELDS`, so ranges work today; the flag adds numeric comparison operators.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/retrieval/test_query_constructor.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/query_constructor.py tests/unit/retrieval/test_query_constructor.py
git commit -m "feat(query-transform): operator/range filter extraction (QT_CONSTRUCT_OPERATORS)"
```

---

## Final verification

- [ ] **Run the full suite**

Run: `pytest -q`
Expected: all pass.

- [ ] **Lint**

Run: `ruff check . --fix && ruff format .`
Expected: clean.

- [ ] **Regression guarantee (manual)**

Confirm with `QT_*` unset that `build_query_transform_pipeline_from_env(...)` returns `None` (covered by `test_returns_none_when_all_flags_unset`) and `RetrievalService` runs the single-query path.

---

## Self-Review Notes (coverage map)

| Spec section | Task(s) |
|---|---|
| M5 AsyncQueryTransformPipeline | 1, 2 |
| M5 CachedQueryTransformPipeline | 3 |
| M5 composition / from_env | 4 |
| M6 MultiQueryGenerator | 5 |
| M6 weighted RRF | 6 |
| M6 semantic dedup | 7 |
| M7 QueryRouter + heuristic fallback | 8 |
| M7 RoutedQueryTransformPipeline | 9 |
| M7 train_query_router.py | 10 |
| M8 QueryTransformBenchmark | 11 |
| M8 eval_runner + --qt-slo-ms | 12 |
| M8 richer QueryConstructor | 13 |
| Regression: all QT_* off ⇒ identical | 4 (factory test), Final verification |
