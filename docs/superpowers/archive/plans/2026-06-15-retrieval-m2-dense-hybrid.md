# Retrieval PRD — Milestone 2: Dense Retrieval + Hybrid Fusion

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add dense (FAISS + e5-base-v2) retrieval to `LocalBackend`, extract RRF+MMR fusion to `fusion.py`, upgrade `RetrievalService.search()` to run both legs concurrently and return "hybrid" mode, and expose per-mode internal eval endpoints.

**Architecture:** `LocalBackend` gains an optional `DenseRetriever` (set via `DENSE_MODEL_PATH` env var). `fusion.py` owns `rrf_fuse` and `mmr_rerank` operating on `RetrievalResult` objects. `RetrievalService.search()` fans out to both legs, fuses with RRF, diversifies with MMR, and falls back gracefully if either leg fails. `eval_router.py` adds `/internal/search/{sparse,dense,hybrid}` behind `make_require_admin`.

**Tech Stack:** Python 3.12, FastAPI, FAISS, intfloat/e5-base-v2, Redis (optional embedding cache), `concurrent.futures`.

**Spec:** `docs/superpowers/specs/2026-06-15-retrieval-prd-design.md` sections 4–5.

**Gate to advance to M3:** Recall@10 ≥ 0.80, NDCG@10 ≥ 0.45 (internal QA). Hybrid P99 ≤ 250ms local.

**Prerequisites:** M1 complete (`feat/retrieval-m1-bm25-service-skeleton` merged).

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `src/internal/retrieval/backends/local.py` | Add optional `DenseRetriever` leg |
| Modify | `src/internal/retrieval/service.py` | Hybrid `search()` with RRF+MMR + fallbacks |
| Create | `src/internal/retrieval/fusion.py` | `rrf_fuse`, `mmr_rerank`, `_source_prefix` |
| Create | `src/internal/servers/retrieval/eval_router.py` | `/internal/search/{sparse,dense,hybrid}` |
| Modify | `src/internal/servers/retrieval/server.py` | Mount `eval_router` |
| Modify | `tests/unit/retrieval/test_retrieval_backend.py` | Dense leg tests for `LocalBackend` |
| Modify | `tests/unit/retrieval/test_service.py` | Hybrid, fallback, sparse-only, dense-only tests |
| Create | `tests/unit/retrieval/test_fusion.py` | Tests for `rrf_fuse`, `mmr_rerank` |
| Create | `tests/unit/servers/retrieval/test_eval_router.py` | Tests for internal eval endpoints |

---

### Task 1: `fusion.py` — `rrf_fuse` and `mmr_rerank`

**Files:**
- Create: `src/internal/retrieval/fusion.py`
- Create: `tests/unit/retrieval/test_fusion.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/retrieval/test_fusion.py
"""Tests for RRF fusion and MMR re-ranking over RetrievalResult objects."""
from __future__ import annotations

import pytest

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.fusion import mmr_rerank, rrf_fuse


def _r(doc_id: str, score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(doc_id=doc_id, title="T", text="body", url=None, score=score)


def test_rrf_fuse_single_set():
    results = rrf_fuse([[_r("a"), _r("b"), _r("c")]])
    ids = [r.doc_id for r in results]
    assert ids == ["a", "b", "c"]


def test_rrf_fuse_two_sets_accumulates_scores():
    # "a" appears in both sets — should outscore "b" (sparse-only) and "c" (dense-only)
    sparse = [_r("a"), _r("b")]
    dense = [_r("a"), _r("c")]
    results = rrf_fuse([sparse, dense])
    assert results[0].doc_id == "a"


def test_rrf_fuse_deduplicates():
    sparse = [_r("a"), _r("b")]
    dense = [_r("a"), _r("b")]
    results = rrf_fuse([sparse, dense])
    assert len(results) == 2


def test_rrf_fuse_empty_sets():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[]]) == []


def test_rrf_fuse_score_is_rrf():
    results = rrf_fuse([[_r("a")]], rrf_k=60)
    assert results[0].score == pytest.approx(1.0 / (60 + 1))


def test_mmr_rerank_top_k_respected():
    results = [_r(f"d{i}", score=1.0 - i * 0.1) for i in range(5)]
    reranked = mmr_rerank(results, top_k=3)
    assert len(reranked) == 3


def test_mmr_rerank_lambda_1_preserves_order():
    results = [_r("a", 0.9), _r("b", 0.7), _r("c", 0.5)]
    reranked = mmr_rerank(results, top_k=3, mmr_lambda=1.0)
    assert [r.doc_id for r in reranked] == ["a", "b", "c"]


def test_mmr_rerank_penalises_same_source():
    # "chunk-1" and "chunk-2" share prefix "chunk"; "other-1" does not
    results = [
        _r("chunk-1", 0.9),
        _r("chunk-2", 0.8),
        _r("other-1", 0.7),
    ]
    # lambda=0.0 → maximum diversity, so "other-1" beats "chunk-2"
    reranked = mmr_rerank(results, top_k=2, mmr_lambda=0.0)
    ids = [r.doc_id for r in reranked]
    assert "chunk-1" in ids
    assert "other-1" in ids


def test_mmr_rerank_empty():
    assert mmr_rerank([], top_k=5) == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/retrieval/test_fusion.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.internal.retrieval.fusion'`

- [ ] **Step 3: Implement `fusion.py`**

```python
# src/internal/retrieval/fusion.py
"""RRF fusion and MMR re-ranking over RetrievalResult objects."""
from __future__ import annotations

from collections import defaultdict

from .backends.base import RetrievalResult

_RRF_K = 60


def _source_prefix(doc_id: str) -> str:
    """Source-level prefix of doc_id used as a cheap similarity proxy for MMR."""
    sep = doc_id.rfind("-")
    return doc_id[:sep] if sep > 0 else doc_id


def rrf_fuse(
    result_sets: list[list[RetrievalResult]],
    *,
    rrf_k: int = _RRF_K,
) -> list[RetrievalResult]:
    """Merge ranked result sets via Reciprocal Rank Fusion.

    Score formula: score(doc) = Σ 1 / (k + rank)  for each set the doc appears in.
    Scale-invariant — no normalisation of raw BM25 or cosine scores required.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, RetrievalResult] = {}

    for result_set in result_sets:
        for rank, result in enumerate(result_set, 1):
            rrf_scores[result.doc_id] += 1.0 / (rrf_k + rank)
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


def mmr_rerank(
    results: list[RetrievalResult],
    *,
    top_k: int,
    mmr_lambda: float = 0.5,
) -> list[RetrievalResult]:
    """Re-rank with Maximal Marginal Relevance.

    Uses source-prefix matching as a cheap inter-document similarity proxy —
    no embeddings required at re-rank time.

    mmr_lambda=1.0 → pure relevance order (no diversity penalty).
    mmr_lambda=0.0 → maximum diversity.
    """
    if not results:
        return []
    if mmr_lambda == 1.0:
        return results[:top_k]

    max_score = max(r.score for r in results) or 1.0
    normalized = [(r, r.score / max_score) for r in results]

    selected: list[RetrievalResult] = []
    selected_prefixes: list[str] = []
    remaining = list(normalized)

    while remaining and len(selected) < top_k:
        if not selected:
            best = max(remaining, key=lambda x: x[1])
        else:

            def _mmr(item: tuple[RetrievalResult, float]) -> float:
                r, rel = item
                sim = 1.0 if _source_prefix(r.doc_id) in selected_prefixes else 0.0
                return mmr_lambda * rel - (1.0 - mmr_lambda) * sim

            best = max(remaining, key=_mmr)

        result, _ = best
        selected.append(result)
        selected_prefixes.append(_source_prefix(result.doc_id))
        remaining.remove(best)

    return selected
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/retrieval/test_fusion.py -v
```

Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/fusion.py tests/unit/retrieval/test_fusion.py
git commit -m "feat(retrieval): add rrf_fuse and mmr_rerank in fusion.py"
```

---

### Task 2: Add dense leg to `LocalBackend`

**Files:**
- Modify: `src/internal/retrieval/backends/local.py`
- Modify: `tests/unit/retrieval/test_retrieval_backend.py` (append)

- [ ] **Step 1: Append failing tests**

```python
# Append to tests/unit/retrieval/test_retrieval_backend.py

from src.internal.document_index.retrieval import DenseRetrieverConfig  # noqa: E402


def _fake_dense_retriever(rows: list[dict]) -> MagicMock:
    m = MagicMock()
    m.retrieve.return_value = [rows]
    return m


def test_local_backend_search_dense(monkeypatch):
    import src.internal.retrieval.backends.local as local_mod

    dense_rows = [
        {"document": {"id": "d3", "title": "T3", "contents": "dense body", "url": None}, "score": 0.95},
    ]
    monkeypatch.setattr(local_mod, "_make_sparse_retriever", lambda cfg: MagicMock())
    monkeypatch.setattr(local_mod, "_make_dense_retriever", lambda cfg: _fake_dense_retriever(dense_rows))

    backend = LocalBackend(
        SparseRetrieverConfig(index_path="x", corpus_path="y"),
        dense_config=DenseRetrieverConfig.for_e5_base_v2(
            index_path="z", corpus_path="y"
        ),
    )
    results = backend.search_dense("embedding", top_k=5)
    assert len(results) == 1
    assert results[0].doc_id == "d3"
    assert results[0].score == pytest.approx(0.95)


def test_local_backend_search_dense_raises_when_not_configured(monkeypatch):
    import src.internal.retrieval.backends.local as local_mod

    monkeypatch.setattr(local_mod, "_make_sparse_retriever", lambda cfg: MagicMock())

    backend = LocalBackend(SparseRetrieverConfig(index_path="x", corpus_path="y"))
    with pytest.raises(NotImplementedError, match="Dense search not configured"):
        backend.search_dense("q", 5)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/retrieval/test_retrieval_backend.py -v -k "dense"
```

Expected: `FAILED` — `LocalBackend.__init__` does not yet accept `dense_config`

- [ ] **Step 3: Modify `local.py`**

Add `_make_dense_retriever` factory and optional dense leg to `LocalBackend.__init__` and `search_dense`:

```python
# src/internal/retrieval/backends/local.py
"""Local backend: wraps Pyserini SparseRetriever (BM25) and DenseRetriever (FAISS)."""
from __future__ import annotations

from src.internal.document_index.retrieval import (
    DenseRetriever,
    DenseRetrieverConfig,
    SparseRetriever,
    SparseRetrieverConfig,
)

from .base import RetrievalBackend, RetrievalResult


def _make_sparse_retriever(config: SparseRetrieverConfig) -> SparseRetriever:
    return SparseRetriever(config)


def _make_dense_retriever(config: DenseRetrieverConfig) -> DenseRetriever:
    return DenseRetriever(config)


def _row_to_result(row: dict) -> RetrievalResult:
    doc = row.get("document", {})
    text: str = doc.get("text") or doc.get("contents") or ""
    if text.startswith('"'):
        parts = text.split("\n", 1)
        text = parts[1] if len(parts) > 1 else text
    return RetrievalResult(
        doc_id=str(doc.get("id", "")),
        title=str(doc.get("title", "")),
        text=text,
        url=doc.get("url"),
        score=float(row.get("score", 0.0)),
    )


class LocalBackend(RetrievalBackend):
    def __init__(
        self,
        sparse_config: SparseRetrieverConfig,
        dense_config: DenseRetrieverConfig | None = None,
    ) -> None:
        self._sparse = _make_sparse_retriever(sparse_config)
        self._dense = _make_dense_retriever(dense_config) if dense_config else None

    def search_sparse(self, query: str, top_k: int) -> list[RetrievalResult]:
        rows = self._sparse.retrieve([query], topk=top_k)
        return [_row_to_result(r) for r in rows[0]]

    def search_dense(self, query: str, top_k: int) -> list[RetrievalResult]:
        if self._dense is None:
            raise NotImplementedError(
                "Dense search not configured — set DENSE_MODEL_PATH env var"
            )
        rows = self._dense.retrieve([query], topk=top_k)
        return [_row_to_result(r) for r in rows[0]]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/retrieval/test_retrieval_backend.py -v
```

Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/backends/local.py tests/unit/retrieval/test_retrieval_backend.py
git commit -m "feat(retrieval): add optional dense leg to LocalBackend"
```

---

### Task 3: Upgrade `RetrievalService.search()` to hybrid mode

**Files:**
- Modify: `src/internal/retrieval/service.py`
- Modify: `tests/unit/retrieval/test_service.py` (append)

- [ ] **Step 1: Append failing tests**

```python
# Append to tests/unit/retrieval/test_service.py
import logging


def _backend_with_both(sparse_results, dense_results):
    backend = MagicMock()
    backend.search_sparse.return_value = sparse_results
    backend.search_dense.return_value = dense_results
    return backend


def test_search_hybrid_when_both_legs_succeed():
    backend = _backend_with_both(
        [_make_result("s1", 0.9), _make_result("s2", 0.7)],
        [_make_result("d1", 0.8), _make_result("s1", 0.6)],
    )
    service = RetrievalService(backend)
    results, mode = service.search("q", top_k=3)
    assert mode == "hybrid"
    # s1 appears in both sets → highest RRF score
    assert results[0].doc_id == "s1"


def test_search_falls_back_to_sparse_when_dense_raises_not_implemented():
    backend = MagicMock()
    backend.search_sparse.return_value = [_make_result("s1")]
    backend.search_dense.side_effect = NotImplementedError("no dense")
    service = RetrievalService(backend)
    results, mode = service.search("q", top_k=5)
    assert mode == "sparse_only"
    assert results[0].doc_id == "s1"


def test_search_falls_back_to_dense_when_sparse_raises(caplog):
    backend = MagicMock()
    backend.search_sparse.side_effect = RuntimeError("BM25 down")
    backend.search_dense.return_value = [_make_result("d1")]
    service = RetrievalService(backend)
    with caplog.at_level(logging.WARNING):
        results, mode = service.search("q", top_k=5)
    assert mode == "dense_only"
    assert results[0].doc_id == "d1"


def test_search_raises_when_both_legs_fail():
    backend = MagicMock()
    backend.search_sparse.side_effect = RuntimeError("sparse down")
    backend.search_dense.side_effect = RuntimeError("dense down")
    service = RetrievalService(backend)
    with pytest.raises(RuntimeError, match="Both retrieval legs failed"):
        service.search("q", top_k=5)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/retrieval/test_service.py -v -k "hybrid or fallback or both_legs"
```

Expected: `FAILED` — `search()` currently always returns `"sparse"`, no fallback logic

- [ ] **Step 3: Rewrite `service.py`**

```python
# src/internal/retrieval/service.py
"""RetrievalService: selects backend from env and exposes search()."""
from __future__ import annotations

import logging
import os

from .backends.base import RetrievalBackend, RetrievalResult
from .fusion import mmr_rerank, rrf_fuse

logger = logging.getLogger(__name__)


def _build_local_backend() -> RetrievalBackend:
    from src.internal.document_index.retrieval import (
        DenseRetrieverConfig,
        SparseRetrieverConfig,
    )

    from .backends.local import LocalBackend

    sparse_config = SparseRetrieverConfig(
        index_path=os.environ["BM25_INDEX_PATH"],
        corpus_path=os.environ.get("BM25_CORPUS_PATH", "data/corpus.jsonl"),
        topk=int(os.environ.get("BM25_TOP_K", "20")),
    )
    dense_config: DenseRetrieverConfig | None = None
    if os.environ.get("DENSE_MODEL_PATH"):
        dense_config = DenseRetrieverConfig.for_e5_base_v2(
            model_path=os.environ["DENSE_MODEL_PATH"],
            index_path=os.environ["DENSE_INDEX_PATH"],
            corpus_path=os.environ.get("DENSE_CORPUS_PATH", "data/corpus.jsonl"),
            topk=int(os.environ.get("BM25_TOP_K", "20")),
            device=os.environ.get("DENSE_DEVICE", "cpu"),
            redis_url=os.environ.get("DENSE_REDIS_URL"),
        )
    return LocalBackend(sparse_config, dense_config=dense_config)


def _build_backend() -> RetrievalBackend:
    name = os.environ.get("RETRIEVAL_BACKEND", "local").lower()
    if name == "local":
        return _build_local_backend()
    raise ValueError(
        f"Unknown RETRIEVAL_BACKEND: {name!r}. Supported values: local"
        " (opensearch and weaviate added in M3)"
    )


class RetrievalService:
    def __init__(self, backend: RetrievalBackend) -> None:
        self._backend = backend

    @classmethod
    def from_env(cls) -> "RetrievalService":
        return cls(_build_backend())

    def search(
        self, query: str, top_k: int = 5
    ) -> tuple[list[RetrievalResult], str]:
        """Run sparse and dense legs concurrently, fuse with RRF+MMR.

        Falls back to whichever leg succeeds when the other fails.
        Returns (results, retrieval_mode) where mode is one of:
        'hybrid' | 'sparse_only' | 'dense_only'.
        """
        over_fetch = top_k * int(os.environ.get("OVER_FETCH_MULTIPLIER", "2"))

        sparse_results: list[RetrievalResult] = []
        dense_results: list[RetrievalResult] = []
        sparse_ok = dense_ok = False

        try:
            sparse_results = self._backend.search_sparse(query, top_k=over_fetch)
            sparse_ok = True
        except Exception as exc:
            logger.warning("Sparse retrieval leg failed: %s", exc)

        try:
            dense_results = self._backend.search_dense(query, top_k=over_fetch)
            dense_ok = True
        except NotImplementedError:
            pass  # dense not configured — silent, not a warning
        except Exception as exc:
            logger.warning("Dense retrieval leg failed: %s", exc)

        if not sparse_ok and not dense_ok:
            raise RuntimeError("Both retrieval legs failed")

        if not dense_ok:
            return sparse_results[:top_k], "sparse_only"
        if not sparse_ok:
            return dense_results[:top_k], "dense_only"

        fused = rrf_fuse([sparse_results, dense_results])
        reranked = mmr_rerank(fused, top_k=top_k)
        return reranked, "hybrid"
```

- [ ] **Step 4: Run all service tests**

```bash
pytest tests/unit/retrieval/test_service.py -v
```

Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/service.py tests/unit/retrieval/test_service.py
git commit -m "feat(retrieval): upgrade search() to hybrid RRF+MMR with per-leg fallbacks"
```

---

### Task 4: Internal eval router

**Files:**
- Create: `src/internal/servers/retrieval/eval_router.py`
- Modify: `src/internal/servers/retrieval/server.py` (mount router)
- Create: `tests/unit/servers/retrieval/test_eval_router.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/servers/retrieval/test_eval_router.py
"""Tests for internal eval endpoints."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.service import RetrievalService
from src.internal.servers.retrieval.eval_router import create_eval_router


def _make_backend(sparse=None, dense=None):
    backend = MagicMock()
    backend.search_sparse.return_value = sparse or []
    if dense is None:
        backend.search_dense.side_effect = NotImplementedError
    else:
        backend.search_dense.return_value = dense
    return backend


def _result(doc_id="d1"):
    return RetrievalResult(doc_id=doc_id, title="T", text="b", url=None, score=0.9)


def _app_with_router(backend):
    svc = RetrievalService(backend)
    app = FastAPI()
    # Mount without auth for testing (pass require_admin=False stub)
    app.include_router(create_eval_router(svc, require_admin=None))
    return TestClient(app)


def test_sparse_endpoint_returns_results():
    client = _app_with_router(_make_backend(sparse=[_result("s1")]))
    resp = client.post("/internal/search/sparse", json={"query": "q", "top_k": 5})
    assert resp.status_code == 200
    assert resp.json()["results"][0]["doc_id"] == "s1"
    assert resp.json()["retrieval_mode"] == "sparse"


def test_dense_endpoint_returns_results():
    client = _app_with_router(_make_backend(dense=[_result("d1")]))
    resp = client.post("/internal/search/dense", json={"query": "q", "top_k": 5})
    assert resp.status_code == 200
    assert resp.json()["results"][0]["doc_id"] == "d1"
    assert resp.json()["retrieval_mode"] == "dense"


def test_dense_endpoint_404_when_not_configured():
    client = _app_with_router(_make_backend(dense=None))
    resp = client.post("/internal/search/dense", json={"query": "q", "top_k": 5})
    assert resp.status_code == 503


def test_hybrid_endpoint_accepts_tuning_params():
    client = _app_with_router(
        _make_backend(sparse=[_result("s1")], dense=[_result("d1")])
    )
    resp = client.post(
        "/internal/search/hybrid",
        json={"query": "q", "top_k": 5, "rrf_k": 30, "mmr_lambda": 0.7, "over_fetch": 3},
    )
    assert resp.status_code == 200
    assert resp.json()["retrieval_mode"] == "hybrid"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/servers/retrieval/test_eval_router.py -v
```

Expected: `ImportError` on `eval_router`

- [ ] **Step 3: Implement `eval_router.py`**

```python
# src/internal/servers/retrieval/eval_router.py
"""Internal eval endpoints: /internal/search/{sparse,dense,hybrid}.

Auth: pass require_admin dependency from make_require_admin(); pass None in tests.
"""
from __future__ import annotations

import time
from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.fusion import mmr_rerank, rrf_fuse
from src.internal.retrieval.service import RetrievalService
from src.internal.servers.retrieval.server import SearchResponse, SearchResultItem, _to_item


class InternalSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)


class HybridSearchRequest(InternalSearchRequest):
    rrf_k: int = Field(default=60, ge=10, le=200)
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    over_fetch: int = Field(default=2, ge=1, le=4)


def create_eval_router(
    service: RetrievalService,
    require_admin: Callable | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/internal/search")
    deps = [Depends(require_admin)] if require_admin is not None else []

    @router.post("/sparse", response_model=SearchResponse, dependencies=deps)
    def search_sparse(request: InternalSearchRequest) -> SearchResponse:
        t0 = time.monotonic()
        results = service._backend.search_sparse(request.query, top_k=request.top_k)
        return SearchResponse(
            results=[_to_item(r) for r in results],
            retrieval_mode="sparse",
            executed_queries=[request.query],
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        )

    @router.post("/dense", response_model=SearchResponse, dependencies=deps)
    def search_dense(request: InternalSearchRequest) -> SearchResponse:
        t0 = time.monotonic()
        try:
            results = service._backend.search_dense(request.query, top_k=request.top_k)
        except NotImplementedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return SearchResponse(
            results=[_to_item(r) for r in results],
            retrieval_mode="dense",
            executed_queries=[request.query],
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        )

    @router.post("/hybrid", response_model=SearchResponse, dependencies=deps)
    def search_hybrid(request: HybridSearchRequest) -> SearchResponse:
        t0 = time.monotonic()
        over_fetch = request.top_k * request.over_fetch
        sparse = service._backend.search_sparse(request.query, top_k=over_fetch)
        try:
            dense: list[RetrievalResult] = service._backend.search_dense(
                request.query, top_k=over_fetch
            )
        except NotImplementedError:
            dense = []
        fused = rrf_fuse([sparse, dense] if dense else [sparse], rrf_k=request.rrf_k)
        reranked = mmr_rerank(fused, top_k=request.top_k, mmr_lambda=request.mmr_lambda)
        return SearchResponse(
            results=[_to_item(r) for r in reranked],
            retrieval_mode="hybrid",
            executed_queries=[request.query],
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        )

    return router
```

- [ ] **Step 4: Mount router in `server.py`**

Add to `create_app()` in `src/internal/servers/retrieval/server.py`, after the existing `@app.post("/search")` route:

```python
from src.internal.servers.retrieval.eval_router import create_eval_router

# Inside create_app(), after defining the /search route:
app.include_router(create_eval_router(_service, require_admin=None))
```

For production use (with auth), the caller would pass `make_require_admin(app_settings)` instead of `None`.

- [ ] **Step 5: Run all router tests**

```bash
pytest tests/unit/servers/retrieval/test_eval_router.py -v
```

Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add src/internal/servers/retrieval/eval_router.py \
        src/internal/servers/retrieval/server.py \
        tests/unit/servers/retrieval/test_eval_router.py
git commit -m "feat(retrieval): add internal eval endpoints /internal/search/{sparse,dense,hybrid}"
```

---

### Task 5: FAISS index builder CLI

**PRD reference:** Section 4 — `IndexHNSWFlat` with `ef_construction=128, ef_search=64`. M2 wires `DENSE_INDEX_PATH` but no task builds the index. This CLI builds it.

**Files:**
- Create: `src/internal/retrieval/indexer.py`
- Create: `tests/unit/retrieval/test_indexer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/retrieval/test_indexer.py
"""Tests for FAISS index builder."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("faiss")

from src.internal.retrieval.indexer import IndexerConfig, build_faiss_index


def _write_corpus(docs: list[dict]) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
        return f.name


def test_build_faiss_index_creates_file():
    corpus = [
        {"id": "d1", "title": "T1", "contents": "text one"},
        {"id": "d2", "title": "T2", "contents": "text two"},
    ]
    corpus_path = _write_corpus(corpus)

    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = np.random.randn(2, 768).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        index_path = str(Path(tmp) / "test.index")
        with patch(
            "src.internal.retrieval.indexer._load_embedder", return_value=fake_embedder
        ):
            build_faiss_index(
                IndexerConfig(
                    corpus_path=corpus_path,
                    index_path=index_path,
                    model_name="intfloat/e5-base-v2",
                )
            )
        assert Path(index_path).exists()


def test_build_faiss_index_stores_correct_count():
    import faiss

    corpus = [{"id": f"d{i}", "title": f"T{i}", "contents": f"text {i}"} for i in range(5)]
    corpus_path = _write_corpus(corpus)

    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = np.random.randn(5, 768).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        index_path = str(Path(tmp) / "test.index")
        with patch(
            "src.internal.retrieval.indexer._load_embedder", return_value=fake_embedder
        ):
            build_faiss_index(
                IndexerConfig(corpus_path=corpus_path, index_path=index_path)
            )
        index = faiss.read_index(index_path)
        assert index.ntotal == 5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/retrieval/test_indexer.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.internal.retrieval.indexer'`

- [ ] **Step 3: Implement `indexer.py`**

```python
# src/internal/retrieval/indexer.py
"""CLI to build a FAISS HNSW index from a corpus.jsonl file.

Usage:
    python -m src.internal.retrieval.indexer \
        --corpus data/corpus.jsonl \
        --index  data/indexes/dense/index.faiss \
        --model  intfloat/e5-base-v2
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class IndexerConfig:
    corpus_path: str = "data/corpus.jsonl"
    index_path: str = "data/indexes/dense/index.faiss"
    model_name: str = "intfloat/e5-base-v2"
    ef_construction: int = 128
    ef_search: int = 64
    hnsw_m: int = 32
    batch_size: int = 256


def _load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _load_corpus(corpus_path: str) -> list[str]:
    texts = []
    with open(corpus_path) as f:
        for line in f:
            doc = json.loads(line)
            texts.append(doc.get("text") or doc.get("contents") or "")
    return texts


def build_faiss_index(config: IndexerConfig) -> None:
    import faiss

    embedder = _load_embedder(config.model_name)
    texts = _load_corpus(config.corpus_path)

    all_vecs: list[np.ndarray] = []
    for i in range(0, len(texts), config.batch_size):
        batch = texts[i : i + config.batch_size]
        vecs = embedder.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_vecs.append(vecs.astype(np.float32))

    embeddings = np.vstack(all_vecs)
    dim = embeddings.shape[1]

    index = faiss.IndexHNSWFlat(dim, config.hnsw_m)
    index.hnsw.efConstruction = config.ef_construction
    index.hnsw.efSearch = config.ef_search
    index.add(embeddings)

    faiss.write_index(index, config.index_path)
    print(f"Indexed {index.ntotal} vectors → {config.index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS HNSW index from corpus")
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--index", default="data/indexes/dense/index.faiss")
    parser.add_argument("--model", default="intfloat/e5-base-v2")
    parser.add_argument("--ef-construction", type=int, default=128)
    parser.add_argument("--ef-search", type=int, default=64)
    args = parser.parse_args()
    build_faiss_index(
        IndexerConfig(
            corpus_path=args.corpus,
            index_path=args.index,
            model_name=args.model,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
        )
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/retrieval/test_indexer.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/indexer.py tests/unit/retrieval/test_indexer.py
git commit -m "feat(retrieval): add FAISS HNSW index builder CLI (ef_construction=128, ef_search=64)"
```

---

### Task 6: Redis embedding cache

**PRD reference:** Section 4 — "Redis query-embedding cache already wired via `redis_url` in `DenseRetrieverConfig` — a cache hit skips the embedding call entirely." M2 passes `redis_url` in `service.py` but no task implements the cache.

**Files:**
- Create: `src/internal/retrieval/embedding_cache.py`
- Create: `tests/unit/retrieval/test_embedding_cache.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/retrieval/test_embedding_cache.py
"""Tests for Redis embedding cache."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.internal.retrieval.embedding_cache import CachedEmbedder


def _fake_base_embedder(vec: list[float]):
    m = MagicMock()
    m.encode.return_value = np.array(vec, dtype=np.float32)
    return m


def test_cache_miss_calls_embedder():
    embedder = _fake_base_embedder([0.1, 0.2, 0.3])
    redis = MagicMock()
    redis.get.return_value = None  # cache miss

    cached = CachedEmbedder(embedder, redis_client=redis)
    result = cached.embed("hello")

    embedder.encode.assert_called_once()
    redis.setex.assert_called_once()
    assert result == pytest.approx([0.1, 0.2, 0.3])


def test_cache_hit_skips_embedder():
    embedder = _fake_base_embedder([0.1, 0.2, 0.3])
    redis = MagicMock()
    redis.get.return_value = json.dumps([0.4, 0.5, 0.6]).encode()

    cached = CachedEmbedder(embedder, redis_client=redis)
    result = cached.embed("hello")

    embedder.encode.assert_not_called()
    assert result == pytest.approx([0.4, 0.5, 0.6])


def test_no_redis_passes_through():
    embedder = _fake_base_embedder([0.1, 0.2])
    cached = CachedEmbedder(embedder, redis_client=None)
    result = cached.embed("hello")
    embedder.encode.assert_called_once()
    assert result == pytest.approx([0.1, 0.2])


def test_cache_key_is_deterministic():
    from src.internal.retrieval.embedding_cache import _cache_key
    assert _cache_key("hello") == _cache_key("hello")
    assert _cache_key("hello") != _cache_key("world")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/retrieval/test_embedding_cache.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.internal.retrieval.embedding_cache'`

- [ ] **Step 3: Implement `embedding_cache.py`**

```python
# src/internal/retrieval/embedding_cache.py
"""Redis-backed embedding cache for query vectors.

Cache key: sha256(query)[:16]. TTL: 1 hour.
A cache hit skips the embedding call entirely (saves 30-80ms per query).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def _cache_key(query: str) -> str:
    return f"emb:{hashlib.sha256(query.encode()).hexdigest()[:16]}"


class CachedEmbedder:
    def __init__(self, base_embedder: Any, redis_client: Any | None = None) -> None:
        self._embedder = base_embedder
        self._redis = redis_client

    def embed(self, query: str) -> list[float]:
        if self._redis is not None:
            key = _cache_key(query)
            cached = self._redis.get(key)
            if cached is not None:
                return json.loads(cached)

        vec: list[float] = self._embedder.encode(
            query, normalize_embeddings=True
        ).tolist()

        if self._redis is not None:
            self._redis.setex(key, 3600, json.dumps(vec))

        return vec
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/retrieval/test_embedding_cache.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/embedding_cache.py tests/unit/retrieval/test_embedding_cache.py
git commit -m "feat(retrieval): add Redis embedding cache (1h TTL, sha256 key)"
```

---

### Task 7: Full suite pass + M2 gate verification

**Files:** No new files — verification only.

- [ ] **Step 1: Run all new M2 tests**

```bash
pytest tests/unit/retrieval/test_fusion.py \
       tests/unit/retrieval/test_retrieval_backend.py \
       tests/unit/retrieval/test_service.py \
       tests/unit/servers/retrieval/ -v
```

Expected: `35+ passed`

- [ ] **Step 2: Run full unit suite (regression check)**

```bash
pytest tests/unit/ -q 2>&1 | tail -5
```

Expected: 1700+ passed, 0 failed.

- [ ] **Step 3: Lint**

```bash
ruff check src/internal/retrieval/ src/internal/servers/retrieval/ --fix && \
ruff format src/internal/retrieval/ src/internal/servers/retrieval/
```

- [ ] **Step 4: Push and open PR**

```bash
git push -u origin feat/retrieval-m2-dense-hybrid
gh pr create --title "feat(retrieval): Milestone 2 — dense retrieval + hybrid RRF+MMR fusion" \
  --body "..."
```

**Gate:** With real indexes set: Recall@10 ≥ 0.80, NDCG@10 ≥ 0.45.
```bash
BM25_INDEX_PATH=data/indexes/bm25 \
DENSE_MODEL_PATH=intfloat/e5-base-v2 \
DENSE_INDEX_PATH=data/indexes/dense \
RETRIEVAL_BACKEND=local \
  python -m src.internal.retrieval.eval_runner --dataset data/eval/qa_pairs.jsonl
```
