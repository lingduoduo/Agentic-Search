# Reranking PRD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single `Reranker` class (local BGE/cross-encoder + Cohere) as an optional post-fusion step inside `RetrievalService`, and extend `eval_runner` to benchmark reranking quality and latency.

**Architecture:** `Reranker` (in `src/internal/retrieval/reranker.py`) wraps the existing `SentenceTransformerReranker` (local) and `cohere_rerank_api()` (Cohere) behind one `rerank(query, results, top_k)` interface. `RetrievalService` accepts an optional `reranker` param; when set, it calls `reranker.rerank()` after RRF+MMR and appends `+reranked` to the mode string. `eval_runner.py` gains a `--reranker` flag that measures NDCG/MRR before and after reranking plus per-query latency percentiles.

**Tech Stack:** Python 3.12, sentence-transformers (`CrossEncoder`), Cohere async SDK, asyncio, dataclasses, pytest.

**Spec:** `docs/superpowers/specs/2026-06-16-reranking-prd-design.md`

**Gate:** `python -m src.internal.retrieval.eval_runner --dataset data/eval/qa_pairs.jsonl --reranker local` reports `reranked.ndcg@10 >= 0.50`, `reranked.mrr >= 0.65`, `latency_ms.p99 <= 800`.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| **Create** | `src/internal/retrieval/reranker.py` | `RerankerConfig` + `Reranker` (local + Cohere dispatch) |
| **Modify** | `src/internal/retrieval/service.py` | Accept optional `reranker`; call after MMR; `from_env()` extension |
| **Modify** | `src/internal/retrieval/eval_runner.py` | `--reranker` / `--reranker_model` flags; latency measurement |
| **Create** | `tests/unit/retrieval/test_reranker.py` | Config validation; local path; Cohere path; `from_env` |
| **Modify** | `tests/unit/retrieval/test_service.py` | Reranker injection; `+reranked` mode suffix |
| **Modify** | `tests/unit/retrieval/test_eval_runner.py` | Reranker flag; latency output |

**Not changed:** `src/internal/servers/retrieval/rerank.py`, `search_nlp_models.py`, `eval_metrics.py`, any existing retrieval backend.

---

### Task 1: `RerankerConfig` + `Reranker`

**Files:**
- Create: `src/internal/retrieval/reranker.py`
- Create: `tests/unit/retrieval/test_reranker.py`

**Background:** `SentenceTransformerReranker` lives in `src/internal/servers/retrieval/rerank.py`. Its `rerank(queries, documents, topk)` method takes `list[str]` queries and `list[list[dict]]` documents where each doc dict is plain JSON (e.g. `{"contents": "title\nbody", "doc_id": "x"}`), and returns `list[list[dict]]` where each item is `{"document": original_dict, "score": float}`. `cohere_rerank_api(query, docs, model_name, api_key)` in `src/internal/natural_language_processing/search_nlp_models.py` is `async` and returns `list[float]` (one score per passage, preserving input order). `RetrievalResult` is a mutable dataclass in `src/internal/retrieval/backends/base.py` with fields: `doc_id: str`, `title: str`, `text: str`, `url: str | None`, `score: float`, `metadata: dict`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/retrieval/test_reranker.py
"""Tests for Reranker (local + Cohere dispatch)."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

import pytest

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.reranker import Reranker, RerankerConfig


def _result(doc_id: str, score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(
        doc_id=doc_id, title=f"Title {doc_id}", text=f"Body of {doc_id}",
        url=None, score=score,
    )


# --- Config validation ---

def test_config_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        RerankerConfig(provider="pinecone").validate()


def test_config_requires_api_key_for_cohere():
    with pytest.raises(ValueError, match="api_key"):
        RerankerConfig(provider="cohere", api_key=None).validate()


def test_config_rejects_zero_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        RerankerConfig(provider="local", batch_size=0).validate()


# --- Local provider ---

def test_local_reranker_reorders_by_score():
    fake_reranker = MagicMock()
    # Returns scored results: d2 higher than d1
    fake_reranker.rerank.return_value = [[
        {"document": {"contents": "Title d1\nBody of d1", "doc_id": "d1"}, "score": 0.3},
        {"document": {"contents": "Title d2\nBody of d2", "doc_id": "d2"}, "score": 0.9},
    ]]

    with patch(
        "src.internal.retrieval.reranker.SentenceTransformerReranker.load",
        return_value=fake_reranker,
    ):
        ranker = Reranker(RerankerConfig(provider="local"))
        results = ranker.rerank("query", [_result("d1", 0.8), _result("d2", 0.2)], top_k=2)

    assert results[0].doc_id == "d2"
    assert results[1].doc_id == "d1"
    assert results[0].score == pytest.approx(0.9)


def test_local_reranker_respects_top_k():
    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = [[
        {"document": {"contents": "Title d1\nBody of d1", "doc_id": "d1"}, "score": 0.9},
        {"document": {"contents": "Title d2\nBody of d2", "doc_id": "d2"}, "score": 0.7},
        {"document": {"contents": "Title d3\nBody of d3", "doc_id": "d3"}, "score": 0.5},
    ]]

    with patch(
        "src.internal.retrieval.reranker.SentenceTransformerReranker.load",
        return_value=fake_reranker,
    ):
        ranker = Reranker(RerankerConfig(provider="local"))
        results = ranker.rerank(
            "query",
            [_result("d1"), _result("d2"), _result("d3")],
            top_k=2,
        )

    assert len(results) == 2


def test_local_reranker_empty_results():
    with patch("src.internal.retrieval.reranker.SentenceTransformerReranker.load"):
        ranker = Reranker(RerankerConfig(provider="local"))
        assert ranker.rerank("q", [], top_k=5) == []


# --- Cohere provider ---

def test_cohere_reranker_reorders_by_score():
    async def fake_cohere(query, passages, model, api_key):
        # Return high score for "Body of d2", low for "Body of d1"
        return [0.2, 0.9]  # d1=0.2, d2=0.9 (preserves input order)

    with patch(
        "src.internal.retrieval.reranker.cohere_rerank_api",
        side_effect=fake_cohere,
    ):
        ranker = Reranker(RerankerConfig(
            provider="cohere",
            model="rerank-english-v3.0",
            api_key="test-key",
        ))
        results = ranker.rerank("q", [_result("d1", 0.8), _result("d2", 0.1)], top_k=2)

    assert results[0].doc_id == "d2"
    assert results[0].score == pytest.approx(0.9)


# --- from_env ---

def test_from_env_returns_none_when_provider_unset(monkeypatch):
    monkeypatch.delenv("RERANKER_PROVIDER", raising=False)
    assert Reranker.from_env() is None


def test_from_env_builds_local_reranker(monkeypatch):
    monkeypatch.setenv("RERANKER_PROVIDER", "local")
    monkeypatch.setenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    with patch("src.internal.retrieval.reranker.SentenceTransformerReranker.load"):
        ranker = Reranker.from_env()
    assert ranker is not None
    assert ranker._config.provider == "local"
    assert ranker._config.model == "BAAI/bge-reranker-base"


def test_from_env_builds_cohere_reranker(monkeypatch):
    monkeypatch.setenv("RERANKER_PROVIDER", "cohere")
    monkeypatch.setenv("RERANKER_MODEL", "rerank-english-v3.0")
    monkeypatch.setenv("COHERE_API_KEY", "ck-test")
    ranker = Reranker.from_env()
    assert ranker is not None
    assert ranker._config.provider == "cohere"
    assert ranker._config.api_key == "ck-test"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/retrieval/test_reranker.py -v
```

Expected: `ImportError` — `src.internal.retrieval.reranker` not found.

- [ ] **Step 3: Implement `src/internal/retrieval/reranker.py`**

```python
# src/internal/retrieval/reranker.py
"""Reranker: single class supporting local cross-encoders and Cohere."""

from __future__ import annotations

import asyncio
import dataclasses
import os
from dataclasses import dataclass
from typing import Literal

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.servers.retrieval.rerank import SentenceTransformerReranker


def _cohere_rerank_api():
    from src.internal.natural_language_processing.search_nlp_models import (
        cohere_rerank_api,
    )
    return cohere_rerank_api


@dataclass(frozen=True)
class RerankerConfig:
    provider: Literal["local", "cohere"]
    model: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 32
    device: str = "cpu"
    api_key: str | None = None
    top_k: int | None = None

    def validate(self) -> None:
        if self.provider not in ("local", "cohere"):
            raise ValueError(f"Unknown provider: {self.provider!r}. Use 'local' or 'cohere'.")
        if self.provider == "cohere" and not self.api_key:
            raise ValueError("api_key is required for provider='cohere'.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")


class Reranker:
    def __init__(self, config: RerankerConfig) -> None:
        config.validate()
        self._config = config
        if config.provider == "local":
            self._local = SentenceTransformerReranker.load(
                config.model,
                batch_size=config.batch_size,
                device=config.device,
            )

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Rescore results and return top_k sorted by descending score."""
        if not results:
            return results
        effective_k = self._config.top_k or top_k
        if self._config.provider == "local":
            return self._rerank_local(query, results, effective_k)
        return self._rerank_cohere(query, results, effective_k)

    def _rerank_local(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        # Embed doc_id so we can map scores back after reranking.
        # passage_to_string() uses "contents" key from the outer dict.
        docs = [
            {"contents": f"{r.title}\n{r.text}", "doc_id": r.doc_id}
            for r in results
        ]
        scored = self._local.rerank([query], [docs], topk=top_k)
        id_to_result = {r.doc_id: r for r in results}
        reranked = []
        for item in scored[0]:
            doc_id = item["document"].get("doc_id")
            if doc_id and doc_id in id_to_result:
                reranked.append(
                    dataclasses.replace(id_to_result[doc_id], score=float(item["score"]))
                )
        return reranked[:top_k]

    def _rerank_cohere(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        cohere_rerank_api = _cohere_rerank_api()
        passages = [r.text for r in results]
        scores = asyncio.run(
            cohere_rerank_api(query, passages, self._config.model, self._config.api_key)
        )
        scored = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
        return [
            dataclasses.replace(r, score=float(s)) for s, r in scored[:top_k]
        ]

    @classmethod
    def from_env(cls) -> "Reranker | None":
        """Build a Reranker from env vars. Returns None if RERANKER_PROVIDER is unset."""
        provider = os.environ.get("RERANKER_PROVIDER")
        if not provider:
            return None
        top_k_raw = os.environ.get("RERANKER_TOP_K")
        return cls(RerankerConfig(
            provider=provider,  # type: ignore[arg-type]
            model=os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            batch_size=int(os.environ.get("RERANKER_BATCH_SIZE", "32")),
            device=os.environ.get("RERANKER_DEVICE", "cpu"),
            api_key=os.environ.get("COHERE_API_KEY"),
            top_k=int(top_k_raw) if top_k_raw else None,
        ))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/retrieval/test_reranker.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/reranker.py tests/unit/retrieval/test_reranker.py
git commit -m "feat(reranking): add Reranker with local BGE and Cohere dispatch"
```

---

### Task 2: Wire `Reranker` into `RetrievalService`

**Files:**
- Modify: `src/internal/retrieval/service.py` (lines 77–136 — `RetrievalService` class)
- Modify: `tests/unit/retrieval/test_service.py`

**Background:** `RetrievalService.__init__` currently takes `backend: RetrievalBackend`. `search()` returns `(results, mode)` where mode is `"hybrid"` | `"sparse_only"` | `"dense_only"`. After MMR reranking, add an optional neural rerank step. `from_env()` calls `_build_backend()` — extend it to also call `Reranker.from_env()`.

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/unit/retrieval/test_service.py

def test_reranker_called_when_injected():
    """Reranker.rerank() must be called with the query and fused results."""
    backend = _sparse_only_backend([_make_result("d1"), _make_result("d2")])
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [_make_result("d2"), _make_result("d1")]

    service = RetrievalService(backend, reranker=mock_reranker)
    results, mode = service.search("q", top_k=2)

    mock_reranker.rerank.assert_called_once()
    call_args = mock_reranker.rerank.call_args
    assert call_args[0][0] == "q"    # query
    assert call_args[0][2] == 2      # top_k
    assert results[0].doc_id == "d2"


def test_mode_has_reranked_suffix():
    """retrieval_mode must end with '+reranked' when a reranker is present."""
    backend = _sparse_only_backend([_make_result("d1")])
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [_make_result("d1")]

    service = RetrievalService(backend, reranker=mock_reranker)
    _, mode = service.search("q", top_k=1)

    assert mode.endswith("+reranked")


def test_no_reranker_mode_unchanged():
    """Without a reranker, mode must not contain '+reranked'."""
    backend = _sparse_only_backend([_make_result("d1")])
    service = RetrievalService(backend)
    _, mode = service.search("q", top_k=1)

    assert "+reranked" not in mode


def test_reranker_receives_filters():
    """filters kwarg must be passed through to backend legs even when reranker is set."""
    backend = _sparse_only_backend([_make_result("d1")])
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [_make_result("d1")]

    service = RetrievalService(backend, reranker=mock_reranker)
    service.search("q", top_k=1, filters={"source": "wiki"})

    backend.search_sparse.assert_called_once_with(
        "q", top_k=2, filters={"source": "wiki"}
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/retrieval/test_service.py::test_reranker_called_when_injected \
       tests/unit/retrieval/test_service.py::test_mode_has_reranked_suffix \
       tests/unit/retrieval/test_service.py::test_no_reranker_mode_unchanged \
       tests/unit/retrieval/test_service.py::test_reranker_receives_filters -v
```

Expected: FAIL — `RetrievalService.__init__` does not accept `reranker` param.

- [ ] **Step 3: Modify `src/internal/retrieval/service.py`**

Change `RetrievalService.__init__` and `from_env()`, and add the rerank step at the end of `search()`. Replace the entire class (lines 77–155) with:

```python
class RetrievalService:
    def __init__(
        self,
        backend: RetrievalBackend,
        reranker: "Reranker | None" = None,
    ) -> None:
        self._backend = backend
        self._reranker = reranker

    @classmethod
    def from_env(cls) -> "RetrievalService":
        """Construct service from environment variables."""
        from src.internal.retrieval.reranker import Reranker

        return cls(_build_backend(), reranker=Reranker.from_env())

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> tuple[list[RetrievalResult], str]:
        """Run sparse and dense legs, fuse with RRF+MMR, optionally rerank.

        Returns (results, retrieval_mode) where mode is e.g.
        'hybrid' | 'sparse_only' | 'dense_only' | 'hybrid+reranked'.
        """
        over_fetch = top_k * int(os.environ.get("OVER_FETCH_MULTIPLIER", "2"))

        sparse_results: list[RetrievalResult] = []
        dense_results: list[RetrievalResult] = []
        sparse_ok = dense_ok = False

        with ThreadPoolExecutor(max_workers=2) as executor:
            sparse_future = executor.submit(
                self._backend.search_sparse, query, top_k=over_fetch, filters=filters
            )
            dense_future = executor.submit(
                self._backend.search_dense, query, top_k=over_fetch, filters=filters
            )

        try:
            sparse_results = sparse_future.result()
            sparse_ok = True
        except Exception as exc:
            logger.warning("Sparse retrieval leg failed: %s", exc)

        try:
            dense_results = dense_future.result()
            dense_ok = True
        except NotImplementedError:
            pass
        except Exception as exc:
            logger.warning("Dense retrieval leg failed: %s", exc)

        if not sparse_ok and not dense_ok:
            raise RuntimeError("Both retrieval legs failed")

        if not dense_ok:
            fused, mode = sparse_results[:top_k], "sparse_only"
        elif not sparse_ok:
            fused, mode = dense_results[:top_k], "dense_only"
        else:
            fused = rrf_fuse([sparse_results, dense_results])
            fused = mmr_rerank(fused, top_k=top_k)
            mode = "hybrid"

        if self._reranker:
            fused = self._reranker.rerank(query, fused, top_k)
            mode = f"{mode}+reranked"

        return fused, mode

    def graph_search(
        self,
        query: str,
        top_k: int = 10,
        initial_k: int = 5,
        max_entity_queries: int = 3,
    ) -> list[RetrievalResult]:
        """Graph-augmented retrieval: seed search → entity expansion → RRF fusion."""
        from .graph_rag import graph_rag_search

        return graph_rag_search(
            query,
            service=self,
            top_k=top_k,
            initial_k=initial_k,
            max_entity_queries=max_entity_queries,
        )
```

Note: `service.py` already has `from __future__ import annotations` on line 1, so the `"Reranker | None"` string annotation works without any extra import. The `Reranker` import inside `from_env()` is a lazy import — no circular-import risk because `reranker.py` does not import from `service.py`.

- [ ] **Step 4: Run all service tests to verify they pass**

```bash
pytest tests/unit/retrieval/test_service.py -v
```

Expected: all pass (previously passing + 4 new).

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/service.py tests/unit/retrieval/test_service.py
git commit -m "feat(reranking): wire Reranker into RetrievalService as optional post-fusion step"
```

---

### Task 3: Extend `eval_runner.py` with `--reranker` flag and latency

**Files:**
- Modify: `src/internal/retrieval/eval_runner.py`
- Modify: `tests/unit/retrieval/test_eval_runner.py`

**Background:** `run_eval(dataset_path, service, top_k)` currently returns `{recall@k, ndcg@k, mrr, num_queries}`. Extend it to accept an optional `reranker: Reranker | None`. When provided, run reranking after retrieval, compute the same metrics on reranked results, measure wall-clock time per rerank call, and return a structured dict with `retrieval`, `reranked`, and `latency_ms` keys. The CLI gains `--reranker {local,cohere}` and `--reranker_model` args.

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/unit/retrieval/test_eval_runner.py
from src.internal.retrieval.reranker import Reranker, RerankerConfig

def test_run_eval_with_reranker_returns_reranked_section():
    """run_eval with a reranker must return 'retrieval', 'reranked', and 'latency_ms'."""
    qa = [{"query": "q1", "relevant_doc_ids": ["d1"]}]
    path = _write_qa(qa)

    mock_reranker = MagicMock(spec=Reranker)
    mock_reranker.rerank.return_value = [
        RetrievalResult(doc_id="d1", title="", text="", url=None, score=0.99)
    ]

    svc = _make_service([["d1"]])
    result = run_eval(path, service=svc, top_k=5, reranker=mock_reranker)

    assert "retrieval" in result
    assert "reranked" in result
    assert "latency_ms" in result
    assert "ndcg@5" in result["retrieval"]
    assert "ndcg@5" in result["reranked"]
    assert "p99" in result["latency_ms"]


def test_run_eval_without_reranker_returns_flat_dict():
    """run_eval without reranker returns the existing flat dict format."""
    qa = [{"query": "q1", "relevant_doc_ids": ["d1"]}]
    path = _write_qa(qa)
    svc = _make_service([["d1"]])

    result = run_eval(path, service=svc, top_k=5)

    assert "retrieval" not in result
    assert "ndcg@5" in result


def test_run_eval_reranker_called_once_per_query():
    """reranker.rerank() must be called exactly once per QA pair."""
    qa = [
        {"query": "q1", "relevant_doc_ids": ["d1"]},
        {"query": "q2", "relevant_doc_ids": ["d2"]},
    ]
    path = _write_qa(qa)

    mock_reranker = MagicMock(spec=Reranker)
    mock_reranker.rerank.return_value = [
        RetrievalResult(doc_id="d1", title="", text="", url=None, score=0.9)
    ]

    svc = _make_service([["d1"], ["d2"]])
    run_eval(path, service=svc, top_k=5, reranker=mock_reranker)

    assert mock_reranker.rerank.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/retrieval/test_eval_runner.py::test_run_eval_with_reranker_returns_reranked_section \
       tests/unit/retrieval/test_eval_runner.py::test_run_eval_without_reranker_returns_flat_dict \
       tests/unit/retrieval/test_eval_runner.py::test_run_eval_reranker_called_once_per_query -v
```

Expected: FAIL — `run_eval` does not accept `reranker` param.

- [ ] **Step 3: Replace `src/internal/retrieval/eval_runner.py` with extended version**

```python
# src/internal/retrieval/eval_runner.py
"""CLI and library for offline retrieval evaluation.

Usage (retrieval only):
    python -m src.internal.retrieval.eval_runner \\
        --dataset data/eval/qa_pairs.jsonl --top_k 10

Usage (with reranking):
    python -m src.internal.retrieval.eval_runner \\
        --dataset data/eval/qa_pairs.jsonl --top_k 10 \\
        --reranker local --reranker_model BAAI/bge-reranker-v2-m3

QA pairs file format (one JSON object per line):
    {"query": "...", "relevant_doc_ids": ["doc-id-1", "doc-id-2"]}
"""

from __future__ import annotations

import argparse
import json
import time

from .eval_metrics import mrr as mrr_score
from .eval_metrics import ndcg_at_k, recall_at_k
from .service import RetrievalService


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = max(0, int(len(sorted_vals) * p / 100) - 1)
    return round(sorted_vals[idx], 1)


def run_eval(
    dataset_path: str,
    *,
    service: RetrievalService | None = None,
    top_k: int = 10,
    reranker=None,  # Reranker | None — avoid circular import at module level
) -> dict:
    """Load QA pairs, run retrieval (and optionally reranking), return metrics.

    Without reranker: returns flat dict {recall@k, ndcg@k, mrr, num_queries}.
    With reranker:    returns {retrieval: {...}, reranked: {...}, latency_ms: {...}}.
    """
    _service = service or RetrievalService.from_env()

    with open(dataset_path) as f:
        qa_pairs = [json.loads(line) for line in f if line.strip()]

    recalls, ndcgs, mrrs = [], [], []
    r_recalls, r_ndcgs, r_mrrs, latencies_ms = [], [], [], []

    for item in qa_pairs:
        query: str = item["query"]
        relevant: set[str] = set(item["relevant_doc_ids"])
        results, _ = _service.search(query, top_k=top_k)
        retrieved = [r.doc_id for r in results]

        recalls.append(recall_at_k(retrieved, relevant, top_k))
        ndcgs.append(ndcg_at_k(retrieved, relevant, top_k))
        mrrs.append(mrr_score(retrieved, relevant))

        if reranker is not None:
            t0 = time.monotonic()
            reranked_results = reranker.rerank(query, results, top_k)
            latencies_ms.append((time.monotonic() - t0) * 1000)
            r_retrieved = [r.doc_id for r in reranked_results]
            r_recalls.append(recall_at_k(r_retrieved, relevant, top_k))
            r_ndcgs.append(ndcg_at_k(r_retrieved, relevant, top_k))
            r_mrrs.append(mrr_score(r_retrieved, relevant))

    n = len(qa_pairs)

    def _avg(lst):
        return round(sum(lst) / n, 4) if n else 0.0

    retrieval_metrics = {
        f"recall@{top_k}": _avg(recalls),
        f"ndcg@{top_k}": _avg(ndcgs),
        "mrr": _avg(mrrs),
        "num_queries": n,
    }

    if reranker is None:
        return retrieval_metrics

    return {
        "retrieval": retrieval_metrics,
        "reranked": {
            f"recall@{top_k}": _avg(r_recalls),
            f"ndcg@{top_k}": _avg(r_ndcgs),
            "mrr": _avg(r_mrrs),
            "num_queries": n,
        },
        "latency_ms": {
            "p50": _percentile(latencies_ms, 50),
            "p95": _percentile(latencies_ms, 95),
            "p99": _percentile(latencies_ms, 99),
            "n": n,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline retrieval evaluation")
    parser.add_argument("--dataset", required=True, help="Path to qa_pairs.jsonl")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--reranker", choices=["local", "cohere"], default=None,
        help="Provider for reranking (omit to skip reranking)"
    )
    parser.add_argument(
        "--reranker_model", default="BAAI/bge-reranker-v2-m3",
        help="Model name for local reranker or Cohere model name"
    )
    args = parser.parse_args()

    reranker = None
    if args.reranker:
        import os
        from src.internal.retrieval.reranker import Reranker, RerankerConfig
        reranker = Reranker(RerankerConfig(
            provider=args.reranker,
            model=args.reranker_model,
            api_key=os.environ.get("COHERE_API_KEY"),
        ))

    metrics = run_eval(args.dataset, top_k=args.top_k, reranker=reranker)
    print(json.dumps(metrics, indent=2))
```

- [ ] **Step 4: Run all eval runner tests to verify they pass**

```bash
pytest tests/unit/retrieval/test_eval_runner.py -v
```

Expected: all pass (previously passing + 3 new).

- [ ] **Step 5: Run the full unit suite to verify nothing broke**

```bash
pytest tests/unit/ -q
```

Expected: all pass (1767+ tests).

- [ ] **Step 6: Commit**

```bash
git add src/internal/retrieval/eval_runner.py tests/unit/retrieval/test_eval_runner.py
git commit -m "feat(reranking): extend eval_runner with --reranker flag and latency percentiles"
```

---

## Running the Gate

After all tasks are complete, run against the labeled dataset:

```bash
# Local BGE (default)
RERANKER_PROVIDER=local \
RERANKER_MODEL=BAAI/bge-reranker-v2-m3 \
python -m src.internal.retrieval.eval_runner \
    --dataset data/eval/qa_pairs.jsonl \
    --top_k 10 \
    --reranker local \
    --reranker_model BAAI/bge-reranker-v2-m3
```

Expected output structure:
```json
{
  "retrieval":  {"recall@10": 0.82, "ndcg@10": 0.48, "mrr": 0.63, "num_queries": 50},
  "reranked":   {"recall@10": 0.82, "ndcg@10": 0.55, "mrr": 0.71, "num_queries": 50},
  "latency_ms": {"p50": 310.0, "p95": 580.0, "p99": 720.0, "n": 50}
}
```

Gate criteria:
- `reranked.ndcg@10 >= 0.50`
- `reranked.mrr >= 0.65`
- `latency_ms.p99 <= 800`
