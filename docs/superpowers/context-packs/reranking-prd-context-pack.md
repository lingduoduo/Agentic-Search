# Generated Context Pack

# Reranking Prd

## Sources

- [Specification: 2026-06-16-reranking-prd-design.md](../specs/2026-06-16-reranking-prd-design.md)
- [Plan: 2026-06-16-reranking-prd.md](../plans/2026-06-16-reranking-prd.md)

## Specification Context

### Out of Scope

- Replacing the existing standalone `POST /rerank` server (kept for the web-app layer)
- Training or fine-tuning reranker models
- Streaming reranked results
- Reranking at the agent loop level (only retrieval-service level)

---

### 2. Architecture

```
POST /search
     │
     ▼
RetrievalService.search(query, top_k, filters)
  ├── ThreadPoolExecutor: sparse leg + dense leg       [M1–M4, unchanged]
  ├── RRF fusion                                       [M2, unchanged]
  ├── MMR rerank (diversity)                           [M2, unchanged]
  └── [optional] Reranker.rerank(query, candidates, top_k)
              │
              ├── provider="local"
              │   └── SentenceTransformerReranker      [existing rerank.py, reused]
              │       (BAAI/bge-reranker-v2-m3,
              │        BAAI/bge-reranker-base,
              │        cross-encoder/ms-marco-*)
              │
              └── provider="cohere"
                  └── cohere_rerank_api()              [existing search_nlp_models.py, reused]
                      (rerank-english-v3.0,
                       rerank-multilingual-v3.0)
```

The `Reranker` is constructed once at startup via `Reranker.from_env()` and injected into `RetrievalService`. When `RERANKER_PROVIDER` is unset, `from_env()` returns `None` and the service skips reranking entirely — zero overhead for callers that don't need it.

The `retrieval_mode` field in `SearchResponse` gains a `+reranked` suffix when reranking ran (e.g. `"hybrid+reranked"`, `"sparse_only+reranked"`).

---

### 9. Testing Strategy

- **Unit:** `test_reranker.py` monkeypatches `SentenceTransformerReranker.load` and `cohere_rerank_api` — no model downloads in CI
- **Unit:** `test_service.py` injects a `MagicMock` reranker; asserts `mode` suffix and that `rerank()` is called with correct args
- **Smoke:** One test asserts local reranking 20 candidates completes in < 5s (very generous; gate is 800ms measured on real hardware via eval_runner)
- **Eval gate:** Run manually via `eval_runner.py` against `data/eval/qa_pairs.jsonl`

## Implementation Plan Context

### Task 1: `RerankerConfig` + `Reranker`

**Files:**
- Create: `src/internal/retrieval/reranker.py`
- Create: `tests/unit/retrieval/test_reranker.py`

**Background:** `SentenceTransformerReranker` lives in `src/internal/servers/retrieval/rerank.py`. Its `rerank(queries, documents, topk)` method takes `list[str]` queries and `list[list[dict]]` documents where each doc dict is plain JSON (e.g. `{"contents": "title\nbody", "doc_id": "x"}`), and returns `list[list[dict]]` where each item is `{"document": original_dict, "score": float}`. `cohere_rerank_api(query, docs, model_name, api_key)` in `src/internal/natural_language_processing/search_nlp_models.py` is `async` and returns `list[float]` (one score per passage, preserving input order). `RetrievalResult` is a mutable dataclass in `src/internal/retrieval/backends/base.py` with fields: `doc_id: str`, `title: str`, `text: str`, `url: str | None`, `score: float`, `metadata: dict`.

- [ ] **Step 1: Write the failing tests**

```python

### tests/unit/retrieval/test_reranker.py

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

### Task 2: Wire `Reranker` into `RetrievalService`

**Files:**
- Modify: `src/internal/retrieval/service.py` (lines 77–136 — `RetrievalService` class)
- Modify: `tests/unit/retrieval/test_service.py`

**Background:** `RetrievalService.__init__` currently takes `backend: RetrievalBackend`. `search()` returns `(results, mode)` where mode is `"hybrid"` | `"sparse_only"` | `"dense_only"`. After MMR reranking, add an optional neural rerank step. `from_env()` calls `_build_backend()` — extend it to also call `Reranker.from_env()`.

- [ ] **Step 1: Write failing tests**

```python

### Append to tests/unit/retrieval/test_service.py

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

_[Section compacted.]_

### Task 3: Extend `eval_runner.py` with `--reranker` flag and latency

**Files:**
- Modify: `src/internal/retrieval/eval_runner.py`
- Modify: `tests/unit/retrieval/test_eval_runner.py`

**Background:** `run_eval(dataset_path, service, top_k)` currently returns `{recall@k, ndcg@k, mrr, num_queries}`. Extend it to accept an optional `reranker: Reranker | None`. When provided, run reranking after retrieval, compute the same metrics on reranked results, measure wall-clock time per rerank call, and return a structured dict with `retrieval`, `reranked`, and `latency_ms` keys. The CLI gains `--reranker {local,cohere}` and `--reranker_model` args.

- [ ] **Step 1: Write failing tests**

```python

### Append to tests/unit/retrieval/test_eval_runner.py

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

_[Section compacted.]_

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
