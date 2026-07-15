# Reranking Optimization Design

**Date:** 2026-06-19
**Status:** Approved

## Overview

Extends the existing `Reranker` class (BGE local + Cohere remote) with latency optimizations and quality improvements through layered wrapper composition. The `Reranker` leaf is unchanged; wrappers compose on top of it.

## Architecture

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

## Milestones

### M5 — Latency Foundations

**`AsyncReranker`** (`src/internal/retrieval/async_reranker.py`)

Wraps any `Reranker`-compatible object. Offloads `rerank()` to a `ThreadPoolExecutor`. Constructor:

```python
AsyncReranker(base_reranker, *, timeout_ms: int = 500, max_workers: int = 4)
```

- `async rerank(query, results, top_k) -> list[RetrievalResult]` — awaitable entry point
- `rerank(...)` sync shim for non-async callers
- Raises `RerankerTimeoutError` if scorer exceeds `timeout_ms`; caller falls back to unranked retrieval results
- `from_env(base_reranker) -> AsyncReranker` factory reads `RERANKER_TIMEOUT_MS`, `RERANKER_MAX_WORKERS`

**`CachedReranker`** (`src/internal/retrieval/cached_reranker.py`)

Wraps `AsyncReranker` (or any reranker). Redis-backed score cache.

- Cache key: `"rrk:" + sha256(f"{query}:{sorted_doc_ids}")[:20]`
- On hit: return cached `list[RetrievalResult]` without calling scorer
- On miss: call through, serialize result, write to Redis with `ttl_seconds`
- Serialization: JSON (same as `ResultCache`)
- `stats() -> dict` — hits, misses, hit_rate
- `from_env(base_reranker) -> CachedReranker` reads `RERANKER_CACHE_REDIS_URL`, `RERANKER_CACHE_TTL_SECONDS`; returns `base_reranker` unchanged if URL not set

**`PassageTruncator`** (`src/internal/retrieval/passage_truncator.py`)

Standalone helper; not a wrapper.

```python
class PassageTruncator:
    def __init__(self, max_tokens: int = 512): ...
    def truncate(self, text: str) -> str: ...
    @staticmethod
    def from_env() -> PassageTruncator: ...  # reads RERANKER_MAX_TOKENS
```

Called inside `Reranker.rerank()` on each passage before encoding. Tokenizes with `str.split()` (whitespace approximation) for zero-dependency truncation; exact tokenizer not required here.

**`RetrievalService` integration (M5)**

`search()` gains optional `reranker` parameter (replaces the existing `self._reranker` field). When `RERANKER_ASYNC=true`, `from_env()` wraps the base reranker in `CachedReranker(AsyncReranker(base))`.

**M5 environment variables:**

| Variable | Default | Description |
|---|---|---|
| `RERANKER_ASYNC` | `false` | Enable async thread-offload |
| `RERANKER_TIMEOUT_MS` | `500` | Per-query scorer timeout |
| `RERANKER_MAX_WORKERS` | `4` | Thread pool size |
| `RERANKER_CACHE_REDIS_URL` | _(unset)_ | Enable Redis score cache |
| `RERANKER_CACHE_TTL_SECONDS` | `300` | Cache TTL |
| `RERANKER_MAX_TOKENS` | `512` | Passage truncation limit (0 = disabled) |

---

### M6 — Latency Enforcement + ONNX

**`ONNXReranker`** (`src/internal/retrieval/onnx_reranker.py`)

Drop-in replacement for the local BGE path. Loads model via `optimum.onnxruntime.ORTModelForSequenceClassification` when `RERANKER_USE_ONNX=true`. Falls back to PyTorch `Reranker` with a logged warning if `optimum` is not installed. Interface identical to `Reranker`; `AsyncReranker` wraps either without changes.

```python
class ONNXReranker:
    def __init__(self, model_name: str, *, device: str = "cpu"): ...
    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]: ...
    @staticmethod
    def from_env() -> "ONNXReranker | Reranker": ...
```

**`eval_runner` additions** (modify existing `src/internal/retrieval/eval_runner.py`)

- `--slo-ms INT` flag: if P99 reranker latency across all queries exceeds this value, exit non-zero
- Latency table in stdout: mean, P50, P90, P99 columns
- Per-query latency stored in output JSONL alongside NDCG/MRR

**M6 environment variables:**

| Variable | Default | Description |
|---|---|---|
| `RERANKER_USE_ONNX` | `false` | Load model via ONNX runtime |

---

### M7 — Quality Improvements

**`TwoStageReranker`** (`src/internal/retrieval/two_stage_reranker.py`)

Chains a fast pre-filter through all N candidates, then a heavy scorer on only the top M.

```python
class TwoStageReranker:
    def __init__(
        self,
        fast_reranker,
        heavy_reranker,
        *,
        pre_filter_top_n: int = 50,
    ): ...

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        # Step 1: fast_reranker scores all results, returns top pre_filter_top_n
        # Step 2: heavy_reranker scores those top_n, returns top_k
        ...

    async def arerank(self, query, results, top_k) -> list[RetrievalResult]: ...

    @staticmethod
    def from_env(fast_reranker, heavy_reranker) -> "TwoStageReranker": ...
```

Both `fast_reranker` and `heavy_reranker` are `AsyncReranker` instances, so each gets independent timeouts and caching. `RetrievalService.from_env()` constructs `TwoStageReranker` when `RERANKER_TWO_STAGE=true`.

**Over-fetch:** `RERANKER_OVER_FETCH_MULTIPLIER` (default 2.0). When any reranker is active, `RetrievalService._search_one()` fetches `ceil(top_k * multiplier)` candidates from retrieval, passes all to the reranker, then truncates to `top_k` after scoring. No new class required — one multiplication in `_search_one`.

**M7 environment variables:**

| Variable | Default | Description |
|---|---|---|
| `RERANKER_TWO_STAGE` | `false` | Enable two-stage pipeline |
| `RERANKER_PRE_FILTER_TOP_N` | `50` | Candidates passed to heavy scorer |
| `RERANKER_FAST_MODEL` | _(inherits RERANKER_MODEL)_ | Model name for fast stage |
| `RERANKER_OVER_FETCH_MULTIPLIER` | `2.0` | Retrieval over-fetch ratio |

---

### M8 — Benchmarking, Cohere v3, Eval Extensions

**`RerankerBenchmark` CLI** (`src/internal/retrieval/reranker_benchmark.py`)

Grid search over model × batch_size × max_tokens configurations.

```
python -m src.internal.retrieval.reranker_benchmark \
  --qa-pairs data/qa.jsonl \
  --models BAAI/bge-reranker-base BAAI/bge-reranker-large \
  --batch-sizes 8 16 32 \
  --max-tokens 256 512 \
  --output results/reranker_bench.jsonl
```

For each `(model, batch_size, max_tokens)` triple:
1. Instantiate `Reranker` with those params
2. Run all QA pairs through `rerank()`
3. Record NDCG@10, MRR, P99, mean latency per query
4. Write one JSONL row per configuration

Prints a ranked table sorted by NDCG@10 on completion. Imports `Reranker` and existing `eval_metrics` — no new classes.

**Cohere v3 document format** (modify `src/internal/retrieval/reranker.py`)

Cohere Rerank v3 API requires `documents=[{"text": str}]` instead of raw strings.

```python
def _cohere_documents(texts: list[str]) -> list[dict] | list[str]:
    import cohere
    if tuple(int(x) for x in cohere.__version__.split(".")[:2]) >= (4, 0):
        return [{"text": t} for t in texts]
    return texts
```

Called inside `_rerank_cohere()`. No interface change to `Reranker`.

**New eval metrics** (append to `src/internal/retrieval/eval_metrics.py`)

```python
def map_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Mean average precision at k."""
    ...

def reranker_improvement_ratio(pre_ndcg: float, post_ndcg: float) -> float:
    """(post / pre) - 1; negative means reranker hurt quality."""
    ...
```

**`eval_runner` additions (M8):**
- `--compare-baseline` flag: runs retrieval without reranking (baseline NDCG), then with reranking; prints `reranker_improvement_ratio`
- MAP@k included in output table alongside NDCG@10 and MRR

---

## Target Quality Gates

| Metric | Current | Target |
|---|---|---|
| P99 reranker latency (20 candidates, CPU) | ≤ 800ms | ≤ 200ms (cache hit ≤ 5ms) |
| NDCG@10 | ≥ 0.50 | ≥ 0.58 |
| MRR | ≥ 0.65 | ≥ 0.72 |

SLO enforcement: `eval_runner --slo-ms 200` exits non-zero if P99 exceeds target. Used in CI.

---

## Testing Strategy

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

## Files Created / Modified

| File | Action |
|---|---|
| `src/internal/retrieval/async_reranker.py` | Create |
| `src/internal/retrieval/cached_reranker.py` | Create |
| `src/internal/retrieval/passage_truncator.py` | Create |
| `src/internal/retrieval/onnx_reranker.py` | Create |
| `src/internal/retrieval/two_stage_reranker.py` | Create |
| `src/internal/retrieval/reranker_benchmark.py` | Create |
| `src/internal/retrieval/reranker.py` | Modify (PassageTruncator call + Cohere v3) |
| `src/internal/retrieval/service.py` | Modify (AsyncReranker integration, over-fetch) |
| `src/internal/retrieval/eval_metrics.py` | Modify (map_at_k, reranker_improvement_ratio) |
| `src/internal/retrieval/eval_runner.py` | Modify (--slo-ms, --compare-baseline, latency table) |
| `tests/unit/retrieval/test_async_reranker.py` | Create |
| `tests/unit/retrieval/test_cached_reranker.py` | Create |
| `tests/unit/retrieval/test_passage_truncator.py` | Create |
| `tests/unit/retrieval/test_onnx_reranker.py` | Create |
| `tests/unit/retrieval/test_two_stage_reranker.py` | Create |
| `tests/unit/retrieval/test_reranker_benchmark.py` | Create |
| `tests/unit/retrieval/test_eval_metrics.py` | Modify (append new tests) |
| `tests/unit/retrieval/test_eval_runner.py` | Modify (append new tests) |
