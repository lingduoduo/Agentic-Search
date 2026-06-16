# Reranking PRD — Design Spec

**Date:** 2026-06-16
**Status:** Draft

---

## 1. Goals & Success Criteria

### Problem

The existing `RetrievalService` (M1–M4) returns hybrid-fused results (RRF + MMR) but has no neural reranking step. The standalone `POST /rerank` microservice exists but is disconnected from the new retrieval pipeline. BGE models and Cohere are not wired into any retrieval-layer path.

### Success Criteria

- A single `Reranker` class supports both local cross-encoders (BGE, ms-marco) and Cohere via one config switch
- `RetrievalService.search()` optionally reranks after RRF fusion; callers that omit `RERANKER_PROVIDER` get unchanged behavior
- NDCG@10 ≥ 0.50 over `data/eval/qa_pairs.jsonl` (vs 0.45 retrieval-only baseline)
- MRR ≥ 0.65 (vs 0.60 baseline)
- Reranker P99 latency ≤ 800ms measured over 20 candidates on CPU

### Out of Scope

- Replacing the existing standalone `POST /rerank` server (kept for the web-app layer)
- Training or fine-tuning reranker models
- Streaming reranked results
- Reranking at the agent loop level (only retrieval-service level)

---

## 2. Architecture

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

## 3. Reranker

### `RerankerConfig`

```python
@dataclass(frozen=True)
class RerankerConfig:
    provider: Literal["local", "cohere"]
    model: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 32
    device: str = "cpu"          # "cpu" | "mps" | "cuda"
    api_key: str | None = None   # required when provider="cohere"
    top_k: int | None = None     # None → use search top_k

    def validate(self) -> None:
        if self.provider not in ("local", "cohere"):
            raise ValueError(f"Unknown provider: {self.provider!r}")
        if self.provider == "cohere" and not self.api_key:
            raise ValueError("api_key is required for provider='cohere'")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
```

### `Reranker`

```python
class Reranker:
    def __init__(self, config: RerankerConfig) -> None:
        config.validate()
        self._config = config
        if config.provider == "local":
            self._local = SentenceTransformerReranker.load(
                config.model, config.batch_size, config.device
            )

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Rescore `results` and return top_k sorted by descending score."""
        if not results:
            return results
        effective_k = self._config.top_k or top_k
        if self._config.provider == "local":
            return self._rerank_local(query, results, effective_k)
        return self._rerank_cohere(query, results, effective_k)

    def _rerank_local(self, query, results, top_k) -> list[RetrievalResult]:
        # Embed doc_id so we can map scores back after reranking
        docs = [{"contents": r.text, "title": r.title, "doc_id": r.doc_id} for r in results]
        scored = self._local.rerank([query], [docs], topk=top_k)
        # scored[0] is list[dict] with "document" (original dict) and "score" keys
        id_to_result = {r.doc_id: r for r in results}
        reranked = []
        for item in scored[0]:
            doc_id = item["document"].get("doc_id")
            if doc_id and doc_id in id_to_result:
                reranked.append(
                    dataclasses.replace(id_to_result[doc_id], score=float(item["score"]))
                )
        return reranked[:top_k]

    def _rerank_cohere(self, query, results, top_k) -> list[RetrievalResult]:
        passages = [r.text for r in results]
        scores = cohere_rerank_api(query, passages, self._config.model, self._config.api_key)
        scored = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
        return [
            dataclasses.replace(r, score=float(s)) for s, r in scored[:top_k]
        ]

    @classmethod
    def from_env(cls) -> "Reranker | None":
        provider = os.environ.get("RERANKER_PROVIDER")
        if not provider:
            return None
        return cls(RerankerConfig(
            provider=provider,
            model=os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            batch_size=int(os.environ.get("RERANKER_BATCH_SIZE", "32")),
            device=os.environ.get("RERANKER_DEVICE", "cpu"),
            api_key=os.environ.get("COHERE_API_KEY"),
            top_k=int(os.environ["RERANKER_TOP_K"]) if os.environ.get("RERANKER_TOP_K") else None,
        ))
```

### Supported Models

| Provider | Model | Notes |
|---|---|---|
| `local` | `BAAI/bge-reranker-v2-m3` | **Default.** Multilingual, ~300ms P99 CPU |
| `local` | `BAAI/bge-reranker-base` | Faster, English-focused, ~200ms P99 CPU |
| `local` | `cross-encoder/ms-marco-MiniLM-L12-v2` | Legacy default; kept for compatibility |
| `cohere` | `rerank-english-v3.0` | API call, ~150ms median |
| `cohere` | `rerank-multilingual-v3.0` | API call, multilingual |

---

## 4. `RetrievalService` Integration

`RetrievalService` accepts an optional `reranker` at construction. `from_env()` builds it automatically.

```python
class RetrievalService:
    def __init__(
        self,
        backend: RetrievalBackend,
        reranker: Reranker | None = None,
    ) -> None:
        self._backend = backend
        self._reranker = reranker

    @classmethod
    def from_env(cls) -> "RetrievalService":
        return cls(_build_backend(), reranker=Reranker.from_env())

    def search(self, query, top_k=5, filters=None):
        # ... existing sparse + dense + RRF + MMR ...
        fused, mode = ...  # existing path unchanged

        if self._reranker:
            fused = self._reranker.rerank(query, fused, top_k)
            mode = f"{mode}+reranked"

        return fused, mode
```

No changes to existing callers — `retrieval_mode` just gains the `+reranked` suffix when active.

---

## 5. Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `RERANKER_PROVIDER` | No | unset (skip reranking) | `local` or `cohere` |
| `RERANKER_MODEL` | No | `BAAI/bge-reranker-v2-m3` | HF model name or Cohere model name |
| `RERANKER_BATCH_SIZE` | No | `32` | Batch size for local model |
| `RERANKER_DEVICE` | No | `cpu` | `cpu`, `mps`, or `cuda` |
| `RERANKER_TOP_K` | No | same as search `top_k` | Cap returned results after reranking |
| `COHERE_API_KEY` | If `provider=cohere` | — | Cohere API key |

---

## 6. Evaluation

### Eval Runner Extension

`eval_runner.py` gains a `--reranker` / `--reranker_model` flag. When provided, results are reranked before metrics are computed, and latency is measured per-query.

```bash
python -m src.internal.retrieval.eval_runner \
    --dataset data/eval/qa_pairs.jsonl \
    --top_k 10 \
    --reranker local \
    --reranker_model BAAI/bge-reranker-v2-m3
```

Output JSON:
```json
{
  "retrieval":  {"recall@10": 0.82, "ndcg@10": 0.48, "mrr": 0.63},
  "reranked":   {"recall@10": 0.82, "ndcg@10": 0.55, "mrr": 0.71},
  "latency_ms": {"p50": 310, "p95": 580, "p99": 720, "n": 50}
}
```

### Gate Criteria

| Metric | Gate | Baseline (retrieval-only) |
|---|---|---|
| NDCG@10 | ≥ 0.50 | 0.45 |
| MRR | ≥ 0.65 | 0.60 |
| Reranker P99 latency | ≤ 800ms | — |

### Existing Metrics Unchanged

`eval_metrics.py` (`recall_at_k`, `ndcg_at_k`, `mrr`) is reused as-is. No new metric functions needed.

---

## 7. File Map

| Action | Path | Responsibility |
|---|---|---|
| **Create** | `src/internal/retrieval/reranker.py` | `RerankerConfig` + `Reranker` (local + Cohere dispatch) |
| **Modify** | `src/internal/retrieval/service.py` | Accept `reranker` param; call after MMR; `from_env()` extension |
| **Modify** | `src/internal/retrieval/eval_runner.py` | Add `--reranker` / `--reranker_model` flags; latency measurement |
| **Create** | `tests/unit/retrieval/test_reranker.py` | `RerankerConfig` validation; local + Cohere paths via monkeypatch; smoke latency test |
| **Modify** | `tests/unit/retrieval/test_service.py` | Add reranker injection tests; `+reranked` mode suffix |

**Not changed:** `src/internal/servers/retrieval/rerank.py`, `search_nlp_models.py`, `eval_metrics.py`, existing backends.

---

## 8. Error Handling

| Scenario | Behavior |
|---|---|
| `RERANKER_PROVIDER` unset | Skip reranking entirely; no log noise |
| Local model fails to load | Raise at startup (fail-fast) |
| Cohere API returns non-200 | Log warning; return MMR-ordered results unchanged |
| Score list length mismatch | Raise `ValueError` (existing `SentenceTransformerReranker` behavior retained) |
| `COHERE_API_KEY` missing with `provider=cohere` | Raise `ValueError` at `RerankerConfig.validate()` |

---

## 9. Testing Strategy

- **Unit:** `test_reranker.py` monkeypatches `SentenceTransformerReranker.load` and `cohere_rerank_api` — no model downloads in CI
- **Unit:** `test_service.py` injects a `MagicMock` reranker; asserts `mode` suffix and that `rerank()` is called with correct args
- **Smoke:** One test asserts local reranking 20 candidates completes in < 5s (very generous; gate is 800ms measured on real hardware via eval_runner)
- **Eval gate:** Run manually via `eval_runner.py` against `data/eval/qa_pairs.jsonl`
