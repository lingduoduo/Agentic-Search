# Retrieval Optimization PRD — Design Spec

**Date:** 2026-06-18
**Status:** Draft
**Depends on:** M1–M4 Retrieval PRD (`2026-06-15-retrieval-prd-design.md`), Reranking PRD (`2026-06-16-reranking-prd-design.md`)

---

## 1. Goals & Success Criteria

### Problem

The M1–M4 retrieval service meets baseline recall and latency targets but leaves measurable quality and throughput on the table:

- BM25 uses static `k1=1.2, b=0.75` — never tuned to this corpus. A grid search and BM25+ variant could lift recall significantly at zero latency cost.
- FAISS `IndexHNSWFlat` stores full float32 vectors. At 10 M documents, that is 30 GB RAM for 768-dim e5-base-v2. IVF-PQ quantization cuts this to ~3 GB with < 2pp recall loss.
- RRF fusion weights are uniform (`1 / (k + rank)` with fixed `k=60`). Learned per-source weights consistently outperform fixed weights on domain-shifted corpora.
- Result caching exists for embeddings (Redis) but not for full search results. Repeated queries — common in agent multi-turn loops — pay full retrieval cost each time.
- No structured SLO enforcement: latency regressions are caught only in manual load tests, not in CI.

### Success Criteria

| Metric | M1–M4 baseline | Optimization target |
|---|---|---|
| Recall@10 (internal QA) | ≥ 0.80 | ≥ 0.87 |
| NDCG@10 (BEIR avg) | ≥ 0.45 | ≥ 0.52 |
| MRR (internal QA) | ≥ 0.60 | ≥ 0.68 |
| P99 latency — hybrid, local | ≤ 250ms | ≤ 120ms |
| P99 latency — hybrid, OpenSearch | ≤ 120ms | ≤ 70ms |
| Embedding memory (10 M docs) | ~30 GB (FP32) | ≤ 4 GB (IVF-PQ) |
| Result cache hit rate | 0% | ≥ 30% on agent workloads |
| Throughput (local) | ~40 QPS | ≥ 120 QPS |

### Out of Scope

- Cross-encoder or LLM reranking (Reranking PRD)
- Connector ingestion pipeline
- Training new embedding models
- UI changes

---

## 2. Architecture

The optimization layer wraps the existing `RetrievalService` without breaking its interface. All changes are additive or internal.

```
POST /search
     │
     ▼
ResultCache.get(query, filters, top_k)        ← NEW M7: Redis result cache
     │ miss
     ▼
QueryOptimizer.expand(query)                  ← NEW M5: expansion + spell-correct
     │
     ▼
RetrievalService.search(expanded_query, ...)
  ├── BM25 leg: SparseRetriever               ← M5: tuned k1/b, BM25+ option
  │   └── QueryExpander (synonyms, acronyms)  ← M5
  │
  ├── Dense leg: FAISSBackend                 ← M6: IVF-PQ, ef_search tuning
  │   └── EmbeddingBatcher (async)            ← M6
  │
  ├── RRF fusion                              ← M7: learned weights per source
  └── MMR rerank                              ← M7: adaptive λ per intent
     │
     ▼
ResultCache.set(...)                          ← NEW M7
     │
     ▼
SearchResponse + latency/cache metrics
```

All new components are **opt-in via env vars** — unset = unchanged M1–M4 behavior.

### New Files

| Path | Role |
|---|---|
| `src/internal/retrieval/query_optimizer.py` | Expansion, spell correction, synonym injection |
| `src/internal/retrieval/result_cache.py` | Redis-backed result cache with TTL + semantic key |
| `src/internal/retrieval/bm25_tuner.py` | Grid search + BM25+ parameter tuning against QA pairs |
| `src/internal/retrieval/index_optimizer.py` | FAISS IVF-PQ builder, ef_search auto-tuner |
| `src/internal/retrieval/fusion_learner.py` | Per-source RRF weight optimizer, adaptive MMR |
| `src/internal/servers/retrieval/optimize_router.py` | `/internal/optimize/*` admin endpoints |

---

## 3. BM25 Optimization

### 3.1 Parameter Tuning

Current `k1=1.2, b=0.75` are BM25 library defaults, never validated on this corpus. A grid search over the labeled QA pairs finds the optimal values.

**`src/internal/retrieval/bm25_tuner.py`:**

```python
class BM25Tuner:
    def grid_search(
        self,
        qa_pairs: list[QAPair],
        k1_range: list[float] = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
        b_range: list[float]  = [0.3, 0.5, 0.6, 0.75, 0.9],
        metric: str = "recall@10",
    ) -> BM25Params:
        """Grid search over (k1, b) pairs. Returns params that maximise metric."""
```

Results written to `data/eval/bm25_params.json`. The tuner is a CLI tool, not part of the serving hot path.

### 3.2 BM25+ Variant

BM25+ adds a lower-bound floor `δ` to term frequency to prevent zero scores on rare terms in long documents:

```
score(q, d) = IDF × (k1 + 1) × (TF/(TF + k1(1-b+b·dl/avgdl)) + δ)
```

Default `δ=1.0`. Enabled via `BM25_VARIANT=bm25plus`. Falls back to standard BM25 when unset.

**Expected gain:** +2–4pp Recall@10 on queries containing rare domain terms.

### 3.3 Query Expansion

**`QueryOptimizer.expand(query)`** adds synonym and acronym expansions before BM25 indexing:

| Expansion type | Source | Example |
|---|---|---|
| Acronyms | `data/query/acronyms.json` (project-maintained) | "ML" → "machine learning" |
| Synonyms | WordNet via `nltk.corpus.wordnet` (nouns only) | "procurement" → "purchasing, sourcing" |
| Spell correction | `symspellpy` edit-distance ≤ 2 | "retreival" → "retrieval" |

Expansion is applied **only to the BM25 leg** — dense retrieval already handles semantic variation.

**Config:**

| Env var | Default | Notes |
|---|---|---|
| `QUERY_EXPANSION_ENABLED` | `false` | Enable acronym + synonym expansion |
| `SPELL_CORRECTION_ENABLED` | `false` | Enable symspellpy correction |
| `EXPANSION_MAX_TERMS` | `3` | Maximum added terms to prevent query bloat |

---

## 4. Vector Search Optimization

### 4.1 IVF-PQ Quantization

`IndexHNSWFlat` stores raw float32 vectors: 768 dims × 4 bytes × N docs. At 10 M docs = 30.7 GB. `IndexIVFPQ` compresses vectors to ~4 bytes/doc:

```
IVF-PQ configuration:
  nlist   = 4096   # Voronoi cells; sqrt(N) rule for 10M docs
  m       = 96     # sub-vectors (768 / 8 bytes each)
  nbits   = 8      # 256 centroids per sub-vector
  nprobe  = 64     # cells searched at query time (recall/latency tradeoff)
```

**`src/internal/retrieval/index_optimizer.py`:**

```python
class FAISSIndexBuilder:
    def build_ivfpq(
        self,
        embeddings: np.ndarray,
        *,
        nlist: int = 4096,
        m: int = 96,
        nbits: int = 8,
        nprobe: int = 64,
        training_sample: int = 500_000,
    ) -> faiss.IndexIVFPQ: ...
```

Building requires a training pass on a random sample (500 K vectors). At re-index time, the builder saves the trained index to `FAISS_INDEX_PATH` alongside a `metadata.json` recording the config. The old `IndexHNSWFlat` remains loadable by setting `FAISS_INDEX_TYPE=hnsw` — no forced migration.

**Expected gain:** Memory −90%, ANN P99 −35%, with < 2pp recall loss at `nprobe=64`.

### 4.2 HNSW ef_search Auto-Tuner

For deployments that stay on `IndexHNSWFlat`, `ef_search` trades recall for latency. The tuner finds the minimum `ef_search` that meets a recall target:

```python
class HNSWTuner:
    def calibrate(
        self,
        index: faiss.IndexHNSWFlat,
        qa_pairs: list[QAPair],
        *,
        target_recall_at_10: float = 0.80,
    ) -> int:  # returns optimal ef_search
```

Result written to `data/eval/hnsw_params.json`. Served via `EF_SEARCH` env var.

### 4.3 Embedding Batching

The current path embeds one query per request. When the agent loop fires N parallel retrievals (multi-step search), each hits the embedding model serially. An async batcher coalesces concurrent requests into one model call:

```python
class EmbeddingBatcher:
    def __init__(self, *, max_batch: int = 32, wait_ms: float = 5.0): ...
    async def embed(self, text: str) -> np.ndarray: ...
```

`wait_ms` is the coalescing window — requests arriving within 5ms are batched together. At 50 QPS, expected to reduce model invocations by ~60%.

---

## 5. Hybrid Retrieval Optimization

### 5.1 Learned RRF Weights

Fixed RRF (`k=60`, uniform weights) is a strong baseline but suboptimal when one source consistently outperforms the other for a query type. A simple linear model learns per-source weights offline:

```python
class FusionLearner:
    def fit(
        self,
        qa_pairs: list[QAPair],
        sparse_results: list[list[SearchResult]],
        dense_results: list[list[SearchResult]],
    ) -> FusionWeights:
        """Returns (w_sparse, w_dense) minimising 1 - Recall@10 on qa_pairs."""
```

Weights are stored in `data/eval/fusion_weights.json` and loaded at service startup if present. Env var `FUSION_WEIGHTS_PATH` overrides the default path. Absent file → fall back to uniform weights.

**Fusion formula with learned weights:**

```
score(doc) = w_sparse × (1 / (k + rank_sparse))
           + w_dense  × (1 / (k + rank_dense))
```

### 5.2 Adaptive MMR Lambda

The current `mmr_lambda=0.5` is fixed. Queries that are highly specific (short, rare terms) benefit from `lambda → 1.0` (pure relevance). Broad conceptual queries benefit from `lambda → 0.3` (more diversity). A query classifier selects the lambda:

```python
def adaptive_mmr_lambda(query: str) -> float:
    tokens = query.split()
    if len(tokens) <= 3:
        return 0.8   # short/specific → prioritize relevance
    if len(tokens) >= 10:
        return 0.3   # long/broad → prioritize diversity
    return 0.5       # default
```

Applied automatically when `ADAPTIVE_MMR=true`. Internal eval endpoints still accept an explicit `mmr_lambda` override.

### 5.3 Result Cache

Repeated or near-identical queries are common in agent multi-turn loops. A Redis result cache stores full `SearchResponse` objects:

```
key   = sha256(canonical_query + filters_json + top_k)
value = SearchResponse JSON
TTL   = 300s (configurable via RESULT_CACHE_TTL)
```

Cache keys are computed on the **canonicalized** query (lowercased, punctuation stripped, sorted filter keys) so minor variations still hit the cache. The hit/miss ratio and latency savings are logged on every request and surfaced via `GET /api/admin/retrieval/stats`.

---

## 6. Evaluation Metrics

Extends the M1–M4 eval framework with latency histograms, throughput tracking, and cache efficiency.

### 6.1 Offline Quality (Expanded)

Adds Precision@K and Hit Rate alongside existing Recall@10, NDCG@10, MRR:

| Metric | Formula | New target |
|---|---|---|
| Precision@5 | `|retrieved ∩ relevant| / 5` | ≥ 0.60 |
| Hit Rate@1 | `1 if any relevant in top 1 else 0` | ≥ 0.70 |
| Recall@10 | (unchanged) | ≥ 0.87 |
| NDCG@10 | (unchanged) | ≥ 0.52 |
| MRR | (unchanged) | ≥ 0.68 |

All computed by `eval_metrics.py` — new metrics are additional functions, no schema change.

### 6.2 Latency SLO Tracking

`eval_runner.py` extended to record per-query latency and emit a structured latency report:

```json
{
  "p50_ms": 45,
  "p95_ms": 98,
  "p99_ms": 118,
  "max_ms": 203,
  "slo_breaches": 2,
  "slo_target_ms": 120
}
```

CI gate fails if `p99_ms > slo_target_ms` on the internal eval set (200 queries). `slo_target_ms` defaults to 120ms for local; configurable via `LATENCY_SLO_MS`.

### 6.3 Throughput Benchmark

Locust load test extended with two new scenarios:

| Scenario | Concurrency | Duration | Target |
|---|---|---|---|
| Sustained query load | 50 | 5 min | ≥ 120 QPS, P99 ≤ 120ms |
| Burst (agent multi-turn) | 200 | 60s | P99 ≤ 300ms, 0 errors |

Results written to `data/eval/load_results.json` for trend tracking.

### 6.4 Cache Efficiency

Reported via `GET /api/admin/retrieval/stats`:

```json
{
  "result_cache_hits": 1203,
  "result_cache_misses": 842,
  "result_cache_hit_rate": 0.588,
  "embedding_cache_hit_rate": 0.72,
  "avg_cache_latency_ms": 3.1,
  "avg_full_latency_ms": 89.4
}
```

Target: result cache hit rate ≥ 0.30 on agent workloads (validated via replay of 500 logged queries from `AgenticSearchStore`).

---

## 7. API

### Existing (no change)

`POST /search`, `GET /health`, `POST /api/feedback`, `GET /api/admin/evals/summary` — all unchanged.

### New: Stats Endpoint

```
GET /api/admin/retrieval/stats
Authorization: admin

→ 200
{
  "result_cache_hit_rate": 0.42,
  "embedding_cache_hit_rate": 0.71,
  "avg_latency_ms": 74,
  "p99_latency_ms": 112,
  "throughput_qps": 87,
  "backend": "local",
  "index_type": "ivfpq",
  "query_expansion_enabled": true
}
```

### New: Live Config Patch

```
PATCH /api/admin/retrieval/config
Authorization: admin

{
  "rrf_k": 80,
  "mmr_lambda": 0.4,
  "nprobe": 96,
  "result_cache_ttl": 600
}

→ 200 { "applied": ["rrf_k", "mmr_lambda", "nprobe", "result_cache_ttl"] }
```

Hot-reloads tunable parameters without restart. Non-tunable params (index type, backend) require restart and are rejected with 400.

### New: Optimization Endpoints (internal)

```
POST /internal/optimize/bm25-tune
  { "qa_pairs_path": "data/eval/qa_pairs.jsonl" }
  → { "k1": 0.9, "b": 0.6, "recall_at_10": 0.86 }

POST /internal/optimize/hnsw-tune
  { "target_recall": 0.82 }
  → { "ef_search": 96, "measured_recall": 0.831 }

POST /internal/optimize/fusion-weights
  { "qa_pairs_path": "data/eval/qa_pairs.jsonl" }
  → { "w_sparse": 0.38, "w_dense": 0.62, "recall_at_10": 0.88 }
```

All admin-only (`Depends(make_require_admin(...))`). Runs offline; takes 10–120s. Results written to `data/eval/` and returned in the response.

---

## 8. File Map

| Action | Path | Responsibility |
|---|---|---|
| **Create** | `src/internal/retrieval/query_optimizer.py` | Expansion, spell correction |
| **Create** | `src/internal/retrieval/result_cache.py` | Redis result cache |
| **Create** | `src/internal/retrieval/bm25_tuner.py` | BM25 parameter grid search |
| **Create** | `src/internal/retrieval/index_optimizer.py` | IVF-PQ builder, HNSW tuner |
| **Create** | `src/internal/retrieval/fusion_learner.py` | Learned RRF weights, adaptive MMR |
| **Create** | `src/internal/servers/retrieval/optimize_router.py` | `/internal/optimize/*` endpoints |
| **Modify** | `src/internal/retrieval/service.py` | Wire QueryOptimizer, ResultCache, FusionLearner |
| **Modify** | `src/internal/retrieval/eval_metrics.py` | Add Precision@5, Hit Rate@1, latency histogram |
| **Modify** | `src/internal/retrieval/eval_runner.py` | Emit latency report, SLO gate |
| **Modify** | `src/internal/servers/retrieval/server.py` | Mount optimize_router, stats endpoint, config patch |
| **Create** | `tests/unit/retrieval/test_query_optimizer.py` | Expansion + spell correction unit tests |
| **Create** | `tests/unit/retrieval/test_result_cache.py` | Cache key, TTL, hit/miss unit tests |
| **Create** | `tests/unit/retrieval/test_bm25_tuner.py` | Grid search correctness tests |
| **Create** | `tests/unit/retrieval/test_index_optimizer.py` | IVF-PQ build smoke test (tiny index) |
| **Create** | `tests/unit/retrieval/test_fusion_learner.py` | Weight fitting, adaptive lambda tests |

---

## 9. Milestones

### Milestone 5 — Query Optimization (~2 weeks)

**Deliverables:**
- `QueryOptimizer` with acronym expansion and spell correction
- BM25+ variant (`BM25_VARIANT=bm25plus`)
- `BM25Tuner.grid_search` CLI, results persisted to `data/eval/bm25_params.json`
- `EXPANSION_MAX_TERMS`, `SPELL_CORRECTION_ENABLED` env vars wired

**Gate:** Recall@10 ≥ 0.84 on internal QA pairs (vs 0.80 M4 baseline). No P99 regression.

---

### Milestone 6 — Index Optimization (~3 weeks)

**Deliverables:**
- `FAISSIndexBuilder.build_ivfpq` — builds and saves IVF-PQ index
- `HNSWTuner.calibrate` — finds optimal `ef_search`
- `EmbeddingBatcher` async coalescer (max 32, wait 5ms)
- `FAISS_INDEX_TYPE=ivfpq|hnsw` env var; `nprobe`, `ef_search` live-configurable
- Throughput benchmark: ≥ 120 QPS (sustained 50 concurrent)

**Gate:** Recall@10 ≥ 0.85. IVF-PQ memory ≤ 4 GB at 10 M docs. P99 ≤ 120ms local.

---

### Milestone 7 — Learned Fusion + Result Cache (~2 weeks)

**Deliverables:**
- `FusionLearner.fit` and `FusionWeights` loading at startup
- `adaptive_mmr_lambda` (enabled by `ADAPTIVE_MMR=true`)
- `ResultCache` with Redis backend and `RESULT_CACHE_TTL` config
- `GET /api/admin/retrieval/stats` endpoint
- `PATCH /api/admin/retrieval/config` hot-reload endpoint

**Gate:** NDCG@10 ≥ 0.52 (BEIR avg). Result cache hit rate ≥ 0.30 on 500-query agent replay.

---

### Milestone 8 — SLO Enforcement + Production Hardening (~1 week)

**Deliverables:**
- CI eval gate: P99 > `LATENCY_SLO_MS` → PR fails
- Locust burst scenario (200 concurrent, 60s)
- `POST /internal/optimize/*` admin endpoints with result persistence
- Structured latency histogram in every `SearchResponse` (debug mode)
- BEIR nfcorpus/fiqa/scifact re-run; results committed to `data/eval/baseline_metrics.json`

**Gate:** P99 ≤ 120ms local, ≤ 70ms OpenSearch. NDCG@10 ≥ 0.52 on all three BEIR tasks. CI green on 3 consecutive PRs.
