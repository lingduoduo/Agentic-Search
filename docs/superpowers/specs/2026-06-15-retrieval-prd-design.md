# Retrieval PRD — Design Spec

> **For agentic workers:** Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement the plan derived from this spec.

**Goal:** A single retrieval service that runs identically in dev and prod, supports BM25, dense, and hybrid retrieval modes transparently, and exposes evaluation endpoints so quality is measurable at every milestone.

**Architecture:** Single `RetrievalService` process with pluggable backends selected via `RETRIEVAL_BACKEND` env var. Consumers always call `POST /search`; internal eval endpoints expose per-mode access for debugging and benchmarking.

**Tech Stack:** Python 3.12, FastAPI, Pyserini (BM25 local), FAISS + e5-base-v2 (dense local), OpenSearch (BM25 + kNN prod), Weaviate (vector prod), Redis (embedding cache), RRF + MMR fusion.

---

## 1. Goals & Success Criteria

### Problem

The current retrieval stack has three separate server processes (demo TF-IDF, BM25/dense, reranker), no shared config contract between local and prod, and no quantitative quality gates — so there is no way to know if a change made retrieval better or worse.

### Success Criteria

| Metric | Baseline target | Stretch |
|---|---|---|
| Recall@10 (internal QA pairs) | ≥ 0.80 | ≥ 0.88 |
| NDCG@10 (BEIR subset) | ≥ 0.45 | ≥ 0.52 |
| MRR (internal QA pairs) | ≥ 0.60 | — |
| P99 latency — hybrid, local | ≤ 250ms | ≤ 150ms |
| P99 latency — hybrid, OpenSearch | ≤ 120ms | ≤ 80ms |
| Online thumbs-up rate | ≥ 65% | ≥ 75% |

### Out of Scope

Cross-encoder reranking improvements, LLM-based reranking, connector ingestion pipeline changes, UI changes.

---

## 2. Architecture

**Single service, pluggable backend.** One `RetrievalService` process owns all three retrieval modes. The backend is selected at startup via `RETRIEVAL_BACKEND=local|opensearch|weaviate`. The factory pattern in `src/internal/document_index/factory.py` is formalized as the single source of truth.

```
┌─────────────────────────────────────────────────────┐
│                  RetrievalService                   │
│                                                     │
│  POST /search               ← agents, frontend      │
│  POST /internal/search/sparse   ← eval, debug       │
│  POST /internal/search/dense    ← eval, debug       │
│  POST /internal/search/hybrid   ← eval, tuning      │
│  GET  /health                                       │
│  GET  /api/admin/evals/summary  ← online signals    │
│                                                     │
│  ┌─────────────┐   ┌──────────────┐                 │
│  │ BM25 Engine │   │ Dense Engine │                 │
│  │ (Pyserini / │   │ (FAISS /     │                 │
│  │  OpenSearch)│   │  OpenSearch /│                 │
│  │             │   │  Weaviate)   │                 │
│  └──────┬──────┘   └──────┬───────┘                 │
│         └────────┬─────────┘                        │
│              RRF Fusion + MMR                       │
└─────────────────────────────────────────────────────┘
         ↑ RETRIEVAL_BACKEND=local|opensearch|weaviate
```

### File Layout (new / changed only)

| Path | Role |
|---|---|
| `src/internal/retrieval/service.py` | `RetrievalService` — owns all three modes, backend selection |
| `src/internal/retrieval/backends/base.py` | `RetrievalBackend` ABC |
| `src/internal/retrieval/backends/local.py` | FAISS + Pyserini wrappers (extracted from `retrieval.py`) |
| `src/internal/retrieval/backends/opensearch.py` | OpenSearch BM25 + kNN backend |
| `src/internal/retrieval/backends/weaviate.py` | Weaviate BM25 + vector backend |
| `src/internal/retrieval/fusion.py` | RRF + MMR (extracted from `hybrid_retriever.py`) |
| `src/internal/servers/retrieval/server.py` | FastAPI app wrapping `RetrievalService` |
| `src/internal/servers/retrieval/eval_router.py` | `/internal/search/*` endpoints |

Existing `retrieval_server.py`, `retrieval_rerank.py`, and `hybrid_retriever.py` remain in place during transition and are removed after Milestone 3.

---

## 3. BM25 (Sparse Retrieval)

Keyword-based retrieval using TF-IDF scoring (BM25 variant). Fast, no GPU required, strong on exact-match queries and rare terms. Required for hybrid fusion.

### Local Backend

Pyserini BM25 over a pre-built Lucene index. Already implemented in `SparseRetriever` / `SparseRetrieverConfig` in `src/internal/document_index/retrieval.py`. Extracted into `src/internal/retrieval/backends/local.py` with no behavior change.

### OpenSearch Backend

Uses OpenSearch's native BM25 (`match` query). Runs in the same OpenSearch cluster as the kNN vector index — one index, two query paths.

### Configuration

| Env var | Default | Notes |
|---|---|---|
| `BM25_K1` | `1.2` | Term frequency saturation |
| `BM25_B` | `0.75` | Document length normalization |
| `BM25_TOP_K` | `20` | Hard cap on BM25 candidates; actual fetch = min(top_k × over_fetch_multiplier, BM25_TOP_K) |
| `BM25_INDEX_PATH` | — | Local only: path to Pyserini index directory |

### Indexing

Document chunking at ≤ 512 tokens, 64-token overlap. Chunk size and overlap are configurable but fixed at index time — changing them requires a full re-index. No change to the existing `chunker.py`.

### Known Limit

Pyserini requires Java 11+. The OpenSearch backend has no local Java dependency. Removing the Java dependency for local mode is deferred to a future PRD.

---

## 4. Vector Search (Dense Retrieval)

Embeds queries and documents into a shared vector space; retrieves by approximate nearest-neighbor (ANN) search. Stronger than BM25 on semantic similarity, paraphrase, and concept-level queries.

### Embedding Models

| Model | Dims | When to use |
|---|---|---|
| `intfloat/e5-base-v2` | 768 | Default local — good quality, runs on CPU |
| `intfloat/e5-large-v2` | 1024 | Higher recall, needs ≥ 8 GB RAM |
| OpenAI `text-embedding-3-small` | 1536 | No local GPU; requires `OPENAI_API_KEY` |

New models are added by extending `DenseRetrieverConfig` presets — no other code changes required.

### Local Backend (FAISS)

`IndexHNSWFlat` with `ef_construction=128, ef_search=64`. Redis query-embedding cache already wired via `redis_url` in `DenseRetrieverConfig` — a cache hit skips the embedding call entirely.

### OpenSearch Backend

kNN plugin with HNSW (`engine=lucene`, `space_type=cosinesimil`). One index holds both BM25 fields and kNN vectors — sparse and dense queries run against the same index, no sync required.

### Weaviate Backend

Vectorizer set to `none` (embeddings pre-computed client-side); `nearVector` query path. Already implemented in `weaviate_document_index.py` — wrapped behind `RetrievalBackend` ABC with no behavior change.

### Normalization

All query embeddings are L2-normalized before search. `normalize_query_embeddings=True` is enforced for cosine similarity backends (OpenSearch, Weaviate); optional for FAISS inner-product indexes.

### Latency Budget

| Step | Budget |
|---|---|
| Embedding call (cached) | ≤ 30ms |
| Embedding call (uncached, CPU) | ≤ 80ms |
| ANN search — local FAISS | ≤ 40ms |
| ANN search — OpenSearch kNN | ≤ 60ms |

---

## 5. Hybrid Retrieval

Runs BM25 and dense retrieval in parallel, fuses results with Reciprocal Rank Fusion (RRF), then re-ranks for diversity with Maximal Marginal Relevance (MMR). This is the default mode for `POST /search`.

### Pipeline

```
query
  ├─► BM25 (top_k × 2 results)  ─┐
  │                               ├─► RRF fusion ─► MMR re-rank ─► top_k results
  └─► Dense ANN (top_k × 2)     ─┘
```

Both legs run concurrently (`concurrent.futures.ThreadPoolExecutor` for local; async for OpenSearch/Weaviate). Added latency over the slower leg is < 5ms.

### RRF Fusion

Extracted from `hybrid_retriever.py` to `src/internal/retrieval/fusion.py`:

```
score(doc) = Σ  1 / (k + rank_in_set)    k = 60
             sets
```

Scale-invariant — no raw score normalization required across BM25 and cosine similarity outputs.

### MMR Re-rank

```
MMR(doc) = λ · relevance(doc) - (1 - λ) · max_similarity(doc, selected)
```

`λ = 0.5` default. Similarity proxy is source-prefix matching — no embeddings required at re-rank time. `λ` is configurable per-request on internal eval endpoints only; fixed at 0.5 on `POST /search`.

### Fallback Behavior

| Failure | Behavior |
|---|---|
| BM25 leg fails | Serve dense-only results; log warning; set `"retrieval_mode": "dense_only"` in response |
| Dense leg fails | Serve BM25-only results; set `"retrieval_mode": "sparse_only"` |
| Both legs fail | 502 with structured error |

### Tunable Parameters (internal endpoints only)

| Param | Default | Range |
|---|---|---|
| `rrf_k` | 60 | 10–200 |
| `mmr_lambda` | 0.5 | 0.0–1.0 |
| `over_fetch_multiplier` | 2× | 1–4× |

---

## 6. Evaluation Metrics

Three signal sources serve distinct feedback loops.

### Offline — Internal Labeled QA Pairs

Hand-annotated `(query, [relevant_doc_ids])` pairs stored in `data/eval/qa_pairs.jsonl`. Extends the existing `src/internal/servers/evals/api.py`.

| Metric | Formula | Target |
|---|---|---|
| Recall@K | `|retrieved ∩ relevant| / |relevant|` | ≥ 0.80 @ K=10 |
| NDCG@K | Normalized discounted cumulative gain | ≥ 0.45 @ K=10 |
| MRR | `1 / rank_of_first_relevant` | ≥ 0.60 |

Run: `python -m src.internal.servers.evals.eval_cli --dataset data/eval/qa_pairs.jsonl`

### Offline — Public Benchmarks (BEIR Subset)

Three BEIR tasks that cover retrieval diversity:

| Dataset | Domain | Queries |
|---|---|---|
| `nfcorpus` | Medical | 3,600 |
| `fiqa` | Finance QA | 648 |
| `scifact` | Scientific claims | 300 |

Downloaded once to `data/beir/`. Eval script wraps `RetrievalService` client so it runs against any configured backend. Reports per-dataset NDCG@10 and aggregate.

### Online — Production Signals

Three signals captured per query in `AgenticSearchStore`:

| Signal | Collection point | Metric |
|---|---|---|
| Thumbs up / down | Frontend `POST /api/feedback` | Thumbs-up rate ≥ 65% |
| Citation clicked | Frontend click event | Click-through rate on retrieved docs |
| Session continued | Next turn within 60s | Engagement rate (secondary) |

Queried via `GET /api/admin/evals/summary` (admin-only). Raw rates only — no ML model on signals in this PRD.

### CI Eval Gate

Offline eval runs on every PR touching `src/internal/retrieval/`. PR fails if:
- Recall@10 drops > 2pp vs. baseline snapshot in `data/eval/baseline_metrics.json`, or
- NDCG@10 drops > 1pp vs. baseline snapshot.

---

## 7. API

### Primary Endpoint (all consumers)

```
POST /search
Authorization: Bearer <session-jwt>        # Depends(user_from_headers)

{
  "query": "procurement approval process",
  "top_k": 10,                              // optional, default 5
  "filters": { "source": "confluence" }    // optional
}

→ 200
{
  "results": [
    {
      "doc_id": "confluence-page-42-chunk-3",
      "title": "Procurement Policy v2",
      "text": "...",
      "url": "https://...",
      "score": 0.031,
      "metadata": {}
    }
  ],
  "retrieval_mode": "hybrid",
  "executed_queries": ["procurement approval process"],
  "latency_ms": 87
}
```

### Internal Eval Endpoints (admin only)

Auth: `Depends(make_require_admin(...))` — matches existing pattern in `src/internal/servers/evals/api.py`. Caller must be in `AppSettings.auth.super_users` or carry `role="admin"` in JWT.

```
POST /internal/search/sparse     # BM25 only — same request/response shape as /search
POST /internal/search/dense      # Dense only — same request/response shape as /search
POST /internal/search/hybrid     # Hybrid with tunable overrides:
  {
    "query": "...",
    "top_k": 10,
    "rrf_k": 60,
    "mmr_lambda": 0.5,
    "over_fetch": 2
  }

GET /api/admin/evals/summary
→ { "thumbs_up_rate": 0.71, "ctr": 0.43, "rated_queries": 512 }
```

### Health

```
GET /health
→ { "status": "ok", "backend": "local|opensearch|weaviate" }
```

### Online Feedback (existing, no change)

```
POST /api/feedback   { "session_id": "...", "signal": "thumbs_up|thumbs_down" }
```

### Versioning

No `/v1/` URL prefix. Breaking changes increment a `Retrieval-API-Version` response header. First version is `1.0`.

---

## 8. Milestones

Phased delivery; each phase requires its gate to be met before the next phase begins.

### Milestone 1 — BM25 Baseline + Service Skeleton (~2 weeks)

**Deliverables:**
- `RetrievalBackend` ABC (`backends/base.py`)
- Local BM25 backend extracted to `backends/local.py`
- `RetrievalService` with `POST /search` and `GET /health`
- Offline eval CLI against internal QA pairs

**Gate:** Recall@10 ≥ 0.75 on internal QA pairs. P99 latency ≤ 300ms local.

---

### Milestone 2 — Dense Retrieval + Hybrid Fusion (~3 weeks)

**Deliverables:**
- Dense leg: FAISS + e5-base-v2 in `backends/local.py`
- RRF + MMR extracted to `fusion.py`
- Hybrid mode as default for `POST /search`
- Internal eval endpoints (`/internal/search/sparse`, `/internal/search/dense`, `/internal/search/hybrid`)
- Redis embedding cache wired
- `retrieval_server.py` deprecated (still running, no new features)

**Gate:** Recall@10 ≥ 0.80, NDCG@10 ≥ 0.45 on internal QA pairs. Hybrid P99 ≤ 250ms local.

---

### Milestone 3 — OpenSearch + Weaviate Backends + CI Eval Gate (~3 weeks)

**Deliverables:**
- `backends/opensearch.py` (BM25 + kNN)
- `backends/weaviate.py` (nearVector)
- BEIR eval script (nfcorpus, fiqa, scifact)
- CI job: PR fails if Recall@10 drops > 2pp or NDCG@10 drops > 1pp
- `RETRIEVAL_BACKEND` env var switching fully operational
- `retrieval_server.py`, `retrieval_rerank.py`, `hybrid_retriever.py` removed

**Gate:** NDCG@10 ≥ 0.45 on all three BEIR datasets. OpenSearch P99 ≤ 120ms. CI green on 3 consecutive PRs.

---

### Milestone 4 — Online Signals + Production Hardening (~2 weeks)

**Deliverables:**
- `POST /api/feedback` persisted to `AgenticSearchStore`
- `GET /api/admin/evals/summary` (admin, `make_require_admin`)
- Structured logging: `retrieval_mode` and `latency_ms` on every query
- Load test at 50 QPS with no P99 regression vs. Milestone 3

**Gate:** Thumbs-up rate ≥ 65% over 500 rated queries. No P99 regression under 50 QPS load.
