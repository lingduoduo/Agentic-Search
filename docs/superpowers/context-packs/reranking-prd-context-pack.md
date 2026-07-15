# Generated Context Pack

# Reranking Prd

## Sources

- [Specification: 2026-06-16-reranking-prd-design.md](../archive/specs/2026-06-16-reranking-prd-design.md)
- [Plan: 2026-06-16-reranking-prd.md](../archive/plans/2026-06-16-reranking-prd.md)

## Specification Context

### Out of Scope

- Replacing the existing standalone `POST /rerank` server (kept for the web-app layer)
- Training or fine-tuning reranker models
- Streaming reranked results
- Reranking at the agent loop level (only retrieval-service level)

---

### 2. Architecture

The `Reranker` is constructed once at startup via `Reranker.from_env()` and injected into `RetrievalService`. When `RERANKER_PROVIDER` is unset, `from_env()` returns `None` and the service skips reranking entirely — zero overhead for callers that don't need it.

The `retrieval_mode` field in `SearchResponse` gains a `+reranked` suffix when reranking ran (e.g. `"hybrid+reranked"`, `"sparse_only+reranked"`).

---

## Implementation Plan Context

### Task 1: `RerankerConfig` + `Reranker`

**Files:**
- Create: `src/internal/retrieval/reranker.py`
- Create: `tests/unit/retrieval/test_reranker.py`

**Background:** `SentenceTransformerReranker` lives in `src/internal/servers/retrieval/rerank.py`. Its `rerank(queries, documents, topk)` method takes `list[str]` queries and `list[list[dict]]` documents where each doc dict is plain JSON (e.g. `{"contents": "title\nbody", "doc_id": "x"}`), and returns `list[list[dict]]` where each item is `{"document": original_dict, "score": float}`. `cohere_rerank_api(query, docs, model_name, api_key)` in `src/internal/natural_language_processing/search_nlp_models.py` is `async` and returns `list[float]` (one score per passage, preserving input

…

### Task 2: Wire `Reranker` into `RetrievalService`

**Files:**
- Modify: `src/internal/retrieval/service.py` (lines 77–136 — `RetrievalService` class)
- Modify: `tests/unit/retrieval/test_service.py`

**Background:** `RetrievalService.__init__` currently takes `backend: RetrievalBackend`. `search()` returns `(results, mode)` where mode is `"hybrid"` | `"sparse_only"` | `"dense_only"`. After MMR reranking, add an optional neural rerank step. `from_env()` calls `_build_backend()` — extend it to also call `Reranker.from_env()`.

- [ ] **Step 1: Write failing tests**

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL — `RetrievalService.__init__` does not accept `reranker` param.

…

### Task 3: Extend `eval_runner.py` with `--reranker` flag and latency

**Files:**
- Modify: `src/internal/retrieval/eval_runner.py`
- Modify: `tests/unit/retrieval/test_eval_runner.py`

**Background:** `run_eval(dataset_path, service, top_k)` currently returns `{recall@k, ndcg@k, mrr, num_queries}`. Extend it to accept an optional `reranker: Reranker | None`. When provided, run reranking after retrieval, compute the same metrics on reranked results, measure wall-clock time per rerank call, and return a structured dict with `retrieval`, `reranked`, and `latency_ms` keys. The CLI gains `--reranker {local,cohere}` and `--reranker_model` args.

- [ ] **Step 1: Write failing tests**

- [ ] **Step 2: Run tests to verify they fail**

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
