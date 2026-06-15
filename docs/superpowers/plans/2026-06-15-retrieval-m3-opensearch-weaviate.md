# Retrieval PRD — Milestone 3: OpenSearch + Weaviate Backends + CI Eval Gate

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `OpenSearchBackend` (BM25 + kNN) and `WeaviateBackend` (BM25 + nearVector) so `RETRIEVAL_BACKEND` can be set to `local|opensearch|weaviate`. Add a BEIR eval script and a CI job that fails PRs when eval metrics drop. Remove the three legacy server files.

**Architecture:** Each new backend wraps a vendor client via a monkeypatchable factory function; no live server is needed for unit tests. `service.py._build_backend()` is extended with two new branches. The CI gate reads `data/eval/baseline_metrics.json` (written by a human after the first green run) and fails if Recall@10 drops > 2pp or NDCG@10 drops > 1pp.

**Tech Stack:** Python 3.12, `opensearch-py`, `weaviate-client>=4.9`, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-06-15-retrieval-prd-design.md` Milestone 3.

**Gate:** NDCG@10 ≥ 0.45 on nfcorpus, fiqa, scifact (requires live indexes). CI green on 3 consecutive PRs.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `src/internal/retrieval/backends/opensearch.py` | BM25 `match` + kNN on OpenSearch |
| Create | `src/internal/retrieval/backends/weaviate.py` | BM25 + `nearVector` on Weaviate |
| Modify | `src/internal/retrieval/service.py` | Wire opensearch/weaviate in `_build_backend()` |
| Create | `src/internal/retrieval/beir_eval.py` | BEIR dataset download + eval runner |
| Create | `.github/workflows/eval-gate.yml` | CI job: pytest + optional eval gate |
| Modify | `src/__init__.py` | Remove lazy refs to deleted modules |
| Delete | `src/internal/servers/retrieval/retrieval_server.py` | Legacy — replaced by server.py |
| Delete | `src/internal/servers/retrieval/retrieval_rerank.py` | Legacy — replaced by eval_router.py |
| Delete | `src/internal/servers/retrieval/hybrid_rerank.py` | Legacy — replaced by eval_router.py |
| Create | `tests/unit/retrieval/test_opensearch_backend.py` | Unit tests for OpenSearchBackend |
| Create | `tests/unit/retrieval/test_weaviate_backend.py` | Unit tests for WeaviateBackend |

---

### Task 1: `backends/opensearch.py`

**Env vars:** `OPENSEARCH_URL` (e.g. `http://localhost:9200`), `OPENSEARCH_INDEX`, `OPENSEARCH_CONTENT_FIELD` (default `content`), `OPENSEARCH_VECTOR_FIELD` (default `content_vector`), `OPENSEARCH_DOC_ID_FIELD` (default `document_id`), `OPENSEARCH_TITLE_FIELD` (default `title`), `OPENSEARCH_URL_FIELD` (default `source_links`).

**Dense search:** Needs a query embedder to convert query text to a vector. Pass `embedder: Callable[[str], list[float]] | None`; if None, `search_dense` raises `NotImplementedError`.

**Hit mapping:** `_source["document_id"]` → `doc_id`, `_source["title"]` → `title`, `_source["content"]` → `text`, `_source.get("source_links")` → `url`, `_score` → `score`.

**Files:**
- Create: `src/internal/retrieval/backends/opensearch.py`
- Create: `tests/unit/retrieval/test_opensearch_backend.py`

### Task 2: `backends/weaviate.py`

**Env vars:** `WEAVIATE_URL` (e.g. `http://localhost:8080`), `WEAVIATE_COLLECTION` (e.g. `Document`).

**Weaviate v4 API:** Uses `weaviate.connect_to_local()` / `connect_to_custom()`. BM25 via `collection.query.bm25(query, limit)`. nearVector via `collection.query.near_vector(near_vector=vec, limit)`. Both return `weaviate.types.WeaviateReturn` with `.objects` list.

**Hit mapping:** `obj.uuid` → `doc_id`, `obj.properties.get("title", "")` → `title`, `obj.properties.get("content", "")` → `text`, `obj.properties.get("source_links")` → `url`, `obj.metadata.score` → `score` (may be None — fall back to `obj.metadata.distance` negated or 0.0).

**Dense search:** Same embedder pattern as OpenSearch backend.

**Files:**
- Create: `src/internal/retrieval/backends/weaviate.py`
- Create: `tests/unit/retrieval/test_weaviate_backend.py`

### Task 3: Wire backends in `service.py`

Add `opensearch` and `weaviate` branches to `_build_backend()`.

**Files:**
- Modify: `src/internal/retrieval/service.py`

### Task 4: BEIR eval script

Thin script that downloads BEIR corpora (nfcorpus, fiqa, scifact) via `ir-datasets` or direct BEIR download, runs `run_eval()` per dataset, writes results JSON, and optionally compares against a baseline.

**Files:**
- Create: `src/internal/retrieval/beir_eval.py`

### Task 5: CI eval gate + legacy removal

Create `.github/workflows/eval-gate.yml`, remove three legacy server files, clean `src/__init__.py`.

**Files:**
- Create: `.github/workflows/eval-gate.yml`
- Modify: `src/__init__.py`
- Delete: three legacy server files
