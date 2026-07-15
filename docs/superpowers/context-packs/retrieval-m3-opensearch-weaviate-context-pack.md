# Generated Context Pack

# Retrieval PRD — Milestone 3: OpenSearch + Weaviate Backends + CI Eval Gate

## Sources

- [Plan: 2026-06-15-retrieval-m3-opensearch-weaviate.md](../plans/2026-06-15-retrieval-m3-opensearch-weaviate.md)

## Implementation Plan Context

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

### Task 5: Metadata filtering in OpenSearch and Weaviate backends

**PRD reference:** Section 7 — `"filters": { "source": "confluence" }` in `POST /search`. M1 Task 9 added post-hoc Python filtering to `LocalBackend`. OpenSearch and Weaviate support server-side filtering — more efficient and must be wired here.

**Files:**
- Modify: `src/internal/retrieval/backends/opensearch.py`
- Modify: `src/internal/retrieval/backends/weaviate.py`
- Modify: `tests/unit/retrieval/test_opensearch_backend.py` (append)
- Modify: `tests/unit/retrieval/test_weaviate_backend.py` (append)

**OpenSearch filtering** — wrap `match` query in `bool+filter` when `filters` is provided:

```python
def _build_sparse_body(
    query: str, content_field: str, filters: dict | None
) -> dict:
    match_clause = {"match": {content_field: query}}
    if not filters:
        return {"query": match_clause}
    return {
        "query": {
            "bool": {
                "must": [match_clause],
                "filter": [{"term": {k: v}} for k, v in filters.items()],
            }
        }
    }
```

For kNN, add a `"filter"` key inside the knn clause:

```python
def _build_dense_body(
    vec: list[float], vector_field: str, top_k: int, filters: dict | None
) -> dict:
    knn_clause: dict = {"vector": vec, "k": top_k}
    if filters:
        knn_clause["filter"] = {
            "bool": {"filter": [{"term": {k: v}} for k, v in filters.items()]}
        }
    return {"query": {"knn": {vector_field: knn_clause}}}
```

Update `search_sparse()` and `search_dense()` signatures to `(self, query, top_k, filters=None)`.

_[Section compacted.]_

### Task 6: CI eval gate + legacy removal

Create `.github/workflows/eval-gate.yml`, remove three legacy server files, clean `src/__init__.py`.

**Files:**
- Create: `.github/workflows/eval-gate.yml`
- Modify: `src/__init__.py`
- Delete: three legacy server files

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
