# Generated Context Pack

# Retrieval PRD — Milestone 3: OpenSearch + Weaviate Backends + CI Eval Gate

## Sources

- [Plan: 2026-06-15-retrieval-m3-opensearch-weaviate.md](../archive/plans/2026-06-15-retrieval-m3-opensearch-weaviate.md)

## Implementation Plan Context

### Task 1: `backends/opensearch.py`

**Env vars:** `OPENSEARCH_URL` (e.g. `http://localhost:9200`), `OPENSEARCH_INDEX`, `OPENSEARCH_CONTENT_FIELD` (default `content`), `OPENSEARCH_VECTOR_FIELD` (default `content_vector`), `OPENSEARCH_DOC_ID_FIELD` (default `document_id`), `OPENSEARCH_TITLE_FIELD` (default `title`), `OPENSEARCH_URL_FIELD` (default `source_links`).

**Dense search:** Needs a query embedder to convert query text to a vector. Pass `embedder: Callable[[str], list[float]] | None`; if None, `search_dense` raises `NotImplementedError`.

**Hit mapping:** `_source["document_id"]` → `doc_id`, `_source["title"]` → `title`, `_source["content"]` → `text`, `_source.get("source_links")` → `url`, `_score` → `score`.

…

### Task 2: `backends/weaviate.py`

**Env vars:** `WEAVIATE_URL` (e.g. `http://localhost:8080`), `WEAVIATE_COLLECTION` (e.g. `Document`).

**Weaviate v4 API:** Uses `weaviate.connect_to_local()` / `connect_to_custom()`. BM25 via `collection.query.bm25(query, limit)`. nearVector via `collection.query.near_vector(near_vector=vec, limit)`. Both return `weaviate.types.WeaviateReturn` with `.objects` list.

**Hit mapping:** `obj.uuid` → `doc_id`, `obj.properties.get("title", "")` → `title`, `obj.properties.get("content", "")` → `text`, `obj.properties.get("source_links")` → `url`, `obj.metadata.score` → `score` (may be None — fall back to `obj.metadata.distance` negated or 0.0).

…

### Task 3: Wire backends in `service.py`

Add `opensearch` and `weaviate` branches to `_build_backend()`.

**Files:**
- Modify: `src/internal/retrieval/service.py`

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
