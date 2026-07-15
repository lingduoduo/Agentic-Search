# Drop the OpenSearch (and dependent Hybrid) document-index backend — design

**Date:** 2026-07-03
**Status:** Approved (design).
**Scope:** Deliberately remove the config-gated OpenSearch backend and the
`HybridDocumentIndex` composite that depends on it. A product decision to shed a
~5.2k-LOC capability this repo does not run. Weaviate (default), Disabled, and
the local FAISS/BM25 path remain.

## Why

`RETRIEVAL_BACKEND=opensearch` / `ENABLE_OPENSEARCH_INDEXING` select an
OpenSearch-backed `DocumentIndex` (~5k LOC). The demo/local stack never runs it,
and the user has decided to drop it. `HybridDocumentIndex` is defined as
"OpenSearch for keyword/KV + Weaviate for vector" — it **cannot function without
OpenSearch**, so it goes too.

## Key distinction (kept vs removed)

- **Removed:** the `HybridDocumentIndex` *composite backend* (`hybrid.py`).
- **Kept:** the `HybridCapable` ABC and `hybrid_retrieval` method — Weaviate
  implements its **own** in-store hybrid (`weaviate_document_index.py:290`), so
  the capability and interface stay.

## Delete (~5,200 LOC)

- `src/internal/document_index/opensearch/` (7 files, 4884 LOC).
- `src/internal/document_index/hybrid.py` (`HybridDocumentIndex`).
- `src/internal/retrieval/backends/opensearch.py` (138) + `src/internal/metrics/opensearch_search.py`.
- Tests: `test_opensearch_client`, `test_opensearch_backend`, `test_hybrid`.

## Rewire (reference cleanup)

1. **Relocate constant:** `DEFAULT_MAX_CHUNK_SIZE` lives in
   `opensearch/constants.py` but is imported by the kept `interfaces.py`. Move
   the definition into `interfaces.py` (its only surviving consumer).
2. **`factory.py`:** drop `_is_opensearch_enabled` / `_is_hybrid_enabled` and the
   OpenSearch/Hybrid builder branches; keep Disabled + Weaviate (default).
   Update the module docstring.
3. **`retrieval/service.py`:** remove `_build_opensearch_backend` and the
   `name == "opensearch"` branch; error message → `Supported values: local, weaviate`.
4. **`configs/{default_config,app_configs}.py`:** remove `ENABLE_OPENSEARCH_INDEXING`,
   `ENABLE_HYBRID_INDEXING`, and `OPENSEARCH_*` settings.
5. **`db/enums.py`:** remove the two dead enums `OpenSearchDocumentMigrationStatus`,
   `OpenSearchTenantMigrationStatus` (zero non-test references).
6. **`document_index/utils.py`:** the stub `observe_/record_/track_opensearch_search`
   metrics helpers — remove if unused after the deletes; keep + rename-neutral only
   if a kept module calls them (verify at implementation).
7. **Comment-only references** (`chunk_content_enrichment.py`, `disabled.py`,
   `servers/indexing/{models,vector_db_insertion}.py`, `metrics/indexing_pipeline.py`
   Celery queue) — clean the stale OpenSearch mentions where trivial; leave
   functional queue names that are out of scope.
8. **`README.md`:** drop OpenSearch/Hybrid from the retrieval-backend descriptions.
9. **Tests:** trim OpenSearch/Hybrid cases from `test_factory`, `test_imports`,
   `test_types`, `test_weaviate`, and the integration tests.

## Constraints / verification

- Weaviate, Disabled, and local backends remain fully functional; the factory's
  default (Weaviate) and `RETRIEVAL_BACKEND=local|weaviate` still resolve.
- `python -c "import src"` resolves; `document_index`/`retrieval` import surfaces load.
- `ruff check` clean; the document-index + retrieval + config unit suites green
  (minus deleted OpenSearch/Hybrid tests).
- Grep proof: zero remaining `OpenSearch*`/`opensearch` code references outside
  deliberately-kept comments.

## Non-goals

- No change to Weaviate, the local path, `HybridCapable`/`hybrid_retrieval`, or
  any non-OpenSearch subsystem.
