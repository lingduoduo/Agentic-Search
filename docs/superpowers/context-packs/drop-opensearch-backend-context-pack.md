# Generated Context Pack

# Drop Opensearch Backend

## Sources

- [Specification: 2026-07-03-drop-opensearch-backend-design.md](../specs/2026-07-03-drop-opensearch-backend-design.md)
- [Plan: 2026-07-03-drop-opensearch-backend.md](../plans/2026-07-03-drop-opensearch-backend.md)

## Specification Context

### Constraints / verification

- Weaviate, Disabled, and local backends remain fully functional; the factory's
  default (Weaviate) and `RETRIEVAL_BACKEND=local|weaviate` still resolve.
- `python -c "import src"` resolves; `document_index`/`retrieval` import surfaces load.
- `ruff check` clean; the document-index + retrieval + config unit suites green
  (minus deleted OpenSearch/Hybrid tests).
- Grep proof: zero remaining `OpenSearch*`/`opensearch` code references outside
  deliberately-kept comments.

## Implementation Plan Context

### Task 1: Break the constant dependency, then delete impls

- [x] **Step 1:** Move `DEFAULT_MAX_CHUNK_SIZE = 512` into `interfaces.py` (drop the `from ...opensearch.constants import` line).
- [x] **Step 2:** `git rm` `document_index/opensearch/` (7 files), `document_index/hybrid.py`, `retrieval/backends/opensearch.py`, `metrics/opensearch_search.py`, and their tests (`test_opensearch_client`, `test_opensearch_backend`, `test_hybrid`).
- [x] **Verify:** grep for surviving imports of the deleted modules.

### Task 2: Rewire factory / service / config

- [x] **Step 1:** `factory.py` — remove OpenSearch/Hybrid branches + helpers; keep Disabled + Weaviate default; fix docstring.
- [x] **Step 2:** `retrieval/service.py` — remove `_build_opensearch_backend` + `"opensearch"` branch; update error message.
- [x] **Step 3:** `configs/{default_config,app_configs}.py` — remove `ENABLE_OPENSEARCH_INDEXING`, `ENABLE_HYBRID_INDEXING`, `OPENSEARCH_*`.
- [x] **Step 4:** `db/enums.py` — remove the two dead OpenSearch migration enums.
- [x] **Verify:** `python -c "import src"` + factory builds Weaviate/Disabled; ruff clean.

### Task 3: Strip remaining references, docs, tests

- [x] **Step 1:** `document_index/utils.py` stub OS-metrics helpers — remove if unused (grep), else leave.
- [x] **Step 2:** Clean stale OpenSearch comments where trivial (`chunk_content_enrichment`, `disabled`, `servers/indexing/*`).
- [x] **Step 3:** `README.md` — drop OpenSearch/Hybrid from backend descriptions.
- [x] **Step 4:** Trim OpenSearch/Hybrid cases from `test_factory`, `test_imports`, `test_types`, `test_weaviate`, integration tests.
- [x] **Verify:** document-index/retrieval/config unit suites green; grep shows zero stray OpenSearch code refs.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
