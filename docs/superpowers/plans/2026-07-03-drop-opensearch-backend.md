# Drop OpenSearch Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Remove the OpenSearch document-index backend + the dependent `HybridDocumentIndex` (~5.2k LOC), keeping Weaviate/Disabled/local fully working.

**Architecture:** Delete the OpenSearch impls + Hybrid composite; relocate one constant out of the OpenSearch package; rewire factory/service/config to the remaining backends; strip references + docs + tests. `HybridCapable`/`hybrid_retrieval` stay (Weaviate implements them).

**Tech Stack:** Python 3, pytest.

**Spec:** `docs/superpowers/specs/2026-07-03-drop-opensearch-backend-design.md`.

## Global Constraints

- Weaviate (default), Disabled, and local FAISS/BM25 remain functional.
- Keep `HybridCapable` ABC + `hybrid_retrieval` (Weaviate's own hybrid).
- `import src` + document-index/retrieval/config unit suites green.

---

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
