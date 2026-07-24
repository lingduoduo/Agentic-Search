# Remove Weaviate entirely (PR3 of 3)

**Date:** 2026-07-24
**Branch:** `chore/prune-weaviate-entirely` (off `main`, post-PR1/PR2 merge)
**Status:** design approved (campaign-level + two scope decisions), pending spec review

## Context

PR3 (final) of the indexing/document-store simplification campaign. PR1 (#461)
removed leaf modules; PR2 (#462) removed the async ingestion cluster. This PR
removes **Weaviate entirely** — the dead write path, the opt-in query backend, and
all wiring — per the user's explicit "remove Weaviate entirely" choice.

A dependency-edge audit confirmed the Weaviate cluster is reachable **only from
tests plus one env-gated branch in `retrieval/service.py`**. Unlike PR2 there are
**no re-export traps**: neither `src/__init__.py` nor `document_index/__init__.py`
nor any backend registry exposes a Weaviate/factory/disabled symbol. The default
`RETRIEVAL_BACKEND=local` path is fully independent.

Two scope decisions (approved):
- **Include `document_index/interfaces.py`** — the `DocumentIndex` ABC + capability
  protocols become 100% dead once factory/disabled/weaviate go (those are its only
  importers). Delete it in this PR for a complete removal.
- **Config: fix docstrings only** — keep `VectorDbSettings`/`DISABLE_VECTOR_DB`
  wired into `AppSettings` (removing risks breakage); just correct the now-stale
  "Weaviate" docstrings. No behavioral change.

## Goal

Delete ~1,430 LOC of dead Weaviate + now-orphaned interface code and drop the
`weaviate-client` dependency, with **zero change to live runtime behavior**. The
retrieval servers, web backend, and the default `local` retrieval backend do not
use any of it.

## Scope

### Files to DELETE (~1,430 LOC src + 813 LOC tests)

| Path | ~LOC | Why dead |
|------|------|----------|
| `src/internal/document_index/weaviate/` (whole dir) | 623 | `WeaviateDocumentIndex` referenced only by `factory.py` + tests |
| `src/internal/document_index/factory.py` | 92 | `get_default_document_index`/`get_all_document_indices` have zero `src/` callers — only tests. Orphan even pre-PR3. |
| `src/internal/document_index/disabled.py` | 103 | `DisabledDocumentIndex` imported only by `factory.py` (deleted) + tests |
| `src/internal/retrieval/backends/weaviate.py` | 120 | `WeaviateBackend` referenced only by the `service.py` weaviate branch + tests |
| `src/internal/document_index/interfaces.py` | 491 | `DocumentIndex` ABC + capability protocols; only importers are the three files above |

Test files to delete: `tests/unit/document_index/test_weaviate.py`,
`test_factory.py`, `test_disabled.py`, `tests/unit/retrieval/test_weaviate_backend.py`.

### Files to EDIT (do NOT delete)

| File | Edit |
|------|------|
| `src/internal/retrieval/service.py` | Remove `_build_weaviate_backend()` and the `weaviate` branch in `_build_backend()`; update the unsupported-backend error message to list `local` only. The default `local` path (`_build_local_backend()`) is untouched and must still construct via `RetrievalService.from_env()`. |
| `docker/docker-compose.yml` | Remove the `weaviate` service block, the `weaviate_data` volume (declaration + mount), the `WEAVIATE_*` env vars in the shared `&app-env` anchor, and trim the header comment that lists `weaviate`. Do NOT touch `postgres`/`redis`/`retrieval`/`web` — none `depends_on` weaviate. |
| `requirements.txt` | Remove `weaviate-client>=4.9.0`. |
| `requirements-unit-test.txt` | Remove `weaviate-client>=4.9.0`. |
| `tests/unit/document_index/test_imports.py` | Remove `test_disabled_importable` and `test_interfaces_importable` (both import deleted modules). Keep `test_document_index_utils_importable`. |
| `src/internal/configs/app_configs.py` | Fix the now-stale "Weaviate" docstring on `VectorDbSettings` (and any `disable_vector_db` doc). KEEP the class and its `AppSettings` wiring — docstring-only. |

### Explicitly out of scope

- `document_index/models.py` and its types (`DocMetadataAwareIndexChunk`,
  `IndexFilters`, `InferenceChunk`, `QueryType`, `Embedding`, `EmbeddingPrecision`)
  — used broadly across context/, chat/, db/, search_pipeline/. Stay live.
- Removing `VectorDbSettings`/`DISABLE_VECTOR_DB` behavior — deferred (vestigial but
  wired; docstring-only touch here).
- The `retrieval/backends/base.py` `RetrievalBackend`/`RetrievalResult` — used by
  `local.py` + `service.py`. Stay.

## Deletion / edit order

1. **Edit `service.py`** — remove the weaviate branch + fix error message. This
   severs the last non-test `src/` edge into the cluster. Verify `local` default
   still builds.
2. **Delete `retrieval/backends/weaviate.py`** + its test.
3. **Delete `document_index/weaviate/`, `factory.py`, `disabled.py`** + their tests;
   trim `test_disabled_importable` from `test_imports.py`.
4. **Delete `document_index/interfaces.py`** (now fully dead) + trim
   `test_interfaces_importable` from `test_imports.py`.
5. **Edit `docker-compose.yml`** (service + volume + env + header comment).
6. **Edit `requirements.txt` + `requirements-unit-test.txt`** (drop weaviate-client).
7. **Edit `app_configs.py`** docstrings.

## Verification / success criteria

1. `python -c "import src"` succeeds throughout.
2. After the `service.py` edit, `RetrievalService.from_env()` with default env
   (`RETRIEVAL_BACKEND` unset → `local`) constructs correctly; a `weaviate` value
   now raises the updated "Supported: local" error.
3. Before deleting `interfaces.py`, `grep -rn` confirms its only importers were the
   already-deleted weaviate/factory/disabled files.
4. `ruff check .` + `ruff format --check .` pass.
5. `pytest` green — only Weaviate-cluster tests removed.
6. No surviving `import weaviate` / `weaviate-client` reference (dependency safely
   dropped).
7. Retrieval servers (`demo`/`hybrid`/`server`) and web backend unaffected.

## Risks

Low–medium. The audit found no re-export traps and confirmed the default backend
path is weaviate-independent. The main care points: (a) the `service.py` edit must
preserve the `local` default and leave a correct error for unknown backends
(criterion 2); (b) `interfaces.py` must be deleted only after its three importers
are gone (criterion 3 + order). Dropping `weaviate-client` is safe once no module
imports it (criterion 6).
