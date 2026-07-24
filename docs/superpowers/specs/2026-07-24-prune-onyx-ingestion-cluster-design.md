# Prune the orphaned Onyx ingestion cluster (PR2 of 3)

**Date:** 2026-07-24
**Branch:** `chore/prune-onyx-ingestion-cluster` (off `origin/main`)
**Status:** design approved (campaign-level), pending spec review

## Context

PR2 of the 3-PR indexing/document-store simplification campaign. PR1 (#461)
removed leaf modules. This PR removes the **orphaned Onyx-heritage asynchronous
ingestion cluster** — the background worker fleet, the `servers/indexing`
pipeline, the connector *classes*, and the two `document_index` modules that only
that cluster consumed. PR3 removes Weaviate.

A dependency-edge audit confirmed the cluster is dead to live code, with exactly
**two severable edges** into it and a set of package `__init__` re-export blocks
that must be trimmed. `web/app.py` has zero cluster imports.

## Goal

Delete ~5,800 LOC of dead ingestion code with **zero change to live runtime
behavior**. The retrieval servers (`demo`/`hybrid`/`server`), the web backend, the
example CLI, and the documented `index_builder` build command do not use any of it.

## What stays (and why)

- **`connectors/models.py`** — LIVE. `Document`, `ConnectorFailure`,
  `ConnectorStopSignal`, `ConnectorCheckpoint`, `HierarchyNode`, `SlimDocument`
  are imported by `document_index/*` and `natural_language_processing`. It depends
  only on stdlib (not on `basic`/`interface`/`web`). Some symbols
  (`IndexingDocument`, `Section`, `SectionType`) become dead-but-harmless once the
  cluster is gone — the file stays intact.
- **The web connectors router** `servers/connectors/api.py` — LIVE and DB-only.
  `create_connectors_router` touches `AgenticSearchStore`/`ConnectorConfig`/
  `StoredDocument`, **never** connector classes. Untouched.
- **`document_index/__init__.py` `_BUILDER_EXPORTS`** — points at the surviving
  `index_builder.py`. Only the `_INDEXING_EXPORTS` half is removed.
- **`docker-compose.yml` `weaviate` service** — backs the PR3 backend, not this
  cluster. Only `worker-light`/`worker-heavy` are removed here.
- **`document_index` build tool** (`index_builder`/`cli`/`chunking`/`embedding`/
  `pipeline`/`faiss_io`) — the documented offline index builder. Untouched.

## Scope

### Files to DELETE (~5,828 LOC + 7 test files)

| Path | ~LOC | Note |
|------|------|------|
| `src/internal/servers/backgroundworker/` (whole dir) | 1,942 | No `__main__` guard; nothing launches it |
| `src/internal/servers/indexing/` (whole dir, recursive) | 2,288 | No outside-cluster importer |
| `src/internal/connectors/basic.py` | — | connector classes |
| `src/internal/connectors/interface.py` | — | connector ABCs |
| `src/internal/connectors/web.py` | — | `WebConnector`, tests-only |
| (basic+interface+web together) | 1,094 | |
| `src/internal/document_index/chunk_content_enrichment.py` | 102 | only `servers/indexing/embedder.py` used it |
| `src/internal/document_index/indexing.py` | 402 | only reached via the lazy `_INDEXING_EXPORTS` path |

Test files to delete (exercise only cluster code):
`tests/unit/servers/backgroundworker/test_docprocessing.py`,
`test_heavy_worker.py`, `test_indexing_pipeline_facade.py`,
`tests/unit/test_connectors.py`, `tests/unit/test_connectors_poll_slim.py`,
`tests/unit/test_indexing_server_facade.py`,
`tests/unit/document_index/test_chunk_content_enrichment.py`.

### Files to EDIT (sever edges — do NOT delete)

| File | Edit |
|------|------|
| `src/__init__.py` | Remove the eager connector-**class** imports/re-exports (~lines 50–81). KEEP the model-backed ones: `Document`, `ConnectorFailure`, `ConnectorCheckpoint`, `HierarchyNode`, `SlimDocument` (these come from `models.py`). |
| `src/internal/connectors/__init__.py` | Drop the `.basic`/`.interface` re-exports; keep only the `.models` block. |
| `src/internal/document_index/__init__.py` | Remove the `_INDEXING_EXPORTS` set and its `__getattr__` branch. KEEP `_BUILDER_EXPORTS` and its branch. |
| `src/internal/servers/web/debug_router.py` | Remove the request-time `MonitoringWorker`/`MonitoringConfig` import (~line 73) and gut the `/workers` handler body so it returns its existing null-safe shape (`{"metrics": null}`). Keep the endpoint. |
| `docker/docker-compose.yml` | Remove the `worker-light` and `worker-heavy` services. KEEP `weaviate`. |
| `tests/unit/document_index/test_imports.py` | Remove `test_chunk_content_enrichment_importable`. Keep the rest. |
| `docs/ingestion.md`, `docs/architecture.md` | Update prose/tree references to the removed dirs. |

### Explicitly out of scope

- All Weaviate code → PR3.
- Ingestion DB tables + `connectors`/`documents` routers → deferred PR4.
- `connectors/models.py` and the `document_index` build tool → stay.

## Deletion order (sever edges before deleting so nothing dangles)

1. Edit the edge files: `src/__init__.py`, `connectors/__init__.py`,
   `document_index/__init__.py`, `debug_router.py`, `docker-compose.yml`,
   `test_imports.py` trim. After this, no live/lazy edge into the cluster remains.
2. Delete the fully-cluster test files.
3. Delete `backgroundworker/` (top-of-chain).
4. Delete `servers/indexing/` (its `embedder` is the sole `chunk_content_enrichment` consumer).
5. Delete `connectors/basic.py`, `web.py`, then `interface.py` (basic/web subclass interface).
6. Delete `chunk_content_enrichment.py` and `indexing.py` (their consumers are now gone).

## Verification / success criteria

1. `python -c "import src"` succeeds after the `src/__init__.py` edit (the eager
   re-export edge is the one that would break package import).
2. Before deleting connector classes, `grep -rn "from src import <Class>"` across
   `src/` (non-test) confirms no live consumer of a removed re-export name.
3. `ruff check .` and `ruff format --check .` pass.
4. `pytest` is green — only cluster tests are removed; no live test loses coverage.
5. Retrieval servers, web backend, and the documented `index_builder` command are
   unaffected (none import the cluster).

## Risks

Medium — larger and edit-bearing (not pure deletion). The two real hazards are the
eager `src/__init__.py` re-export (breaks `import src` if the class deletion lands
before the edit — mitigated by ordering edges first) and the `debug_router`
`/workers` consumer (mitigated by gutting to the existing null-safe path). Both are
caught by criteria 1–2. `connectors/models.py` gaining dead-but-harmless symbols is
acceptable and noted for a later audit pass.
