# Prune dead index/document leaf modules (PR1 of 3)

**Date:** 2026-07-24
**Branch:** `chore/prune-dead-index-leaf-modules` (off `origin/main`)
**Status:** design approved, pending spec review

## Context

This is **PR1 of a 3-PR dead-code-removal campaign** simplifying the indexing /
document-store side of Agentic-Search. A reachability audit found ~6,900 LOC of
Onyx-heritage code with no live callers. The campaign, safest-first:

1. **PR1 (this spec)** — three leaf modules with zero live importers + one stale doc block.
2. **PR2** — the orphaned Onyx ingestion cluster (`backgroundworker/*`,
   `servers/indexing/*`, connector classes, `chunk_content_enrichment.py`).
3. **PR3** — Weaviate entirely (dead write path + opt-in query backend + `RETRIEVAL_BACKEND=weaviate` wiring).

PR4 (ingestion DB tables + coupled admin routers) is **deferred** — it changes the
admin API surface and is a separate decision.

Each PR is standalone, branches off `origin/main`, and carries its own spec + plan.

## Goal

Delete two dead leaf modules and fix one dangling documentation reference, with
**zero change to live runtime behavior**. This PR is deliberately the low-risk
warm-up: every target has zero non-test importers, verified below.

## Scope

### Delete — source modules (155 LOC)

| File | LOC | Evidence it's dead |
|------|-----|--------------------|
| `src/internal/retrieval/indexer.py` | 115 | Dead duplicate FAISS-HNSW builder. Zero non-test importers. Superseded by the live `document_index` build tool (`index_builder`/`cli`/`faiss_io`). |
| `src/internal/document_index/document_metadata.py` | 40 | Exports only `DocumentMetadata` (zero src importers). Only reference is a smoke-test import. Not re-exported by the package `__init__`. |

**Scope correction (2026-07-24):** `document_index/indexing.py` (402 LOC) was
originally slated for this PR but is **moved to PR2**. It is not a clean leaf: the
package `__init__.py` lazily re-exports its symbols via `__getattr__`
(`_INDEXING_EXPORTS`), and its only non-test consumer is
`backgroundworker/docprocessing.py` — a PR2 target. Deleting it in isolation
breaks `from src.internal.document_index import <symbol>` and leaves the
`_INDEXING_EXPORTS` block dangling. It must be removed together with
`docprocessing.py`, the facade tests, and the lazy-export block, all in PR2. The
original audit's plain-grep missed the lazy re-export.

### Delete / edit — tests

- Delete `tests/unit/retrieval/test_indexer.py` (tests only the deleted `indexer.py`).
- Edit `tests/unit/document_index/test_imports.py`: remove `test_document_metadata_importable`.
  Leave `test_chunk_content_enrichment_importable` — that module is a **PR2** target, not this one.

### Fix — documentation

- `docs/retrieval.md` ~L468-476: remove the "Build an IVF-PQ FAISS index" example.
  It imports `from src.internal.retrieval.index_optimizer import FAISSIndexBuilder`,
  a module already deleted in the RRF-consolidation work (PR #370). The reference
  is dangling — no such module exists.

### Explicitly out of scope

- `chunk_content_enrichment.py`, `servers/indexing/*`, workers, connectors → PR2.
- All Weaviate code → PR3.
- Ingestion DB tables + `connectors`/`documents` routers → deferred PR4.
- `embedding_cache.py` — the audit traced it as **live** (`document_index/retrieval.py:12`
  imports it), despite an archived plan listing it for deletion. Leave it untouched;
  flag the discrepancy for a later look, do not act on it here.

## Verification / success criteria

1. `grep -rn` for each deleted module's import path across `src/` and `examples/`
   returns nothing before deletion (already confirmed).
2. `ruff check .` passes (no orphaned imports introduced).
3. `pytest` is green — the only removed tests are those that exercised the deleted
   modules; no live test loses coverage of live code.
4. The retrieval server stacks (`demo`, `hybrid`, `server`) and the documented
   `python -m src.internal.document_index.index_builder` build command are unaffected
   (none imported the deleted modules).

## Risks

Minimal. All deletions are import-reachability-verified dead. The one judgment call
is the doc edit (removing rather than rewriting the IVF-PQ example) — acceptable
because the underlying module no longer exists and nothing replaced it.
