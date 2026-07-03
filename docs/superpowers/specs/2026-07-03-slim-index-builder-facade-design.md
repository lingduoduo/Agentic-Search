# Slim the index_builder back-compat facade — design

**Date:** 2026-07-03
**Status:** Approved (design).
**Scope:** `src/internal/document_index/index_builder.py` (the #363 back-compat
facade) + the two test files that import internals through it. Local
FAISS/BM25 indexing path only — no change to the Weaviate/OpenSearch backends,
the interface ABCs, or `servers/indexing/`.

## Problem

PR #363 split the 1.5k-LOC `index_builder` into cohesive modules (`chunking`,
`embedding`, `faiss_io`, `pipeline`, `cli`, `_common`) and left `index_builder.py`
as a back-compat facade so `from …index_builder import X` and
`python -m …index_builder` keep working.

The facade re-exports **~60 names**, of which **~30 are private helpers**
(`_split_text`, `_Corpus`, `_encode_batch`, `_embed_texts`, `_require_faiss`, …).
A back-compat facade should expose the **public** API, not act as a side door
into internals. Concretely:

- ~25 of the re-exported names are never imported anywhere — dead re-exports.
- The private names that *are* used are pulled in by **tests reaching through
  the shim into internals** (`test_index_builder.py`, `test_indexing_pipeline.py`,
  `test_rerank.py`) — a boundary smell: tests should import internals from the
  module that owns them.

## Change (surgical, behavior-preserving)

1. **Facade re-exports public names only.** Keep every non-underscore name
   (configs/models, `chunk_documents`, `embed_chunks`, `run_indexing_pipeline`,
   `write_faiss_index`, `prepare_texts`, `pooling`, `load_model`, `IndexBuilder`,
   `main`, `parse_args`, the separators/constants, …) so all external public
   imports and the documented `python -m …index_builder` entrypoint keep working.
   Remove **all ~30 private (`_`-prefixed) re-exports** and drop them from
   `__all__`.

2. **Repoint the two test files** that import facade-routed privates to import
   them from the owning module instead:
   - `test_index_builder.py`: `_Corpus` → `faiss_io` (its other private uses are
     `monkeypatch` string targets to the real `cli`/`_common` modules, not
     facade imports — unchanged).
   - `test_indexing_pipeline.py`: `_split_paragraphs`,
     `_split_sentences_in_paragraph` → `chunking`.

   (`test_rerank.py` does not import the facade — it only patches
   `…retrieval.rerank._require_torch` by string — so it is untouched.)

3. **Repoint the two in-package internal modules** that reached through the
   facade into siblings — `retrieval.py` (imported `_encode_batch`,
   `_normalize_embedding_rows`, `_require_faiss`, `_require_torch` + publics) and
   `indexing.py` (public names) — to import directly from the home modules
   (`_common`, `chunking`, `embedding`, `faiss_io`, `pipeline`). After this, **no
   module inside `document_index/` depends on its own back-compat facade**; the
   facade serves only cross-package/external callers (which legitimately use its
   public API — those are left unchanged).

## Non-goals

- No signature/behavior changes to any chunking/embedding/faiss function.
- No change to the enterprise vector-DB backends, `interfaces.py` ABCs, the
  `factory`, or `servers/indexing/`.
- The facade file stays (back-compat + CLI entrypoint) — only its surface slims.

## Testing

- **Primary gate (behavior-preserving proof):** the indexing unit suites pass
  unchanged — `test_index_builder.py`, `test_indexing_pipeline.py`,
  `test_rerank.py`, plus anything importing the facade's public API.
- Grep proof: zero `_`-prefixed names remain in the facade's imports/`__all__`;
  every private the tests use is imported from its home module.
- `ruff check` clean (no unused imports left behind).

## Files touched

- **Modify:** `src/internal/document_index/index_builder.py`,
  `src/internal/document_index/retrieval.py`,
  `src/internal/document_index/indexing.py`,
  `tests/unit/test_index_builder.py`, `tests/unit/test_indexing_pipeline.py`.
