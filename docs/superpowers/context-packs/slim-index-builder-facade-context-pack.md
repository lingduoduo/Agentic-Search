# Generated Context Pack

# Slim Index Builder Facade

## Sources

- [Specification: 2026-07-03-slim-index-builder-facade-design.md](../archive/specs/2026-07-03-slim-index-builder-facade-design.md)
- [Plan: 2026-07-03-slim-index-builder-facade.md](../archive/plans/2026-07-03-slim-index-builder-facade.md)

## Specification Context

### Overview

**Date:** 2026-07-03
**Status:** Approved (design).
**Scope:** `src/internal/document_index/index_builder.py` (the #363 back-compat
facade) + the two test files that import internals through it. Local
FAISS/BM25 indexing path only — no change to the Weaviate/OpenSearch backends,
the interface ABCs, or `servers/indexing/`.

## Implementation Plan Context

### Task 1: Repoint test imports to home modules

- [x] **Step 1:** `test_index_builder.py` — import `_Corpus` from `…faiss_io`, `_encode_batch` from `…embedding`, `_require_faiss`/`_require_torch` from `…_common`.
- [x] **Step 2:** `test_indexing_pipeline.py` — import `_split_paragraphs`, `_split_sentences_in_paragraph` from `…chunking`.
- [x] **Verify:** the three test files pass with the new imports (facade still has the privates at this point).

### Task 2: Slim the facade

- [x] **Step 1:** In `index_builder.py`, remove every `_`-prefixed name from the per-module import blocks and from `__all__`; keep all public names + `main`.
- [x] **Step 2:** `ruff check src/internal/document_index/index_builder.py` — clean.
- [x] **Verify:** `python -c "from src.internal.document_index import index_builder as m; [getattr(m,n) for n in m.__all__]"` resolves; `python -m src.internal.document_index.index_builder --help` still works.

### Task 3: Full verification

- [x] Indexing unit suites green: `test_index_builder`, `test_indexing_pipeline`, `test_rerank` (+ facade public-API importers).
- [x] Grep: zero `_`-prefixed names in the facade imports/`__all__`.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
