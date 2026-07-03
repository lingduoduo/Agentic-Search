# Slim index_builder Facade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Reduce the `index_builder` back-compat facade to its public API and stop tests from reaching through it into internals.

**Architecture:** The facade keeps re-exporting the public (non-underscore) surface from the cohesive modules; all private re-exports are removed. The three tests that used facade-routed privates import them from the owning module instead.

**Tech Stack:** Python 3, pytest. No new deps.

**Spec:** `docs/superpowers/specs/2026-07-03-slim-index-builder-facade-design.md`.

## Global Constraints

- **Behavior-preserving.** No public API removed; the documented `python -m …index_builder` entrypoint and every external public import keep working.
- **Local FAISS/BM25 path only.** No touch to enterprise backends, `interfaces.py`, `factory`, or `servers/indexing/`.
- **Green gate:** the indexing unit suites pass unchanged.

---

## File Structure

- **Modify** `src/internal/document_index/index_builder.py` — drop all `_`-prefixed re-exports from the import blocks and `__all__`; keep public names.
- **Modify** `tests/unit/test_index_builder.py`, `tests/unit/test_indexing_pipeline.py` — import the used private helpers from their home modules.

---

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
