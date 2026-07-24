# Retire the dead natural_language_processing package

**Date:** 2026-07-24
**Branch:** `chore/retire-nlp-package` (off `main`)
**Status:** design approved (scope), pending spec review

## Context

A simplification audit of `natural_language_processing/` vs `document_index/` found
that the cross-package "tokenization duplication" is a mirage (the live tokenizers
serve genuinely different purposes), but that **`natural_language_processing/` is
~90% dead Onyx heritage** — superseded by `document_index/embedding.py`,
`document_index/embedding_cache.py`, `retrieval/`, and the rerank server. Its only
live thread into the running system is one ~20-line async function,
`cohere_rerank_api`, reached from `retrieval/reranker.py` when
`RERANKER_PROVIDER=cohere`.

Verified independently (grep + reachability): the **only** non-test src importer of
the entire package is `retrieval/reranker.py:18` (guarded try/except); there is no
`src/__init__.py` re-export of any NLP symbol; the `EmbeddingModel`/`RerankingModel`
grep hits elsewhere are unrelated config classes (`EmbeddingModelDetail`,
`SupportedEmbeddingModel`), not the NLP model wrappers; `CohereBillingLimitError` is
used only by `cohere_rerank_api`.

## Goal

Extract the single live function into a small module next to its caller, then delete
essentially the entire `natural_language_processing/` package (~2,300 LOC), with zero
change to live behavior. Scope (approved): retire the package; also delete the
test-only, superseded `query_embedding_cache.py` + its test.

## Scope

### Create
- `src/internal/retrieval/cohere_rerank.py` — a self-contained module holding
  `cohere_rerank_api` (verbatim body from `search_nlp_models.py:678-697`) and
  `CohereBillingLimitError` (verbatim from `exceptions.py`). Imports: `cohere`
  (`AsyncClient as CohereAsyncClient`, `cohere.core.api_error.ApiError`) and
  `document_index.utils.setup_logger` (→ `logger`). Placed in `retrieval/` because
  `retrieval/reranker.py` is its sole caller and already depends on `document_index`.

### Edit
- `src/internal/retrieval/reranker.py` — change the guarded import (lines 18-22) from
  `...natural_language_processing.search_nlp_models import cohere_rerank_api` to
  `from src.internal.retrieval.cohere_rerank import cohere_rerank_api`. Keep the
  `try/except ImportError → cohere_rerank_api = None` guard (cohere may be
  uninstalled). No other reranker change; the call site (`reranker.py:118`) is
  unchanged.

### Delete
- The entire `src/internal/natural_language_processing/` package (all files:
  `search_nlp_models.py`, `utils.py`, `constants.py`, `english_stopwords.py`,
  `_stubs.py`, `exceptions.py`, `query_embedding_cache.py`, `__init__.py`).
- `tests/unit/test_query_embedding_cache.py` (the only test that imports the package;
  the live query-embedding cache is `document_index/embedding_cache.py`).

### Explicitly out of scope
- Relocating `setup_logger`/`log_function_time` out of `document_index/utils.py` (the
  separate "misplaced helpers" smell — a later, `metrics/`-touching change). The new
  `cohere_rerank.py` keeps importing `setup_logger` from where it lives today.
- Any `document_index/` change (it is the live stack; nothing to simplify here).

## Verification / success criteria

1. `retrieval/reranker.py`'s cohere path is behavior-identical: `_rerank_cohere` still
   calls `cohere_rerank_api(...)`; the name still resolves in the `reranker` module
   namespace, so `tests/unit/retrieval/test_reranker.py`'s patch of
   `src.internal.retrieval.reranker.cohere_rerank_api` still works unchanged.
2. `python -c "import src"` succeeds; importing `retrieval.reranker` with `cohere`
   absent still degrades to `cohere_rerank_api = None` (guard intact).
3. `grep -rn "natural_language_processing" src/ tests/ examples/` returns nothing.
4. `ruff check .` + `ruff format --check .` pass; `pytest` green (only
   `test_query_embedding_cache.py` removed — no live test loses coverage).
5. The reranker factory path (`RERANKER_PROVIDER=cohere`) still resolves
   `cohere_rerank_api` from the new module.

## Risks

Low. The package is verified dead except one function with a single guarded caller.
The one subtlety — the reranker test patches a string target on the `reranker` module,
which is unaffected by moving the import source — is covered by criterion 1. Moving
`CohereBillingLimitError` into the new module is safe (no other importer). The
`cohere`-uninstalled degradation path is preserved by keeping reranker's try/except.
