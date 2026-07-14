# Task 4 report

## Summary

- Added `SearchPipeline`, which builds bounded follow-up context and composes retrieval, ranking, and grounded inference behind the existing five-value result tuple.
- Added deterministic empty/unreachable handling and an evidence-only fallback when inference fails.
- Adapted the existing web hybrid provider policy into the stage contracts. Provider precedence, filters, route response models, and hybrid ranking remain owned by the existing web helpers.
- Removed a runtime-only serving protocol import from `stages.py` to prevent a collection-time web/search-pipeline import cycle.

## TDD evidence

- Red: `pytest tests/unit/search_pipeline/test_pipeline.py -q` failed during collection because `src.internal.search_pipeline.pipeline` did not exist.
- Green: the focused and adjacent verification below passes.

## Verification

- `pytest tests/unit/search_pipeline tests/unit/servers/web/test_reranking.py tests/unit/test_execution_fallbacks.py tests/unit/servers/web/test_web_experience_app.py -q` — 88 passed.
- `ruff check src/internal/search_pipeline src/internal/servers/web/app.py tests/unit/search_pipeline/test_pipeline.py` — passed.
- `ruff format --check src/internal/search_pipeline tests/unit/search_pipeline` — passed.
- `git diff --check` — passed.

## Concerns

- None known. Public routes and response models are unchanged; stage metadata is carried only in the existing `extra` mapping.
