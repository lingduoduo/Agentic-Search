# Task 3 report

## Implementation

- Added `DefaultRankingStage` to centralize stable deduplication, optional reranking,
  reranker timeout/error degradation, MMR truncation, document reindexing, and
  explicit ranking metadata.
- Routed direct, hybrid-finalization, retrieval, and browser web ranking paths
  through the shared stage.
- Reused `RerankHTTPRankingStage`, preserving the standalone `/rerank` request
  adapter and leaving `RetrievalService` weighted-RRF ownership unchanged.
- Kept `_rerank_documents` as a compatibility wrapper with no ranking policy of
  its own.
- Made malformed reranker indices skippable, matching the former web behavior.

## TDD evidence

- Red: `pytest tests/unit/search_pipeline/test_ranking.py -q` failed during
  collection because `src.internal.search_pipeline.ranking` did not exist.
- Intermediate red: the reranker-order test demonstrated that MMR consumes
  reranker scores, and compatibility tests exposed metadata and malformed-index
  behavior at the centralized boundary.
- Green: focused and adjacent ranking/web tests passed (32 tests).

## Verification

- `pytest tests/unit/search_pipeline/test_ranking.py tests/unit/servers/web/test_reranking.py -q`
- `ruff check src/internal/search_pipeline/ranking.py src/internal/search_pipeline/stages.py src/internal/servers/web/app.py tests/unit/search_pipeline/test_ranking.py tests/unit/servers/web/test_reranking.py`
- `git diff --check`

## Concerns

- None known. The compatibility wrapper now returns centralized ranking metadata
  (`source_provider` and `mmr_rank`) while preserving document order and scores on
  reranker degradation.

## Review fixes

- Added non-monotonic-score timeout/error regressions. On degradation the stage
  now preserves deduplicated input order exactly and only truncates to `top_k`;
  it does not invoke score-based MMR.
- Added a web rerank payload assertion and restored the prior request body:
  document contents are `<title>\n<content>`, `return_scores` remains enabled,
  and `rerank_topk` is omitted. The standalone adapter defaults remain unchanged.
- Red evidence: the focused suite failed three tests before implementation (both
  degradation variants reordered by score, and the web payload used plain content
  plus `rerank_topk`).
- Green evidence: focused suite passed 10 tests; focused plus adjacent ranking,
  browser, and hybrid suites passed 32 tests. Ruff and `git diff --check` passed.
