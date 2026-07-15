# Generated Context Pack

# Simplify Model Services

## Sources

- [Specification: 2026-07-13-simplify-model-services-design.md](../archive/specs/2026-07-13-simplify-model-services-design.md)
- [Plan: 2026-07-13-simplify-model-services.md](../archive/plans/2026-07-13-simplify-model-services.md)

## Specification Context

### Goal

Make the existing APIs reliably execute one understandable, session-aware pipeline—retrieve candidates, rank/rerank evidence, generate a grounded answer, and persist the turn—without adding new public endpoints or changing current request/response contracts.

## Implementation Plan Context

### Task 1: Session retrieval context

**Files:**
- Create: `src/internal/search_pipeline/__init__.py`
- Create: `src/internal/search_pipeline/context.py`
- Test: `tests/unit/search_pipeline/test_context.py`

**Interfaces:**
- Produces: `RetrievalContext(query, retrieval_query, history)` and `build_retrieval_context(query, history, max_messages=6)`.

- [ ] Write failing tests for standalone queries, pronoun follow-ups, continuation cues, bounded history, and exclusion of assistant tool/evidence markup.
- [ ] Run `pytest tests/unit/search_pipeline/test_context.py -q`; expect import failure.

…

### Task 2: Internal stage contracts and adapters

**Files:**
- Create: `src/internal/search_pipeline/models.py`
- Create: `src/internal/search_pipeline/stages.py`
- Test: `tests/unit/search_pipeline/test_stages.py`

**Interfaces:**
- Produces: `CandidateSet`, `RankedEvidence`, `GeneratedAnswer`; async `RetrievalStage`, `RankingStage`, `InferenceStage` protocols; adapters around existing `SearchClient`, fusion functions, rerank HTTP call, and serving model boundary.

- [ ] Write failing protocol/serialization tests using lightweight fakes.
- [ ] Run `pytest tests/unit/search_pipeline/test_stages.py -q`; expect missing modules.
- [ ] Implement normalized internal models and protocols without exporting them through FastAPI schemas.

…

### Task 3: Centralize ranking and degradation

**Files:**
- Create: `src/internal/search_pipeline/ranking.py`
- Modify: `src/internal/servers/web/app.py`
- Test: `tests/unit/search_pipeline/test_ranking.py`
- Test: `tests/unit/servers/web/test_reranking.py`

**Interfaces:**
- Produces: `DefaultRankingStage.rank(query, candidates, top_k)` with deduplication, optional reranker, MMR, and explicit metadata.

- [ ] Write failing tests proving duplicate removal, reranker ordering, MMR truncation, and preservation of pre-rerank order on timeout/error.
- [ ] Run the focused tests and observe failures against duplicated current helpers.

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
