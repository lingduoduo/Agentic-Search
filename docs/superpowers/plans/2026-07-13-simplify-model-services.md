# Existing Search Pipeline Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify the internals behind existing APIs into a session-aware retrieve → rank/rerank → grounded inference pipeline.

**Architecture:** Internal stage protocols and adapters isolate retrieval, ranking, and inference payloads while `/api/agent`, `/api/agent/stream`, `/retrieve`, `/search`, and `/rerank` stay unchanged. Async ingestion remains the producer of searchable indexes.

**Tech Stack:** Python, FastAPI, Pydantic, asyncio, pytest.

## Global Constraints

- Add no public endpoint and change no existing request or response schema.
- Preserve async indexing and every existing retrieval/reranking server entry point.
- Search inference must not answer without evidence.
- Optional reranking failure preserves the best prior ordering.
- Access filters continue to apply to internal retrieval.

---

### Task 1: Session retrieval context

**Files:**
- Create: `src/internal/search_pipeline/__init__.py`
- Create: `src/internal/search_pipeline/context.py`
- Test: `tests/unit/search_pipeline/test_context.py`

**Interfaces:**
- Produces: `RetrievalContext(query, retrieval_query, history)` and `build_retrieval_context(query, history, max_messages=6)`.

- [ ] Write failing tests for standalone queries, pronoun follow-ups, continuation cues, bounded history, and exclusion of assistant tool/evidence markup.
- [ ] Run `pytest tests/unit/search_pipeline/test_context.py -q`; expect import failure.
- [ ] Implement deterministic context construction; preserve the original query and prepend only the most recent relevant user topic to follow-ups.
- [ ] Rerun the test and expect all cases to pass.
- [ ] Commit with `git commit -m "feat: add session retrieval context"`.

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
- [ ] Implement adapters that translate existing endpoint payloads and preserve filters, scores, provider metadata, and citations.
- [ ] Rerun tests and commit with `git commit -m "refactor: define search pipeline stages"`.

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
- [ ] Move web-layer ranking decisions behind `DefaultRankingStage`; retain `RetrievalService`'s backend RRF behavior and existing `/rerank` payload.
- [ ] Run `pytest tests/unit/search_pipeline/test_ranking.py tests/unit/servers/web/test_reranking.py -q` and expect pass.
- [ ] Commit with `git commit -m "refactor: centralize candidate ranking"`.

### Task 4: Compose the existing API query pipeline

**Files:**
- Create: `src/internal/search_pipeline/pipeline.py`
- Modify: `src/internal/servers/web/app.py`
- Test: `tests/unit/search_pipeline/test_pipeline.py`
- Test: `tests/unit/test_execution_fallbacks.py`
- Test: `tests/unit/servers/web/test_web_experience_app.py`

**Interfaces:**
- Produces: `SearchPipeline.run(query, history, filters, top_k, source_provider)` returning the existing `(answer, citations, documents, intent, extra)` contract.

- [ ] Write failing end-to-end fake-stage tests for successful evidence, empty evidence, unreachable retrieval, reranker degradation, and inference failure.
- [ ] Run focused tests and verify failures before integration.
- [ ] Compose context → retrieval → ranking → inference while retaining existing provider precedence and deterministic no-evidence responses.
- [ ] Replace duplicated web orchestration calls with the pipeline without changing routes or response models.
- [ ] Run focused tests and expect all to pass.
- [ ] Commit with `git commit -m "refactor: compose session-aware search pipeline"`.

### Task 5: Persistence, streaming parity, and compatibility

**Files:**
- Modify: `src/internal/servers/web/app.py`
- Modify: `src/internal/servers/web/request_capture.py`
- Test: `tests/unit/servers/web/test_sse_streaming.py`
- Test: `tests/unit/servers/web/test_web_experience_app.py`
- Test: `tests/unit/servers/retrieval/test_demo_retrieval.py`
- Test: `tests/unit/servers/retrieval/test_new_server.py`
- Test: `tests/unit/test_rerank.py`

**Interfaces:**
- Consumes: pipeline stage metadata.
- Produces: consistent capture metadata while preserving existing JSON and SSE schemas.

- [ ] Write failing tests for stage metadata, persisted citations/documents, and identical final data in streaming/non-streaming requests.
- [ ] Implement capture/persistence mapping through the existing `_finalize_response` path.
- [ ] Run all listed endpoint compatibility tests and expect pass.
- [ ] Commit with `git commit -m "feat: trace search pipeline stages"`.

### Task 6: Documentation and regression verification

**Files:**
- Modify: `README.md`
- Modify: `docs/api-reference.md`
- Modify: `docs/architecture.md`
- Modify: `docs/retrieval.md`
- Modify: `docs/request-routing.md`
- Modify: `docs/training-and-evaluation.md`

**Interfaces:**
- Documents: async indexing → session retrieval → ranking/reranking → grounded inference behind unchanged APIs.

- [ ] Update maintained docs and explicitly state that no new API was introduced.
- [ ] Run `pytest tests/unit/search_pipeline tests/unit/servers/retrieval tests/unit/servers/web/test_reranking.py tests/unit/test_rerank.py tests/unit/test_execution_fallbacks.py tests/unit/servers/web/test_web_experience_app.py -q`.
- [ ] Run `ruff check src/internal/search_pipeline src/internal/servers/web/app.py`.
- [ ] Run `ruff format --check src/internal/search_pipeline tests/unit/search_pipeline`.
- [ ] Run `git diff --check` and validate local Markdown links.
- [ ] Commit with `git commit -m "docs: explain existing search pipeline"`.
