# Existing Search Pipeline Simplification Design

## Goal

Make the existing APIs reliably execute one understandable, session-aware pipeline—retrieve candidates, rank/rerank evidence, generate a grounded answer, and persist the turn—without adding new public endpoints or changing current request/response contracts.

## Existing API contract

This work preserves:

- `POST /api/agent` and `POST /api/agent/stream` as the public session-aware orchestration APIs;
- `/retrieve` and `/search` as retrieval-server APIs;
- `/rerank` as the optional standalone reranking API;
- existing health, session, feedback, and administrative endpoints;
- existing request and response bodies.

No `/v1/retrieve`, `/v1/rank`, `/v1/generate`, or new orchestration endpoint will be introduced.

## Actual flow

```text
Asynchronous ingestion/indexing
  connectors/uploads → background jobs → parse/chunk/embed → searchable indexes
                                                        ↓
/api/agent or /api/agent/stream
  → load bounded session history
  → build retrieval context from current query + relevant prior turns
  → call existing retrieval API for candidates
  → fuse sparse/dense/query-variant rankings
  → apply MMR and optional /rerank model
  → generate only from session history + ranked evidence
  → persist query, answer, citations, documents, and execution metadata
```

Async indexing is upstream data preparation, not part of a query request. Query-time retrieval reads the indexes it produces.

## Internal boundaries

The refactor introduces internal interfaces, not public APIs:

- `RetrievalStage.retrieve(query, history, filters, top_k)` returns normalized candidates and retrieval metadata.
- `RankingStage.rank(query, candidates, top_k)` owns deduplication, RRF/MMR ordering, and optional external or in-process reranking.
- `InferenceStage.generate(query, history, evidence)` produces a grounded answer and citations.
- `SearchPipeline.run(...)` composes the stages and returns the existing internal tuple consumed by response finalization.

Each stage can continue using current HTTP services or in-process implementations. The web endpoint should not know their payload conversion details.

## Session-aware retrieval

The retrieval stage receives bounded persisted history. A deterministic context builder decides whether prior turns affect retrieval:

- standalone queries use the current query unchanged;
- follow-ups with pronouns or continuation cues include the most recent user topic;
- generated retrieval text remains bounded and never includes assistant tool markup or full evidence dumps;
- access filters remain attached only to internal retrieval calls;
- the original user query remains available for answer generation and observability.

This reuses sessions without blindly concatenating every prior message.

## Ranking and reranking ownership

Ranking is a single internal stage with this order:

1. normalize and deduplicate candidates;
2. retain backend sparse/dense fusion already performed by `RetrievalService`;
3. apply cross-provider/query-variant fusion when multiple candidate sets exist;
4. apply optional cross-encoder/Cohere `/rerank` scoring;
5. apply MMR diversity selection and truncate to `top_k`.

The stage records which operations ran and degrades to the previous valid ordering if optional reranking fails. It must not lose candidates solely because the reranking service is unavailable.

## Grounded inference

Inference receives the original query, bounded conversation history, and final ranked evidence. Search-path generation may answer only from that evidence. Empty evidence returns the existing no-results/unreachable response; it does not invoke internal model knowledge as a substitute.

Explicit conversational modes retain their current behavior, but their retrieval and evidence formatting should reuse the shared stages where compatible.

## Persistence and observability

The existing `AgenticSearchStore` remains the persistence layer. Finalization stores the user and assistant messages as today. Routing metadata additionally records a consistent stage summary:

- retrieval query and provider;
- candidate count;
- ranking operations and final evidence count;
- reranker used or degradation reason;
- inference mode/model;
- citations and document IDs associated with the answer.

SSE and non-streaming responses continue to expose their existing schemas.

## Failure behavior

- Index or retrieval unavailable: try the configured provider fallback and then return the existing unreachable response.
- Retrieval succeeds with no candidates: return the existing no-results response.
- Reranker unavailable or timed out: preserve pre-rerank ordering.
- Inference unavailable with evidence present: use the existing deterministic search rendering when that path supports it; otherwise return the current explicit error.
- One stage never silently invokes a different stage or changes the surfaced intent.

## File direction

- Add focused internal pipeline modules under `src/internal/search_pipeline/` for context building, stage interfaces, ranking, and composition.
- Reuse `src/context/retrieval/client.py`, `src/internal/retrieval/service.py`, `src/internal/retrieval/fusion.py`, `src/model/serving.py`, and existing agent loops through adapters.
- Reduce duplicated orchestration in `src/internal/servers/web/app.py` without changing its routes or Pydantic API models.
- Keep existing retrieval and reranking server modules and CLIs operational.

## Testing

Tests cover session-context construction, stage boundaries, ranking order and fallback, grounded empty-evidence behavior, persistence metadata, streaming parity, existing endpoint compatibility, and a full in-process query pipeline with fakes. Existing retrieval, reranking, routing, and web endpoint suites remain regression gates.

## Non-goals

- New public or versioned APIs.
- Replacing indexing/background-job architecture.
- Changing retrieval or reranker algorithms.
- Removing existing server processes or endpoints.
- Training models during an API request.
