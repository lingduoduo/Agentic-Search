# Simplified Model Services Design

## Goal

Provide one simple local API process for retrieval, ranking, and inference while preserving the ability to deploy and scale each capability independently in production.

## Current problems

The repository currently exposes related model-serving behavior through several unrelated entry points and payload shapes:

- retrieval uses `/retrieve` in the demo/hybrid servers and `/search` in `RetrievalService`;
- ranking uses a separate `/rerank` application and nested batch-oriented document shapes;
- inference is primarily an in-process `ServerManager` protocol or an external OpenAI-compatible `/v1/completions` server;
- health checks, lifecycle management, configuration, error responses, and HTTP clients differ by service;
- local development requires reasoning about multiple processes even when independent scaling is unnecessary.

## Chosen approach

Build composable capability modules around one canonical contract set. Each capability owns a protocol, router factory, runtime adapter, and legacy endpoint adapter. A unified application mounts all enabled routers for local development; single-capability applications mount the same routers for independent production deployment.

This avoids a proxy-only facade that would preserve all underlying duplication, and avoids a monolith whose components would need to be extracted again for scale-out.

## Architecture

```text
Shared contracts
  RetrieveRequest / RetrieveResponse / Document
  RankRequest / RankResponse
  GenerateRequest / GenerateResponse / Usage
  HealthResponse / CapabilityStatus / ErrorResponse

Capability protocols
  Retriever.retrieve(...)
  Ranker.rank(...)
  Generator.generate(...)

Router factories
  create_retrieval_router(retriever)
  create_ranking_router(ranker)
  create_inference_router(generator)

Deployment profiles
  unified app:     /v1/retrieve  /v1/rank  /v1/generate  /health
  retrieval app:  /v1/retrieve                              /health
  ranking app:                  /v1/rank                     /health
  inference app:                         /v1/generate        /health
```

The same canonical models and router factories are used in every profile. Deployment topology changes configuration and process boundaries, not API behavior.

## Canonical API

### Retrieval

`POST /v1/retrieve`

Request fields:

- `query: str`
- `top_k: int = 5`
- `filters: dict | None`

Response fields:

- `documents: list[Document]`
- `mode: str`
- `executed_queries: list[str]`
- `latency_ms: float`

### Ranking

`POST /v1/rank`

Request fields:

- `query: str`
- `documents: list[Document]`
- `top_k: int | None`

Response fields:

- `documents: list[Document]`, reordered with ranking scores
- `model: str | None`
- `latency_ms: float`

### Inference

`POST /v1/generate`

Request fields:

- exactly one of `prompt: str` or `messages: list[Message]`
- `model: str | None`
- `max_tokens`, `temperature`, `top_p`, and `stop`

Response fields:

- `text: str`
- `model: str`
- `finish_reason: str`
- `usage: Usage`
- `latency_ms: float`

### Health

`GET /health` returns process status plus one readiness record per configured capability. A capability can be disabled, initializing, ready, or unavailable without preventing unrelated capabilities from serving.

## Shared document model

All canonical endpoints use one document shape:

- `id: str`
- `title: str`
- `content: str`
- `url: str | None`
- `score: float`
- `metadata: dict[str, object]`

Adapters are responsible for converting legacy `text`, `contents`, nested `document`, and `doc_id` fields at the boundary.

## Compatibility

Legacy entry points remain operational during migration:

- `/retrieve` and `/search` adapt into the canonical retrieval runtime;
- `/rerank` adapts into the canonical ranking runtime;
- `/v1/completions` adapts into the canonical inference runtime;
- existing legacy response bodies remain unchanged on legacy routes;
- legacy responses include deprecation headers pointing to the canonical endpoint.

Internal clients migrate to canonical endpoints and models. Legacy routes are not removed by this project.

## Configuration

The deployment profile is selected with:

```dotenv
MODEL_SERVICES_PROFILE=unified
MODEL_SERVICES_CAPABILITIES=retrieval,ranking,inference
```

`MODEL_SERVICES_PROFILE` accepts `unified`, `retrieval`, `ranking`, or `inference`. The explicit capability list is primarily for unified deployments that intentionally omit heavy optional models.

Existing capability-specific settings continue to configure implementations, including retrieval backend/index settings, reranker model/device settings, and inference model/server settings. This project centralizes selection and lifecycle; it does not rename every backend-specific variable.

## Lifecycle and failure isolation

Capability runtimes initialize lazily on first use or explicit readiness warmup. Heavy optional imports such as PyTorch, sentence-transformers, or Transformers stay behind their capability boundary.

Failure rules:

- a ranking model load failure does not stop retrieval or inference;
- a missing inference model does not stop retrieval or ranking;
- readiness identifies the unavailable capability and a safe public reason;
- capability initialization is concurrency-safe and runs once;
- application shutdown closes initialized remote clients and model managers.

## Errors

Canonical endpoints use one error envelope:

```json
{
  "error": {
    "code": "capability_unavailable",
    "message": "Ranking is not configured",
    "capability": "ranking",
    "retryable": false
  }
}
```

Status mapping:

- `400` or `422`: invalid request;
- `502`: configured remote backend failed;
- `503`: capability disabled, unavailable, or still initializing;
- `504`: backend timeout.

Raw provider, model, filesystem, or stack-trace details are logged but not returned to clients.

## File boundaries

- `src/internal/model_services/contracts.py`: canonical Pydantic models.
- `src/internal/model_services/interfaces.py`: capability protocols.
- `src/internal/model_services/runtime.py`: lazy, concurrency-safe capability holders and shared errors.
- `src/internal/model_services/retrieval.py`: retrieval router, implementation adapter, and legacy adapters.
- `src/internal/model_services/ranking.py`: ranking router, implementation adapter, and legacy adapter.
- `src/internal/model_services/inference.py`: generation router, `ServerManager` adapter, and OpenAI compatibility adapter.
- `src/internal/model_services/app.py`: profile configuration, app composition, lifecycle, and health.
- Existing files under `src/internal/servers/retrieval/` and the inference entry point become thin compatibility launchers that import these factories.
- Canonical HTTP clients live beside the shared contracts so the web app and agent code do not duplicate payload conversion.

## Migration sequence

1. Add canonical contracts, protocols, runtime errors, and isolated router tests.
2. Adapt `RetrievalService` and preserve `/search` and `/retrieve` compatibility.
3. Adapt `SentenceTransformerReranker` and preserve `/rerank` compatibility.
4. Adapt the serving-side `ServerManager` and preserve `/v1/completions` compatibility.
5. Add unified and single-capability app profiles with shared health and lifecycle.
6. Move internal clients to canonical endpoints and normalized documents.
7. Simplify startup scripts and documentation around the unified local profile.

Each step leaves a runnable, backward-compatible system.

## Testing

Tests cover:

- request and response serialization for every canonical contract;
- router behavior with lightweight fake retriever, ranker, and generator implementations;
- unified and each single-capability deployment profile;
- lazy initialization, concurrency safety, readiness, shutdown, and failure isolation;
- stable error envelopes and HTTP status mapping;
- legacy endpoint request and response compatibility;
- canonical HTTP clients;
- an in-process retrieve → rank → generate smoke flow;
- regression suites for existing retrieval, reranking, web routing, and agent inference behavior.

No test requires downloading a production model; model-specific integration tests remain opt-in.

## Documentation and local operation

The README will recommend one unified local command. The API, architecture, configuration, retrieval, and training guides will distinguish unified local deployment from independent production profiles and list compatibility endpoints. Existing commands remain documented during migration.

## Non-goals

- Removing legacy endpoints in this change.
- Replacing retrieval, ranking, or inference algorithms.
- Combining model weights or forcing all capabilities onto one device.
- Building an orchestration pipeline that automatically retrieves, ranks, and generates in one request; the services remain composable primitives.
- Renaming every existing backend-specific environment variable.
