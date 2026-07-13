# Simplified Model Services Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide one local API process for retrieval, ranking, and inference while keeping identical capability routers independently deployable.

**Architecture:** Canonical Pydantic contracts and capability protocols sit behind three router factories. A profile-driven app mounts all routers or exactly one; existing servers become compatibility launchers and keep legacy payloads operational.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic v2, asyncio, pytest, FastAPI `TestClient`.

## Global Constraints

- Canonical endpoints are `/v1/retrieve`, `/v1/rank`, `/v1/generate`, and `/health`.
- Legacy `/retrieve`, `/search`, `/rerank`, and `/v1/completions` routes remain available with unchanged legacy response bodies.
- Heavy model dependencies load lazily and one failed capability must not stop another capability.
- Unified and single-capability profiles use the same router factories and contracts.
- No production model download is required by unit tests.
- This work does not replace algorithms, combine model weights, or add an automatic retrieve-rank-generate orchestration endpoint.

---

### Task 1: Canonical contracts and capability protocols

**Files:**
- Create: `src/internal/model_services/__init__.py`
- Create: `src/internal/model_services/contracts.py`
- Create: `src/internal/model_services/interfaces.py`
- Test: `tests/unit/model_services/test_contracts.py`

**Interfaces:**
- Produces: `Document`, `RetrieveRequest`, `RetrieveResponse`, `RankRequest`, `RankResponse`, `Message`, `GenerateRequest`, `GenerateResponse`, `Usage`, `CapabilityStatus`, `HealthResponse`, `ErrorDetail`, `ErrorResponse`; `Retriever`, `Ranker`, and `Generator` protocols.

- [ ] **Step 1: Write failing serialization and validation tests**

Test that `Document` defaults `score=0.0` and `metadata={}`, `GenerateRequest` accepts exactly one of `prompt`/`messages`, `RankRequest` accepts canonical documents, and protocol-compatible fakes expose async `retrieve`, `rank`, and `generate` methods.

```python
def test_generate_request_requires_exactly_one_input():
    with pytest.raises(ValidationError):
        GenerateRequest()
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="x", messages=[Message(role="user", content="x")])
```

- [ ] **Step 2: Run the tests and verify the package is missing**

Run: `pytest tests/unit/model_services/test_contracts.py -q`
Expected: collection fails with `ModuleNotFoundError: src.internal.model_services`.

- [ ] **Step 3: Implement contracts and protocols**

Use `model_validator(mode="after")` for the prompt/messages invariant. Protocol signatures are:

```python
class Retriever(Protocol):
    async def retrieve(self, request: RetrieveRequest) -> RetrieveResponse: ...

class Ranker(Protocol):
    async def rank(self, request: RankRequest) -> RankResponse: ...

class Generator(Protocol):
    async def generate(self, request: GenerateRequest) -> GenerateResponse: ...
```

- [ ] **Step 4: Run tests and commit**

Run: `pytest tests/unit/model_services/test_contracts.py -q`
Expected: all tests pass.

```bash
git add src/internal/model_services tests/unit/model_services/test_contracts.py
git commit -m "feat: add canonical model service contracts"
```

### Task 2: Lazy runtime and shared errors

**Files:**
- Create: `src/internal/model_services/runtime.py`
- Test: `tests/unit/model_services/test_runtime.py`

**Interfaces:**
- Consumes: canonical capability protocols.
- Produces: `CapabilityRuntime[T]`, `CapabilityUnavailable`, `install_error_handlers(app)`, and status conversion for `/health`.

- [ ] **Step 1: Write failing lifecycle tests**

Cover one-time concurrent initialization, disabled capability, failed factory isolation, `aclose()` forwarding, and stable error JSON.

```python
runtime = CapabilityRuntime("ranking", factory, enabled=True)
first, second = await asyncio.gather(runtime.get(), runtime.get())
assert first is second
assert calls == 1
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/unit/model_services/test_runtime.py -q`
Expected: import fails because `runtime.py` does not exist.

- [ ] **Step 3: Implement the runtime**

Guard initialization with `asyncio.Lock`; states are `disabled`, `initializing`, `ready`, and `unavailable`. Map `CapabilityUnavailable` to 503 and add typed helpers for 502 and 504 without exposing exception text.

- [ ] **Step 4: Run tests and commit**

Run: `pytest tests/unit/model_services/test_runtime.py -q`
Expected: all tests pass.

```bash
git add src/internal/model_services/runtime.py tests/unit/model_services/test_runtime.py
git commit -m "feat: add isolated capability runtime"
```

### Task 3: Retrieval and ranking routers with legacy adapters

**Files:**
- Create: `src/internal/model_services/retrieval.py`
- Create: `src/internal/model_services/ranking.py`
- Modify: `src/internal/servers/retrieval/server.py`
- Modify: `src/internal/servers/retrieval/rerank.py`
- Test: `tests/unit/model_services/test_retrieval_api.py`
- Test: `tests/unit/model_services/test_ranking_api.py`
- Test: `tests/unit/servers/retrieval/test_new_server.py`
- Test: `tests/unit/test_rerank.py`

**Interfaces:**
- Consumes: `CapabilityRuntime[Retriever]`, `CapabilityRuntime[Ranker]`, `RetrievalService.search`, and `SentenceTransformerReranker.rerank`.
- Produces: `create_retrieval_router(runtime, include_legacy=True)`, `create_ranking_router(runtime, include_legacy=True)`, `RetrievalServiceAdapter`, and `SentenceTransformerRankerAdapter`.

- [ ] **Step 1: Write failing canonical endpoint tests**

Use async fakes and assert `/v1/retrieve` returns canonical documents and `/v1/rank` returns reordered canonical documents. Assert legacy routes retain their current keys and include `Deprecation: true` plus `Link: </v1/...>; rel="successor-version"`.

- [ ] **Step 2: Verify failure**

Run: `pytest tests/unit/model_services/test_retrieval_api.py tests/unit/model_services/test_ranking_api.py -q`
Expected: router imports fail.

- [ ] **Step 3: Implement adapters and routers**

Run synchronous model work through `asyncio.to_thread`. Conversion maps `RetrievalResult.text` to `Document.content`, preserves metadata, and replaces only the canonical score with the reranker score.

- [ ] **Step 4: Convert old app factories into thin composition wrappers**

`server.create_app(service)` constructs `RetrievalServiceAdapter(service)` and mounts the retrieval router plus existing eval/admin routers. `rerank.create_app(config)` constructs a lazy ranker runtime and mounts the ranking router. Preserve CLI arguments and public factory names.

- [ ] **Step 5: Run canonical and compatibility tests**

Run: `pytest tests/unit/model_services/test_retrieval_api.py tests/unit/model_services/test_ranking_api.py tests/unit/servers/retrieval/test_new_server.py tests/unit/test_rerank.py -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/internal/model_services src/internal/servers/retrieval/server.py src/internal/servers/retrieval/rerank.py tests/unit/model_services tests/unit/servers/retrieval/test_new_server.py tests/unit/test_rerank.py
git commit -m "feat: unify retrieval and ranking APIs"
```

### Task 4: Inference router and ServerManager adapter

**Files:**
- Create: `src/internal/model_services/inference.py`
- Test: `tests/unit/model_services/test_inference_api.py`
- Test: `tests/unit/test_model_serving.py`

**Interfaces:**
- Consumes: canonical generation contracts and serving-side `ServerManager.generate(request_id, prompt_ids, sampling_params)`.
- Produces: `ServerManagerGenerator`, `create_inference_router(runtime, include_legacy=True)`, canonical `/v1/generate`, and OpenAI-compatible `/v1/completions`.

- [ ] **Step 1: Write failing canonical and OpenAI compatibility tests**

Use a fake tokenizer and manager. Assert `/v1/generate` accepts prompt input and `/v1/completions` retains `choices[0].text`, `finish_reason`, `model`, and usage fields.

- [ ] **Step 2: Verify failure**

Run: `pytest tests/unit/model_services/test_inference_api.py -q`
Expected: inference router import fails.

- [ ] **Step 3: Implement the adapter and router**

Encode prompt text, call `ServerManager.generate`, decode output IDs, and return canonical usage. For messages, render role/content lines through one deterministic helper. Keep sampling parameter names aligned with `OpenAIServerManager`.

- [ ] **Step 4: Run tests and commit**

Run: `pytest tests/unit/model_services/test_inference_api.py tests/unit/test_model_serving.py -q`
Expected: all tests pass.

```bash
git add src/internal/model_services/inference.py tests/unit/model_services/test_inference_api.py tests/unit/test_model_serving.py
git commit -m "feat: add canonical inference API"
```

### Task 5: Unified and independent deployment profiles

**Files:**
- Create: `src/internal/model_services/app.py`
- Create: `src/internal/model_services/__main__.py`
- Test: `tests/unit/model_services/test_app.py`
- Test: `tests/integration/test_model_services_flow.py`
- Modify: `.env.example`

**Interfaces:**
- Consumes: all router factories and capability runtimes.
- Produces: `ModelServicesSettings.from_env()`, `create_app(settings, factories=None)`, CLI module, shared `/health`, and four profiles.

- [ ] **Step 1: Write failing profile and health tests**

Assert unified mounts all canonical endpoints; each single profile mounts only its capability; disabled endpoints return 404; health lists configured capabilities and preserves readiness isolation.

- [ ] **Step 2: Write the in-process flow test**

Inject fake retriever/ranker/generator factories, call retrieve → rank → generate, and assert canonical `Document` objects cross boundaries without legacy conversion.

- [ ] **Step 3: Verify failure**

Run: `pytest tests/unit/model_services/test_app.py tests/integration/test_model_services_flow.py -q`
Expected: app module import fails.

- [ ] **Step 4: Implement settings, composition, lifecycle, and CLI**

Parse `MODEL_SERVICES_PROFILE` and comma-separated `MODEL_SERVICES_CAPABILITIES`; reject unknown values at startup. Use FastAPI lifespan to close initialized runtimes. CLI uses existing `add_host_port_args` and `run_uvicorn_app` helpers.

- [ ] **Step 5: Document environment defaults and run tests**

Add commented unified-profile settings to `.env.example`.

Run: `pytest tests/unit/model_services tests/integration/test_model_services_flow.py -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/internal/model_services tests/unit/model_services tests/integration/test_model_services_flow.py .env.example
git commit -m "feat: add composable model service profiles"
```

### Task 6: Canonical clients, docs, and full regression verification

**Files:**
- Create: `src/internal/model_services/client.py`
- Modify: `src/context/retrieval/client.py`
- Modify: `src/internal/servers/web/app.py`
- Modify: `README.md`
- Modify: `docs/api-reference.md`
- Modify: `docs/architecture.md`
- Modify: `docs/configuration.md`
- Modify: `docs/retrieval.md`
- Modify: `docs/training-and-evaluation.md`
- Test: `tests/unit/model_services/test_client.py`
- Test: `tests/unit/test_execution_fallbacks.py`

**Interfaces:**
- Consumes: canonical endpoints and contracts.
- Produces: async `ModelServicesClient` with `retrieve`, `rank`, and `generate`; compatibility wrapper for `SearchClient`; unified local-development documentation.

- [ ] **Step 1: Write failing client tests**

Mock HTTP responses and assert canonical serialization, timeout handling, and normalized documents. Verify `SearchClient.retrieve_one` still returns `SearchResult` through its adapter.

- [ ] **Step 2: Implement the canonical client and migrate internal callers**

Centralize session ownership, retries, and error decoding in `ModelServicesClient`. Keep `SearchClient` as a compatibility facade; update new web service configuration to use canonical URLs without changing public `/api/agent` payloads.

- [ ] **Step 3: Update maintained documentation**

Document one local command:

```bash
MODEL_SERVICES_PROFILE=unified python -m src.internal.model_services
```

Include canonical and legacy endpoint tables, independent production profiles, readiness semantics, and the fact that inference serving never trains model weights.

- [ ] **Step 4: Run focused suites**

Run: `pytest tests/unit/model_services tests/unit/servers/retrieval tests/unit/test_rerank.py tests/unit/test_model_serving.py tests/unit/test_execution_fallbacks.py -q`
Expected: all tests pass.

- [ ] **Step 5: Run static and documentation checks**

Run: `ruff check src/internal/model_services src/internal/servers/retrieval src/context/retrieval/client.py`

Run: `ruff format --check src/internal/model_services tests/unit/model_services`

Run: `git diff --check`

Expected: all commands exit zero.

- [ ] **Step 6: Commit**

```bash
git add src/internal/model_services/client.py src/context/retrieval/client.py src/internal/servers/web/app.py README.md docs tests/unit/model_services/test_client.py tests/unit/test_execution_fallbacks.py
git commit -m "docs: document unified model services"
```
