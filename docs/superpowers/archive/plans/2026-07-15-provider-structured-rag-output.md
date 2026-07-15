# Provider-Enforced Structured RAG Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce the internal grounded-RAG `AnswerDraft` with provider-native JSON Schema when supported, while preserving prompt-only compatibility and all local semantic verification.

**Architecture:** A canonical schema and typed completion metadata live at the RAG boundary. Concrete providers declare capability and may apply schema constraints; guarded generation consumes the richer result, performs one controlled schema downgrade only for explicit unsupported-schema errors, and still runs the existing parser and evidence verifier.

**Tech Stack:** Python 3.12, dataclasses/enums/mapping proxies, requests, pytest, Ruff, OpenAI Chat Completions-compatible HTTP.

## Global Constraints

- Provider schema enforcement never replaces `parse_answer_draft` or `verify_answer_draft`.
- Generic OpenAI-compatible endpoints default to `PROMPT_ONLY`; only explicitly schema-capable configuration uses `JSON_SCHEMA`.
- Only explicit unsupported-`response_format`/`json_schema` failures may downgrade to prompt-only.
- Authentication, rate-limit, timeout, transport, and server failures must propagate without downgrade.
- A schema downgrade does not consume or increase the single semantic corrective retry.
- Refusal text, prompts, schemas, evidence bodies, model output, tool arguments, and exception bodies must not appear in new public responses or traces.
- Existing RAG/MCP response fields and `GroundedGenerationConfig(enabled=False)` behavior remain unchanged.
- No new external provider dependency, fact checker, or dynamic endpoint probing is introduced.

---

### Task 1: Canonical Schema and Structured Completion Contract

**Files:**
- Modify: `src/context/models.py`
- Create: `src/context/structured_output.py`
- Modify: `src/context/__init__.py`
- Create: `tests/unit/test_rag_structured_output.py`

**Interfaces:**
- Produces: `StructuredOutputCapability`, `StructuredOutputRequest`, `StructuredCompletionMetadata`, `SchemaUnsupportedError`, `answer_draft_json_schema()`.
- Preserves: `LLMClient.complete(messages, **kwargs) -> LLMResponse | str` and existing fake clients.

- [ ] **Step 1: Write failing schema and contract tests**

```python
def test_answer_draft_schema_is_strict_and_copy_safe():
    first = answer_draft_json_schema()
    assert first["additionalProperties"] is False
    assert first["required"] == ["claims", "missing_information", "abstain"]
    assert first["properties"]["claims"]["items"]["additionalProperties"] is False
    first["properties"]["abstain"]["type"] = "string"
    assert answer_draft_json_schema()["properties"]["abstain"]["type"] == "boolean"


def test_structured_metadata_defaults_preserve_legacy_clients():
    response = LLMResponse(text='{"claims": [], "missing_information": [], "abstain": true}')
    assert response.structured.applied is False
    assert response.structured.refused is False
    assert response.structured.incomplete_reason is None
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/unit/test_rag_structured_output.py`

Expected: collection/import failures because the structured-output types and schema function do not exist.

- [ ] **Step 3: Implement the canonical schema and additive models**

```python
class StructuredOutputCapability(str, Enum):
    JSON_SCHEMA = "json_schema"
    PROMPT_ONLY = "prompt_only"


@dataclass(frozen=True)
class StructuredOutputRequest:
    name: str
    schema: dict[str, object]
    strict: bool = True


@dataclass(frozen=True)
class StructuredCompletionMetadata:
    requested: bool = False
    applied: bool = False
    downgraded: bool = False
    refused: bool = False
    incomplete_reason: str | None = None


class SchemaUnsupportedError(RuntimeError):
    """The provider explicitly rejected JSON Schema output enforcement."""
```

Add `structured: StructuredCompletionMetadata = field(default_factory=StructuredCompletionMetadata)` to `LLMResponse`. Implement `answer_draft_json_schema()` with `copy.deepcopy` over a private constant matching the approved spec exactly. Export the public types/functions.

- [ ] **Step 4: Run GREEN and adjacent models tests**

Run: `pytest -q tests/unit/test_rag_structured_output.py tests/unit/test_context_pipeline.py tests/unit/test_rag_safety.py`

Expected: all tests pass with no warnings.

- [ ] **Step 5: Commit**

```bash
git add src/context/models.py src/context/structured_output.py src/context/__init__.py tests/unit/test_rag_structured_output.py
git commit -m "feat: add structured RAG completion contract"
```

---

### Task 2: OpenAI-Compatible Schema Enforcement and Capability

**Files:**
- Modify: `src/internal/llm/interfaces.py`
- Modify: `src/internal/llm/providers.py`
- Modify: `src/internal/llm/multi_llm.py`
- Modify: `tests/unit/servers/web/test_stage_emits_llm.py`
- Create: `tests/unit/test_llm_structured_output.py`

**Interfaces:**
- Consumes: `StructuredOutputCapability`, `StructuredOutputRequest`, `StructuredCompletionMetadata`, `SchemaUnsupportedError`.
- Produces: `OpenAICompatibleLLM.structured_output_capability` and `complete(..., structured_output=...)` behavior.

- [ ] **Step 1: Write failing provider request/response tests**

```python
def test_native_openai_complete_sends_strict_json_schema(fake_session, schema_request):
    llm = configured_llm(provider="openai")
    llm.complete(MESSAGES, structured_output=schema_request)
    body = fake_session.post.call_args.kwargs["json"]
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": schema_request.name,
            "strict": True,
            "schema": schema_request.schema,
        },
    }


def test_generic_compatible_endpoint_is_prompt_only():
    llm = configured_llm(provider="openai-compatible")
    assert llm.structured_output_capability is StructuredOutputCapability.PROMPT_ONLY
```

Also cover refusal, `finish_reason == "length"`, explicit unsupported-schema HTTP 400 classification, unrelated HTTP 400 propagation, HTTP 429 propagation, HTTP 500 propagation, timeout propagation, and ensuring capture metadata contains only structured booleans/categories—not schema, messages, response text, refusal text, or exception bodies.

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/unit/test_llm_structured_output.py tests/unit/servers/web/test_stage_emits_llm.py`

Expected: failures because capability, request forwarding, response metadata, and error classification are absent.

- [ ] **Step 3: Implement capability and request forwarding**

Add a provider capability property to the interface with a safe `PROMPT_ONLY` default. Native OpenAI configuration returns `JSON_SCHEMA`; generic compatible configuration remains `PROMPT_ONLY` unless `custom_config["supports_json_schema"]` is exactly the case-insensitive string `"true"`.

In `complete`, when capability is `JSON_SCHEMA` and `structured_output` is present, add:

```python
body["response_format"] = {
    "type": "json_schema",
    "json_schema": {
        "name": structured_output.name,
        "strict": structured_output.strict,
        "schema": structured_output.schema,
    },
}
```

Return `LLMResponse` when structured output was requested so refusal and incomplete state are retained. Preserve the string return for legacy calls without structured output.

Classify a 400 as `SchemaUnsupportedError` only when the structured request was present and the sanitized provider error identifies `response_format` or `json_schema` as unsupported/unknown. Re-raise every other error unchanged. Never include the raw response body in the new exception message or trace metadata.

- [ ] **Step 4: Keep LiteLLM forwarding compatible**

Retain its existing `response_format` forwarding and map the new request wrapper to the existing parameter shape at the adapter boundary. Do not add provider-specific Anthropic or Gemini request construction.

- [ ] **Step 5: Run GREEN and provider regressions**

Run: `pytest -q tests/unit/test_llm_structured_output.py tests/unit/servers/web/test_stage_emits_llm.py tests/unit/test_llm_providers.py`

Expected: all tests pass; unrelated HTTP failures remain exceptions.

- [ ] **Step 6: Commit**

```bash
git add src/internal/llm/interfaces.py src/internal/llm/providers.py src/internal/llm/multi_llm.py tests/unit/test_llm_structured_output.py tests/unit/servers/web/test_stage_emits_llm.py
git commit -m "feat: enforce provider structured output"
```

---

### Task 3: Guarded RAG Integration, Controlled Downgrade, and Observability

**Files:**
- Modify: `src/context/pipeline.py`
- Modify: `src/context/models.py`
- Modify: `src/internal/mcp_server/tools/chat.py`
- Modify: `tests/unit/test_grounded_generation.py`
- Modify: `tests/unit/test_rag_pipeline_integration.py`
- Modify: `tests/unit/test_mcp_server.py`
- Modify: `tests/unit/observability/test_tracer.py`

**Interfaces:**
- Consumes: `answer_draft_json_schema()`, structured capability/request/result metadata, `SchemaUnsupportedError`.
- Produces: guarded generation that requests schemas when supported and records safe downgrade/refusal/incomplete aggregate metadata.

- [ ] **Step 1: Write failing guarded-generation tests**

Add sequence clients that prove:

```python
def test_schema_unsupported_downgrades_without_consuming_semantic_retry():
    llm = SequenceLLM([
        SchemaUnsupportedError("structured output unsupported"),
        MALFORMED_JSON,
        SUPPORTED_DRAFT,
    ], capability=StructuredOutputCapability.JSON_SCHEMA)
    result = generate_answer(REQUEST, llm=llm)
    assert len(llm.calls) == 3
    assert llm.calls[0].structured_output is not None
    assert llm.calls[1].structured_output is None
    assert llm.calls[2].structured_output is None
    assert result.retry_count == 1
    assert result.structured_output_downgraded is True
```

Add separate tests with explicit assertions for native schema success, prompt-only client compatibility, refusal-to-abstention, incomplete-then-corrective-success, no downgrade on ordinary provider errors, and legacy disabled mode never requesting a schema:

```python
assert native_llm.calls[0].structured_output.name == "answer_draft"
assert prompt_only_llm.calls[0].structured_output is None
assert refusal_result.answer == CANONICAL_ABSTENTION
assert refusal_result.abstained is True
assert incomplete_result.retry_count == 1
with pytest.raises(requests.Timeout):
    generate_answer(REQUEST, llm=timeout_llm)
assert legacy_llm.calls[0].structured_output is None
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/unit/test_grounded_generation.py tests/unit/test_rag_pipeline_integration.py`

Expected: new assertions fail because guarded generation does not pass a structured request or interpret structured metadata.

- [ ] **Step 3: Implement bounded schema-aware attempts**

Create one `StructuredOutputRequest(name="answer_draft", schema=answer_draft_json_schema())` per guarded generation. For each semantic attempt:

```python
try:
    raw = llm.complete(messages, structured_output=request_or_none)
except SchemaUnsupportedError:
    if request_or_none is None:
        raise
    downgraded = True
    request_or_none = None
    raw = llm.complete(messages)
```

Once downgraded, all later corrective calls remain prompt-only. Refusal produces canonical abstention immediately. Incomplete output becomes verifier feedback and may use the one semantic retry. Local parse and evidence verification remain unchanged for every text response.

Add defaulted result metadata such as `structured_output_applied: bool = False`, `structured_output_downgraded: bool = False`, and safe refusal/incomplete categories. Do not add raw provider content.

- [ ] **Step 4: Add safe caller/trace metadata**

Preserve existing MCP keys and add only aggregate structured-output booleans/categories. Update tracing tests to assert forbidden prompt/schema/evidence/output/refusal strings are absent.

- [ ] **Step 5: Run GREEN and integration regressions**

Run: `pytest -q tests/unit/test_grounded_generation.py tests/unit/test_rag_pipeline_integration.py tests/unit/test_mcp_server.py tests/unit/test_agentic_rag.py tests/unit/observability/test_tracer.py`

Expected: all tests pass; semantic retries never exceed one.

- [ ] **Step 6: Commit**

```bash
git add src/context/pipeline.py src/context/models.py src/internal/mcp_server/tools/chat.py tests/unit/test_grounded_generation.py tests/unit/test_rag_pipeline_integration.py tests/unit/test_mcp_server.py tests/unit/observability/test_tracer.py
git commit -m "feat: integrate schema-constrained RAG output"
```

---

### Task 4: Documentation, Archive, Context Packs, and Final Verification

**Files:**
- Modify: `docs/retrieval.md`
- Move: `docs/superpowers/specs/2026-07-15-provider-structured-rag-output-design.md` to `docs/superpowers/archive/specs/`
- Move: `docs/superpowers/plans/2026-07-15-provider-structured-rag-output.md` to `docs/superpowers/archive/plans/`
- Regenerate: `docs/superpowers/context-packs/INDEX.md`
- Create: `docs/superpowers/context-packs/provider-structured-rag-output-context-pack.md`

**Interfaces:**
- Documents the implemented capability/downgrade contract and keeps generated planning context current.

- [ ] **Step 1: Document operator behavior**

Document native schema enforcement, generic compatible prompt-only default, explicit configuration, the narrow unsupported-schema downgrade, refusal/incomplete handling, local semantic verification, and the distinction between structural and factual correctness.

- [ ] **Step 2: Archive approved sources and regenerate**

Run:

```bash
git mv docs/superpowers/specs/2026-07-15-provider-structured-rag-output-design.md docs/superpowers/archive/specs/
git mv docs/superpowers/plans/2026-07-15-provider-structured-rag-output.md docs/superpowers/archive/plans/
python scripts/generate_context_packs.py
python scripts/generate_context_packs.py --check
```

Expected: active spec/plan directories contain only `.gitkeep`; generator reports the new spec/plan counts with no duplicates, drift, placeholders, or broken links.

- [ ] **Step 3: Run final verification**

```bash
pytest -q tests/unit/test_rag_structured_output.py tests/unit/test_llm_structured_output.py tests/unit/test_grounded_generation.py tests/unit/test_rag_pipeline_integration.py tests/unit/test_mcp_server.py tests/unit/test_agentic_rag.py tests/unit/observability/test_tracer.py
env HF_HOME=/tmp/agentic-search-hf-cache pytest -q
ruff check src/context src/internal/llm src/internal/mcp_server/tools/chat.py tests/unit
ruff format --check src/context src/internal/llm src/internal/mcp_server/tools/chat.py tests/unit
python scripts/generate_context_packs.py --check
git diff --check
```

Expected: focused and full suites pass; Ruff, formatting, generated artifacts, and whitespace are clean.

- [ ] **Step 4: Commit, push, and update draft PR #415**

```bash
git add docs/retrieval.md docs/superpowers/archive docs/superpowers/context-packs
git commit -m "docs: document structured RAG output"
git push
gh pr edit 415 --body-file /tmp/provider-structured-rag-pr-body.md
```

Update the PR body with provider-enforced schema behavior, fallback boundaries, verification evidence, and remaining provider limitations. Keep the PR in draft state.

## Completion Criteria

- Schema-capable native OpenAI requests carry the canonical strict JSON Schema.
- Generic compatible endpoints remain prompt-only unless explicitly enabled.
- Only explicit schema-unsupported responses downgrade; unrelated provider failures propagate.
- Refusal and incomplete states are retained and handled without exposing raw content.
- Schema downgrade does not increase the single semantic corrective retry.
- Local parsing and evidence verification run for every successful draft.
- Legacy clients and disabled guarded generation remain compatible.
- Documentation, archived sources, context pack, tests, and draft PR #415 are current.
