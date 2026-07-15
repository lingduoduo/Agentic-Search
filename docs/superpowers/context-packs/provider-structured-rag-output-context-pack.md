# Generated Context Pack

# Provider Structured Rag Output

## Sources

- [Specification: 2026-07-15-provider-structured-rag-output-design.md](../archive/specs/2026-07-15-provider-structured-rag-output-design.md)
- [Plan: 2026-07-15-provider-structured-rag-output.md](../archive/plans/2026-07-15-provider-structured-rag-output.md)

## Specification Context

### Goal

Strengthen grounded RAG generation by using provider-enforced JSON Schema when the configured provider and model support it, while retaining the existing prompt, parser, semantic evidence verifier, one-retry limit, and canonical abstention behavior.

## Implementation Plan Context

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

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/unit/test_rag_structured_output.py`

…

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

…

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

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
