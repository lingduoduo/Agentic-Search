# Generated Context Pack

# Grounded Rag Safety

## Sources

- [Specification: 2026-07-15-grounded-rag-safety-design.md](../archive/specs/2026-07-15-grounded-rag-safety-design.md)
- [Plan: 2026-07-15-grounded-rag-safety.md](../archive/plans/2026-07-15-grounded-rag-safety.md)

## Specification Context

### Goal

Reduce hallucinations across the shared `src/context` RAG pipeline by constraining answers to verified retrieval and read-only tool evidence, enforcing a structured internal draft, retrying one failed verification, and abstaining when supported claims cannot answer the question.

### Scope

The safety layer applies to `generate_answer` and `answer_with_retrieval`, so MCP and other callers using the shared context pipeline receive the same behavior. Existing public answer and citation fields remain backward-compatible.

This phase supports registered read-only tools for dynamic facts. Side-effecting tools and tools without an explicit read-only classification are not eligible.

### Architecture

The pipeline becomes:

1. Retrieve documents using the existing search path.
2. Optionally invoke eligible read-only tools when the question requires dynamic or system-specific facts.
3. Normalize retrieval and tool results into a unified evidence bundle with stable IDs and provenance.
4. Ask the LLM for a strict internal `AnswerDraft` rather than unconstrained prose.
5. Deterministically verify every claim against its cited evidence.
6. If verification fails, retry once with the exact verifier findings and the original evidence.
7. Remove any claims that remain unsupported after retry.

…

### Verification and Retry

`VerificationResult` records:

- Supported claims.
- Unsupported claims with reasons.
- Dangling or unknown evidence IDs.
- Deterministic confidence score.
- Whether a retry occurred.
- Final status: `verified`, `partial`, or `abstained`.

Verification checks both identifier validity and lexical support against every cited evidence item. It reuses the existing grounding tokenizer and overlap semantics where possible, while operating claim-by-claim instead of only sentence-by-citation.

On first failure, the pipeline makes exactly one corrective LLM call. The retry prompt includes:

- The original question.
- The unchanged evidence bundle.
- The original structured draft.

…

## Implementation Plan Context

### Task 1: Add evidence, structured-draft, and verification primitives

**Files:**

- Modify: `src/context/models.py`
- Create: `src/context/safety.py`
- Modify: `src/context/__init__.py`
- Create: `tests/unit/test_rag_safety.py`

1. Write failing tests for evidence normalization, strict draft parsing, unknown evidence IDs, supported/unsupported claim verdicts, supported-only rendering, canonical abstention, and deterministic confidence.
2. Add `EvidenceSource`, `AnswerClaim`, `AnswerDraft`, `ClaimVerdict`, `VerificationResult`, and `VerificationStatus` models. Use stable `D*` IDs for retrieved documents and `T*` IDs for tool evidence.

…

### Task 2: Add the approved read-only tool evidence gateway

**Files:**

- Create: `src/context/tool_evidence.py`
- Modify: `src/context/__init__.py`
- Create: `tests/unit/test_rag_tool_evidence.py`

1. Write failing tests proving that read-only tools can supply evidence, side-effecting and unspecified tools are rejected, calls are bounded, timeouts/errors degrade safely, and serialization/IDs are deterministic.
2. Add `ToolSafety` (`READ_ONLY`, `SIDE_EFFECTING`, `UNSPECIFIED`), tool descriptors, requests, registry/selector protocols, and an async `collect_tool_evidence` gateway.
3. Default to `max_calls=2` and `timeout_seconds=5.0`; stringify tool results as stable JSON and emit only normalized `T*` evidence.

…

### Task 3: Guard generation with structured output, retry, and abstention

**Files:**

- Modify: `src/context/models.py`
- Modify: `src/context/prompts.py`
- Modify: `src/context/pipeline.py`
- Modify: `src/context/__init__.py`
- Modify/Create: focused pipeline tests under `tests/unit/`

1. Add sequence-based fake-LLM tests for a valid supported draft, one corrective retry, partial support, malformed output, total abstention, and exactly two maximum LLM calls. Add a compatibility test for disabling the guard.
2. Add `GroundedGenerationConfig(enabled=True, max_retries=1, overlap_threshold=0.15)`. Extend requests with optional evidence/safety/sufficiency inputs and results with defaulted `confidence`, `verification_status`, `abstained`, and `tool_evidence` metadata.

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
