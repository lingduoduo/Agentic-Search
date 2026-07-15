# Grounded RAG Safety Design

## Goal

Reduce hallucinations across the shared `src/context` RAG pipeline by constraining answers to verified retrieval and read-only tool evidence, enforcing a structured internal draft, retrying one failed verification, and abstaining when supported claims cannot answer the question.

## Scope

The safety layer applies to `generate_answer` and `answer_with_retrieval`, so MCP and other callers using the shared context pipeline receive the same behavior. Existing public answer and citation fields remain backward-compatible.

This phase supports registered read-only tools for dynamic facts. Side-effecting tools and tools without an explicit read-only classification are not eligible.

## Architecture

The pipeline becomes:

1. Retrieve documents using the existing search path.
2. Optionally invoke eligible read-only tools when the question requires dynamic or system-specific facts.
3. Normalize retrieval and tool results into a unified evidence bundle with stable IDs and provenance.
4. Ask the LLM for a strict internal `AnswerDraft` rather than unconstrained prose.
5. Deterministically verify every claim against its cited evidence.
6. If verification fails, retry once with the exact verifier findings and the original evidence.
7. Remove any claims that remain unsupported after retry.
8. Render supported claims with citations, or return an explicit abstention when no supported answer remains.

Tool failure degrades to retrieval evidence. Verification failure fails closed.

## Components

### Evidence Model

`EvidenceSource` normalizes every usable source:

- Stable evidence ID.
- Text content.
- Human-readable title.
- Optional URL.
- Provenance type: `retrieval` or `tool`.
- Tool name for tool evidence.
- Original metadata needed by callers and tracing.

Retrieved documents retain their existing `[D1]`, `[D2]` identifiers. Tool evidence receives stable `[T1]`, `[T2]` identifiers in invocation order.

### Read-Only Tool Gateway

The shared RAG pipeline accepts an optional tool gateway rather than depending directly on `ToolAgentLoop`. The gateway exposes registered tools with an explicit read-only classification and returns normalized evidence.

Constraints:

- Only explicitly read-only tools are visible to the RAG selector.
- Unknown, side-effecting, and unspecified tools are rejected before invocation.
- Tool calls are bounded by configurable count and timeout.
- Tool errors and timeouts are recorded, then generation continues using available retrieval evidence.
- Tool outputs are treated as untrusted data, never as instructions.

The selector is deterministic when no model-based selector is configured. It may use explicit routing metadata or a caller-provided selection callback; the safety layer does not infer arbitrary tool calls from generated prose.

### Structured Draft

The LLM produces an internal `AnswerDraft`:

- `claims`: ordered `AnswerClaim` objects containing claim text and cited evidence IDs.
- `missing_information`: facts required but absent from the evidence.
- `abstain`: explicit boolean.

Each factual claim must cite at least one known evidence ID. Schema parsing rejects malformed payloads, extra top-level fields, empty claim text, and unknown evidence identifiers.

The structured draft is internal. Callers continue receiving rendered answer text and existing citation lists.

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
- Exact unsupported claim and citation findings.
- An instruction to remove unsupported material rather than invent new evidence.

If the retry remains partially unsupported, only supported claims are rendered. If no supported claim answers the question, the pipeline abstains.

### Abstention

The canonical abstention is:

> I don't know based on the available evidence.

The pipeline abstains when:

- No usable evidence exists.
- The draft explicitly abstains and provides no supported claims.
- Structured output cannot be parsed after the one allowed corrective attempt.
- All claims remain unsupported after retry.

Missing information may be appended in a concise second sentence when the structured draft identifies it without speculation.

### Confidence

Confidence is deterministic, not self-reported by the model. It combines:

- Fraction of claims that pass verification.
- Evidence coverage across the final claims.
- Retrieval/evidence sufficiency when available.

The value is clamped to `[0.0, 1.0]`. Abstained results have confidence `0.0`. A fully verified answer can still have less than `1.0` when evidence sufficiency is weak.

## Public Result Compatibility

`AnswerGenerationResult` preserves:

- `answer`
- `citations`
- `context`
- `prompt`
- `grounding_report`

It adds optional fields with backward-compatible defaults:

- `confidence`
- `verification`
- `abstained`
- `tool_evidence`

Existing callers that only consume answer and citations require no changes. MCP and web adapters may expose the new fields without removing or renaming current fields.

## Extractive Fallback

When no LLM is configured, the existing extractive synthesis remains available. It renders only text selected directly from evidence and returns the same confidence and verification metadata. It does not perform an LLM retry.

If extractive ranking finds no relevant evidence, it returns the canonical abstention instead of emitting the first unrelated document sentence.

## Observability

Tracing records:

- Retrieval evidence count.
- Tool evidence count and tool names.
- Tool failure/timeout counts.
- Draft parsing status.
- Unsupported claim count.
- Retry reason and whether retry occurred.
- Final verification status.
- Confidence and abstention.

Sensitive evidence content and raw tool output are not logged.

## Testing

Tests must prove:

- Retrieval-only answers preserve existing public fields.
- Dynamic questions invoke only eligible read-only tools.
- Side-effecting, unknown, and unspecified tools are rejected.
- Malformed drafts and unknown evidence IDs fail validation.
- Unsupported claims trigger exactly one corrective retry.
- Partially supported retries return only supported claims.
- Fully unsupported answers use the canonical abstention.
- Tool timeout and failure degrade to retrieval evidence.
- Confidence is deterministic from verification and sufficiency signals.
- Extractive fallback abstains when relevant evidence is absent.
- MCP and other shared-pipeline callers retain existing fields and can expose optional safety metadata.
- Existing grounding, citation, and context-pipeline tests remain green.

## Non-Goals

- Allowing side-effecting tools inside the shared RAG pipeline.
- Replacing `ToolAgentLoop` or its approval flow.
- Guaranteeing that source evidence itself is true.
- Adding a second external fact-checking model in this phase.
- Changing the public response into a mandatory new schema.
- Retrying generation more than once.
