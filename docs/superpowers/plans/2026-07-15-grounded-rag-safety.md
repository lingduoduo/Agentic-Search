# Grounded RAG Safety Implementation Plan

> **Required sub-skill:** Use `superpowers:executing-plans` for inline execution or `superpowers:subagent-driven-development` for delegated execution. Apply `superpowers:test-driven-development` to every behavior change and `superpowers:verification-before-completion` before claiming completion.

**Goal:** Reduce hallucinations across the shared RAG pipeline by constraining generation to retrieved or approved read-only tool evidence, verifying individual claims, retrying once with verifier feedback, and abstaining when evidence is insufficient.

**Architecture:** Normalize retrieval and tool results into evidence records with stable IDs. Ask the LLM for a strict internal claim schema, verify every claim deterministically against cited evidence, retry once when needed, and render only supported claims. Preserve existing public response fields while adding optional confidence and verification metadata.

**Tech stack:** Python 3.12, standard-library dataclasses/protocols/JSON/asyncio, pytest, Ruff.

## Global constraints

- Preserve all existing `AnswerGenerationResult` and MCP response fields.
- Add only backward-compatible optional/defaulted metadata fields.
- Execute only explicitly registered read-only tools; never run side-effecting or unspecified tools.
- Permit at most one corrective retry.
- Remove unsupported claims after retry and use exactly `I don't know based on the available evidence.` when no supported answer remains.
- Do not expose full evidence payloads, prompts, or raw tool logs in public responses or traces.
- Do not introduce an external fact-checker or replace the existing agent tool loop.

### Task 1: Add evidence, structured-draft, and verification primitives

**Files:**

- Modify: `src/context/models.py`
- Create: `src/context/safety.py`
- Modify: `src/context/__init__.py`
- Create: `tests/unit/test_rag_safety.py`

1. Write failing tests for evidence normalization, strict draft parsing, unknown evidence IDs, supported/unsupported claim verdicts, supported-only rendering, canonical abstention, and deterministic confidence.
2. Add `EvidenceSource`, `AnswerClaim`, `AnswerDraft`, `ClaimVerdict`, `VerificationResult`, and `VerificationStatus` models. Use stable `D*` IDs for retrieved documents and `T*` IDs for tool evidence.
3. Implement `evidence_from_context`, `parse_answer_draft`, `verify_answer_draft`, and `render_verified_answer`. Require exact JSON keys and reject malformed or unknown citations.
4. Reuse the grounding tokenizer/overlap semantics. Compute confidence as `0.6 * support_ratio + 0.2 * evidence_coverage + 0.2 * evidence_sufficiency`, substituting `support_ratio` when sufficiency is unavailable and clamping to `[0, 1]`.
5. Run `pytest -q tests/unit/test_rag_safety.py` and commit as `feat: add RAG claim verification core`.

### Task 2: Add the approved read-only tool evidence gateway

**Files:**

- Create: `src/context/tool_evidence.py`
- Modify: `src/context/__init__.py`
- Create: `tests/unit/test_rag_tool_evidence.py`

1. Write failing tests proving that read-only tools can supply evidence, side-effecting and unspecified tools are rejected, calls are bounded, timeouts/errors degrade safely, and serialization/IDs are deterministic.
2. Add `ToolSafety` (`READ_ONLY`, `SIDE_EFFECTING`, `UNSPECIFIED`), tool descriptors, requests, registry/selector protocols, and an async `collect_tool_evidence` gateway.
3. Default to `max_calls=2` and `timeout_seconds=5.0`; stringify tool results as stable JSON and emit only normalized `T*` evidence.
4. Run `pytest -q tests/unit/test_rag_tool_evidence.py` and commit as `feat: add read-only RAG tool evidence`.

### Task 3: Guard generation with structured output, retry, and abstention

**Files:**

- Modify: `src/context/models.py`
- Modify: `src/context/prompts.py`
- Modify: `src/context/pipeline.py`
- Modify: `src/context/__init__.py`
- Modify/Create: focused pipeline tests under `tests/unit/`

1. Add sequence-based fake-LLM tests for a valid supported draft, one corrective retry, partial support, malformed output, total abstention, and exactly two maximum LLM calls. Add a compatibility test for disabling the guard.
2. Add `GroundedGenerationConfig(enabled=True, max_retries=1, overlap_threshold=0.15)`. Extend requests with optional evidence/safety/sufficiency inputs and results with defaulted `confidence`, `verification_status`, `abstained`, and `tool_evidence` metadata.
3. Add strict structured-draft and corrective prompts that require evidence IDs and explicitly allow uncertainty.
4. Update `generate_answer` to parse and verify the draft, retry once with concise verifier feedback, then remove unsupported claims or return the canonical abstention. Keep the legacy path available when the guard is explicitly disabled.
5. Tighten extractive fallback so it abstains when no relevant snippet exists instead of selecting the first document. Preserve the existing grounding report behavior.
6. Run the focused pipeline/model/prompt tests and commit as `feat: enforce grounded RAG generation`.

### Task 4: Integrate retrieval, tools, MCP output, and safe tracing

**Files:**

- Modify: `src/context/pipeline.py`
- Modify: `src/internal/mcp_server/tools/chat.py`
- Modify/Create: context pipeline integration tests
- Modify: relevant MCP chat/auth tests

1. Write failing integration tests covering retrieval evidence, selected read-only tool evidence, unsafe-tool rejection, tool failure fallback, and backward-compatible MCP output.
2. Extend `answer_with_retrieval` with optional `tool_registry`, `tool_selector`, `max_tool_calls=2`, `tool_timeout_seconds=5.0`, and safety configuration. Collect tool evidence after retrieval and before guarded generation.
3. Add only non-sensitive trace metadata: evidence counts/types, tool names/statuses, retry count, verification status, confidence, and abstention. Never trace evidence bodies or raw tool output.
4. Preserve current MCP keys and add `confidence`, `verification_status`, `abstained`, and tool-source summaries.
5. Run the focused context and MCP tests and commit as `feat: integrate verified evidence across RAG callers`.

### Task 5: Document, archive, regenerate context packs, and verify

**Files:**

- Move the approved design and this plan to `docs/superpowers/archive/`
- Regenerate generated context-pack artifacts and their index
- Update user-facing RAG documentation where the repository currently documents this pipeline

1. Document the evidence contract, approved-tool policy, retry/abstention behavior, confidence semantics, compatibility switch, and operational metadata.
2. Use `git mv` to archive the design and plan after implementation approval, then run the repository context-pack generator and its checks.
3. Run focused tests, Ruff checks/format validation, `git diff --check`, and the full test suite. If the two known environment-dependent tests remain unchanged, report them explicitly and run the accepted suite with those exact tests deselected.
4. Inspect the final diff for public-schema compatibility, accidental evidence leakage, unbounded retries/tool calls, and generated-artifact drift.
5. Commit as `docs: document grounded RAG safety`, push the branch, and update draft PR #414 with the design, verification evidence, and known environment limitations.

## Completion criteria

- Every generated factual claim is tied to normalized retrieval or approved read-only tool evidence.
- Unsupported claims receive no more than one corrective retry and are never rendered afterward.
- No supported answer produces the exact canonical abstention.
- Existing callers continue to receive their established fields.
- Tool calls are allowlisted, read-only, bounded, timed out, and failure-tolerant.
- Focused and accepted full-suite verification passes, generated context artifacts are current, and the draft PR describes the change accurately.
