# Task 3 Report: Guarded Generation

## Status

Implemented guarded-by-default answer generation with strict structured drafts,
one bounded corrective retry, deterministic verification metadata, unsupported
claim filtering, and canonical abstention. Legacy free-text behavior remains
available only through `GroundedGenerationConfig(enabled=False)`.

## RED evidence

- `pytest -q tests/unit/test_grounded_generation.py -x`
  - Collected 10 tests.
  - Failed `test_guarded_generation_renders_valid_supported_draft` because the
    structured JSON was returned verbatim instead of being parsed and rendered.
- A subsequent focused RED for the no-evidence branch:
  `pytest -q tests/unit/test_grounded_generation.py -x`
  - Failed `test_guarded_generation_abstains_without_calling_llm_when_evidence_is_empty`
    because the LLM was called and exhausted its response sequence.

## GREEN evidence

- `pytest -q tests/unit/test_grounded_generation.py`
  - `11 passed in 0.74s` after the no-evidence correction.
- `pytest -q tests/unit/test_context_pipeline.py tests/unit/test_grounding.py tests/unit/test_rag_safety.py tests/unit/test_grounded_generation.py`
  - `66 passed in 0.82s`.
- `ruff check src/context/models.py src/context/prompts.py src/context/pipeline.py src/context/__init__.py tests/unit/test_grounded_generation.py tests/unit/test_context_pipeline.py`
  - `All checks passed!`
- `git diff --check`
  - Exit 0, no whitespace errors.

## Files

- `src/context/models.py`: added `GroundedGenerationConfig`, optional request
  evidence/safety/sufficiency inputs, and backward-compatible result metadata.
- `src/context/prompts.py`: added strict structured-draft and corrective prompts.
- `src/context/pipeline.py`: added guarded parsing, verification, one retry,
  filtering/abstention, metadata, and conservative extractive fallback.
- `src/context/__init__.py`: exported the new public config and prompt builders.
- `tests/unit/test_grounded_generation.py`: sequence-based guard coverage.
- `tests/unit/test_context_pipeline.py`: made the legacy free-text compatibility
  expectation explicitly disable the guard.

## Full regression run

`pytest -q tests/unit` produced `2474 passed, 13 failed`. Twelve failures are
downstream agentic/MCP tests whose finite fake-LLM sequences still provide
legacy free text to the now guarded-by-default `generate_answer`; updating those
adapters/tests belongs to Task 4 pipeline integration. The remaining failure is
an unrelated sandbox denial writing a Hugging Face cache lock under the user
home directory.

## Self-review

- Retry count is hard-capped to one corrective call even if callers construct a
  config with a larger `max_retries`; total LLM calls never exceed two.
- Explicit model abstention and empty evidence do not trigger unnecessary retries.
- Parse failures receive concise parser feedback; support failures receive exact
  unsupported claim text and verifier reason.
- Final rendering uses the existing `render_verified_answer`, so unsupported
  claims cannot leak after the retry.
- Existing grounding-report verification still runs after guarded rendering and
  its focused regression tests pass.
- No registry/selector or MCP/web adapter integration was added.

## Concerns

- Task 4 must update agentic/MCP generation callers and their free-text fake LLM
  fixtures to emit the structured draft contract (or explicitly choose legacy
  mode where compatibility is genuinely intended).
- Full-suite Hugging Face cache verification requires a writable external cache
  location; it is unrelated to these changes.
