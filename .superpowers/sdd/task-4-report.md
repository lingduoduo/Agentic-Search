# Task 4 report

## Status

Implemented retrieval-plus-tool evidence integration, bounded read-only execution,
failure-tolerant fallback, guarded safety configuration, metadata-only tracing,
and backward-compatible MCP response metadata. Updated downstream agentic and
MCP fixtures to the guarded structured-draft contract.

## RED evidence

- `pytest -q tests/unit/test_rag_pipeline_integration.py tests/unit/test_mcp_server.py -x`
  - Collected 36 tests.
  - Failed `test_pipeline_combines_retrieval_and_selected_read_only_tool` with
    `TypeError: answer_with_retrieval() got an unexpected keyword argument
    'tool_registry'`.
- First broad regression run, `pytest -q tests/unit`, reported `2482 passed, 11
  failed`: ten legacy agentic free-text fixtures exhausted their finite response
  sequences under guarded retry, and one unrelated Hugging Face cache lock was
  denied by the sandbox. This directly confirmed the Task 3 handoff noted in the
  brief.

## GREEN evidence

- Focused integration and MCP: `pytest -q tests/unit/test_rag_pipeline_integration.py
  tests/unit/test_mcp_server.py` -> `36 passed in 2.11s`.
- Context, gateway, guarded generation, agentic, MCP/auth, and tracing regression:
  the final focused command collected 126 tests -> `126 passed in 4.44s`.
- Full unit suite with a sandbox-writable cache:
  `env HF_HOME=/tmp/agentic-search-hf-cache pytest -q tests/unit` ->
  `2493 passed, 6 warnings in 40.84s`.
- Ruff over every modified Python file -> `All checks passed!`.
- `git diff --check` -> exit 0.

## Files

- `src/context/pipeline.py`: optional gateway and safety inputs, retrieval/tool
  evidence assembly before guarded generation, and safe summary tracing.
- `src/context/tool_evidence.py`: optional status reporting for rejected, failed,
  and successful registered tool calls without arguments or outputs.
- `src/context/models.py`: optional result retry count used by safe tracing.
- `src/internal/mcp_server/tools/chat.py`: additive confidence, verification,
  abstention, and tool-source summary fields on success and fallback responses.
- `tests/unit/test_rag_pipeline_integration.py`: retrieval/tool integration,
  unsafe rejection, tool-failure fallback, and trace-redaction coverage.
- `tests/unit/test_mcp_server.py`: additive MCP contract and guarded fixtures.
- `tests/unit/test_agentic_rag.py` and
  `tests/unit/servers/web/test_web_experience_app.py`: guarded structured fixtures
  preserving routing/control-flow intent.
- `tests/unit/observability/test_tracer.py`: prompt text is explicitly excluded
  from trace attributes.

## Self-review

- Reused `collect_tool_evidence`; no second registry, gateway, or verifier exists.
- Selector visibility remains read-only only; duplicate, unsafe, unspecified, and
  unknown tools cannot execute. Calls remain bounded and individually timed out.
- Retrieval remains usable if selection, invocation, serialization, or timeout
  handling yields no tool evidence.
- Trace attributes contain counts/types, registered tool names/statuses, retry
  count, verification status, confidence, and abstention only. Query/prompt text,
  retrieval bodies, tool arguments, raw results, exception text, and search URLs
  are absent.
- Existing MCP keys remain unchanged; new fields are additive and tool summaries
  expose names only, never bodies.
- Guarded retry count is zero or one, matching the existing hard retry cap.

## Concerns

- None. The six full-suite warnings are pre-existing dependency/deprecation and
  empty-gradient warnings; there were no test failures.

## Review fixes: bounded selector and canonical MCP abstention

### RED evidence

- `pytest -q tests/unit/test_rag_tool_evidence.py tests/unit/test_mcp_server.py -x`
  - Collected 46 tests.
  - Failed `test_selector_exceptions_degrade_to_no_tool_evidence[False]` because
    the synchronous selector exception escaped `collect_tool_evidence`.
- `pytest -q tests/unit/test_rag_tool_evidence.py -k iteration_failure`
  - Failed because an exception raised while traversing the selector's iterable
    escaped instead of degrading to retrieval-only answering.

### GREEN evidence

- `pytest -q tests/unit/test_rag_tool_evidence.py tests/unit/test_mcp_server.py
  tests/unit/test_agentic_rag.py` -> `64 passed in 4.39s`.
- Final context/integration/MCP/agentic regression command -> `131 passed`.
- Ruff check and format check over all review-modified Python files -> clean.
- `git diff --check` -> exit 0.

### Changes and review

- Synchronous selector calls, returned awaitables, and bounded selection
  consumption each have an explicit timeout using `timeout_seconds`; sync/async
  exceptions, timeouts, and iteration failures all return no tool evidence.
- Selection traversal is capped at `max_tool_calls * 4` before request handling,
  so rejected or unknown selections cannot cause unbounded consumption.
- Empty and exception MCP paths use the canonical abstention exactly, with
  confidence `0.0`, verification status `abstained`, `abstained=True`, retry count
  `0`, empty citations/sources/tool summaries, and a generic stable error on the
  exception path. Raw exception text is not returned.
- MCP success metadata now also carries the guarded retry count. Existing response
  keys and retrieval/tool behavior are preserved.
- Agentic guarded-contract coverage now asserts the rendered verified answer and
  citation, rather than only checking for a non-empty answer.

### Concerns

- None.

## Review fix: fail-closed tool discovery

### RED evidence

- `pytest -q tests/unit/test_rag_tool_evidence.py
  tests/unit/test_rag_pipeline_integration.py -x`
  - Collected 21 tests.
  - Failed `test_discovery_failures_degrade_to_no_tool_evidence[descriptors0]`
    because `registry.list_tools()` raised before the optional-tool failure
    boundary.

### GREEN evidence

- `pytest -q tests/unit/test_rag_tool_evidence.py
  tests/unit/test_rag_pipeline_integration.py` -> `21 passed in 0.87s`.
- Final gateway/pipeline/context/MCP/agentic/tracing regression command ->
  `135 passed in 4.29s`.
- Ruff check and final Ruff format check over the modified Python files -> clean.
- `git diff --check` -> exit 0.

### Changes and review

- Registry discovery, descriptor shape validation, duplicate counting, and eligible
  descriptor construction now share the optional-tool fail-closed boundary.
- Discovery exceptions, non-list results, non-`ToolDescriptor` entries, empty or
  non-string names, and invalid safety values return no tool evidence before the
  selector is called or any tool can execute.
- Pipeline coverage proves that a supported retrieval-only guarded answer still
  succeeds when tool discovery raises.

### Operational note for Task 5

- Timing out a synchronous selector returns control to the RAG pipeline, but the
  worker may continue running in the executor because Python cannot safely cancel
  an already-running thread. Task 5 operational documentation must call out this
  executor-lifetime behavior; no unsafe thread-cancellation redesign was attempted.

### Concerns

- None beyond the documented executor-lifetime behavior assigned to Task 5.

## Summary

- Added `SearchPipeline`, which builds bounded follow-up context and composes retrieval, ranking, and grounded inference behind the existing five-value result tuple.
- Added deterministic empty/unreachable handling and an evidence-only fallback when inference fails.
- Adapted the existing web hybrid provider policy into the stage contracts. Provider precedence, filters, route response models, and hybrid ranking remain owned by the existing web helpers.
- Removed a runtime-only serving protocol import from `stages.py` to prevent a collection-time web/search-pipeline import cycle.

## TDD evidence

- Red: `pytest tests/unit/search_pipeline/test_pipeline.py -q` failed during collection because `src.internal.search_pipeline.pipeline` did not exist.
- Green: the focused and adjacent verification below passes.

## Verification

- `pytest tests/unit/search_pipeline tests/unit/servers/web/test_reranking.py tests/unit/test_execution_fallbacks.py tests/unit/servers/web/test_web_experience_app.py -q` — 88 passed.
- `ruff check src/internal/search_pipeline src/internal/servers/web/app.py tests/unit/search_pipeline/test_pipeline.py` — passed.
- `ruff format --check src/internal/search_pipeline tests/unit/search_pipeline` — passed.
- `git diff --check` — passed.

## Concerns

- None known. Public routes and response models are unchanged; stage metadata is carried only in the existing `extra` mapping.

## ACL follow-up

- Added a regression for filtered `source_provider="auto"` fanout covering internal retrieval, SerpAPI, and browser fallback.
- Internal retrieval now receives the access filter object; SerpAPI and browser are called without internal ACL metadata.
- Red evidence: the focused test observed `"not-passed"` for the retrieval leg instead of the expected user filter.
