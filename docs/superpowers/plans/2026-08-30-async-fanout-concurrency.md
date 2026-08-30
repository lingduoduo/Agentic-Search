# Async fan-out concurrency — plan

Design: `docs/superpowers/specs/2026-08-30-async-fanout-concurrency-design.md`

## Task 1 — Audit

Enumerate every coroutine in `src/` and classify by anti-pattern: blocking calls
inside `async def`, awaits inside loops, coroutines with no await, deprecated
event-loop APIs.

*Verify:* the audit distinguishes genuinely independent I/O from sequential
dependencies; each candidate is confirmed by reading the call site, not by the
pattern match alone.

## Task 2 — Baseline

Benchmark the three candidate sites with their I/O stubbed at a fixed 50 ms, so
the measurement is scheduling shape rather than network noise. Warm lazy imports
outside the measured span.

*Verify:* each baseline equals `n × 50 ms`, confirming sum-of-latencies.

## Task 3 — `AgenticRAGLoop.run`

Gather the round's `retrieve_context` calls; merge results in `novel_queries`
order inside the existing per-query `try`.

*Verify:* `tests/unit/test_agentic_rag.py` green, including the pre-existing
`test_run_handles_retrieval_error_gracefully`.

## Task 4 — `collect_tool_evidence`

Split the loop into a sequential screening pass and a gathered invocation pass.

*Verify:* `tests/unit/test_rag_tool_evidence.py` green, including the
`status_callback` ordering assertions.

## Task 5 — `register_mcp_tools`

Gather per-server discovery; keep registration walking `specs` in order.

*Verify:* `tests/unit/test_mcp_client.py` and
`tests/unit/test_agent_callable_tools.py` green.

## Task 6 — Regression guards

Add one test per site asserting peak in-flight concurrency > 1.

*Verify:* mutation-check — force each call site back to sequential and confirm
only the new guard (plus, for AgenticRAG, the existing error-path test) turns
red.

## Task 7 — Full verification

*Verify:* `ruff check . && ruff format .` clean; full `pytest` green; the three
changed modules still import with `torch` blocked from `sys.meta_path` (the
torch-free CI job).
