# Agent Dispatch Consolidation — Design

Status: shipped (consolidated in PR #347, alongside the router and the
deterministic auto-search). This doc covers the **dispatch** piece.
Date: 2026-06-29

## Goal

One place builds and runs each agent loop; one place assembles the response.

## Problem

The router (`_run_auto_routed`) and the explicit-mode chain (`_run_agent_impl`)
in `src/internal/servers/web/app.py` were two dispatch sites building/running the
same loops with copy-pasted construction, the `output.context.turns → documents`
extraction, and the persist→read→`AgentExperienceResponse` tail (~6×).

## Approach

Per-loop **runners** returning the canonical
`(answer, citations, documents, intent, extra)` tuple, plus one response tail:

- `_run_search_agent` — `SearchAgentLoop`; `extra` carries `control_flow_trace` +
  `num_turns`.
- `_run_agentic_rag` — `AgenticRAGLoop`; `extra["rounds_used"]`.
- `_run_tool_agent` — `ToolAgentLoop`; `extra` carries `tool_calls`, `num_turns`,
  and `_assistant_fallback`. `answer = output.final_answer or ""` (no fallback
  inside the runner — the two callers apply opposite empty-answer policies).
- `_search_agent_documents` — the single `turns → documents` extractor.
- `_finalize_response(db, session_id, *, answer, citations, documents, intent,
  hook_metadata, extra, mode)` — persists (merging `extra` into metadata; popping
  `tool_calls` and converting `control_flow_trace` to views) and builds the response.

**Runners hold no policy.** Capability *degradation* stays in `_run_auto_routed`;
explicit-mode 400 *guards* stay at those call sites; the tool-agent empty-answer
split is expressed via `extra["_assistant_fallback"]` (auto degrades to RAG on
empty; explicit falls back to the last assistant message). `_run_direct_llm` stays
inline (single caller). Every branch of `_run_agent_impl` ends in
`_finalize_response`.

## Behavior convergence (additive only)

Sharing one runner makes three behaviors converge — additive, nothing removed:
auto-routed `search_agent` now emits `control_flow_trace` + `num_turns`; explicit
`tool_agent` now populates `tool_calls` + documents + inferred intent; `on_trace`
is a runner param.

## Preserved

`/api/agent` + `/api/agent/stream` contracts, `route_query`, degradation chain,
explicit-mode 400 guards, the two tool sets (via `with_search_tool`), sampling
params, dedup, citation ordering.

## Tests

- `tests/unit/servers/web/test_loop_runners.py` — runner tuple/`extra` contracts.
- Existing web suite updated to the converged contract.

## Out of scope

- M10 `Router`-into-loop wiring (PR B); `route_query` / auto-search internals.
