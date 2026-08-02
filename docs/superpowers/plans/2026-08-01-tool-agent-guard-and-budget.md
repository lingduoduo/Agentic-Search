# Tool-Agent Guard and Budget Implementation Plan

**Goal:** Move the tool agent's exclusion rule from a downstream name match to a registration property, and stop long answers being cut by budgets nobody configured.

**Architecture:** `ToolEntry.agent_callable` set at seed/MCP registration; the runner filters on it instead of on names. Budgets become configuration, with the wall-clock timeout — the one that actually bites — exposed for the first time.

**Tech Stack:** Python 3.12, FastAPI, transformers, pytest.

**Spec:** `docs/superpowers/specs/2026-08-01-tool-agent-guard-and-budget-design.md`

## Global Constraints

- Work on branch `fix/tool-agent-guard-and-budget`. Never commit to `main`.
- Excluded tools stay registered and directly invocable — only agent loops lose them.
- No change to `ToolAgentLoop` rollout accounting (the training path depends on it).
- Default timeout unchanged; it becomes configurable, not longer.
- `python3 -m pytest` and `ruff check . && ruff format .` pass before commit.

## Tasks

- [x] **Task 1 — `agent_callable` on the registry.** Field on `ToolEntry`, kwarg
      on `register`, `agent_tools()` accessor, exposed in both summaries.
      *Verify:* default True; a False tool is hidden from `agent_tools()` but
      still returned by `get`/`list`.

- [x] **Task 2 — Set it where the knowledge lives.** `seed_tools` marks
      `rag_routing_tool`; `register_mcp_tools` marks per-server
      `agent_exclude`, defaulting to `DEFAULT_AGENT_EXCLUDE` and overridable
      via `AGENTIC_SEARCH_MCP_AGENT_EXCLUDE`.
      *Verify:* seeded RAG tool and the recursive MCP tool are both registered
      yet absent from `agent_tools()`; a server without the exclusion keeps it.

- [x] **Task 3 — Runner drops its name list.** `_SHADOWED_TOOL_NAMES` deleted;
      `tool_registry.agent_tools()` used instead.
      *Verify:* existing tool-selection tests still pass unchanged.

- [x] **Task 4 — Answer budget.** `tool_agent_max_tokens` config caps one
      generation; `response_length` scales past it so tool traffic cannot starve
      the answer.
      *Verify:* `max_tokens` matches config, `response_length` strictly exceeds it.

- [x] **Task 5 — Parser scaffolding.** `JSONToolParser` strips `</?tool_call>`
      when it extracted calls.
      *Verify:* a Hermes-wrapped call yields empty content; prose around a call
      survives.

- [x] **Task 6 — Expose the wall clock.** `AGENTIC_SEARCH_GENERATION_TIMEOUT`
      plumbed to the local server manager, which hardcoded 120s.
      *Verify:* default 120.0; env override respected.

- [x] **Task 7 — Verify against the real model**, on a question long enough to
      hit every cap.

## Verification

| Gate | Command | Result |
| --- | --- | --- |
| Unit + regression | `python3 -m pytest` | 2808 passed |
| Lint | `ruff check . && ruff format .` | clean |
| End-to-end | live model + retrieval, long-answer prompt | 11 tokens of markup → 298 tokens of prose |
