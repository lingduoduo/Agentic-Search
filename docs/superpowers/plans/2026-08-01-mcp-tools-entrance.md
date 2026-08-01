# MCP Tools Entrance Implementation Plan

**Goal:** Let the web process pull tools *in* from configured MCP servers so the tool agent can call something other than retrieval wrappers.

**Architecture:** One new module, `src/internal/tools/mcp_client.py`, plus config, a startup task, and three small guards. Off unless `AGENTIC_SEARCH_MCP_SERVERS` is set.

**Tech Stack:** Python 3.12, FastAPI, `mcp` 1.27 client SDK (already installed — no new dependency), pytest.

**Spec:** `docs/superpowers/specs/2026-08-01-mcp-tools-entrance-design.md`

## Global Constraints

- Work on branch `feat/mcp-tools-entrance`. Never commit to `main`.
- Default-off: with no config set, behaviour is byte-for-byte what it was.
- An unreachable MCP server must never stop the web process from starting.
- No new dependency.
- `python3 -m pytest` and `ruff check . && ruff format .` pass before commit.

## Tasks

- [x] **Task 1 — Config.** `AGENTIC_SEARCH_MCP_SERVERS` (`name=url` pairs) and
      `AGENTIC_SEARCH_MCP_TOKEN` on `AppSettings`.
      *Verify:* parse tests for pairs, empty, malformed, token→header.

- [x] **Task 2 — Client module.** `McpServerSpec`, `parse_mcp_servers`,
      `_connect`, `register_mcp_tools`; each remote tool wrapped as a
      `FunctionTool` registered with `source="mcp"`, `provider_id=<server>`.
      *Verify:* tools register under the right source/provider, invoking one
      calls the remote server, remote errors surface as text, an unreachable
      server registers nothing without raising.

- [x] **Task 3 — Export-loop guard.** `_exportable_entries()` in the outbound
      bridge filters `source == "mcp"`.
      *Verify:* after pulling tools in, nothing is exportable.

- [x] **Task 4 — Recursion guard.** `ask_agentic_search` joins
      `_SHADOWED_TOOL_NAMES`.
      *Verify:* the agent's shadow list contains it.

- [x] **Task 5 — Dev Console grouping.** `catalog_from_registry` groups `mcp`
      entries by `provider_id` alongside `openapi`.
      *Verify:* catalog shows one group per MCP server.

- [x] **Task 6 — Startup wiring.** Schedule discovery as a background task in
      lifespan, not an await.
      *Verify:* live run shows discovery firing after "Application startup
      complete"; awaiting instead 401s against our own server.

- [x] **Task 7 — Docs.** Inbound section in `docs/mcp.md`.

## Verification

| Gate | Command | Result |
| --- | --- | --- |
| Unit + regression | `python3 -m pytest` | 2794 passed |
| Lint | `ruff check . && ruff format .` | clean |
| End-to-end | live MCP :8090 + web :7860 | 13 tools registered under server `agentic`; `expand_query` and `save_memory` round-trip |
