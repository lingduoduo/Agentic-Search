# Move tool-agent material from the README into docs/tool-engine.md

**Date:** 2026-07-21
**Status:** Approved

## Problem

Completing the per-route README extraction (search-engine, chat-engine), the tool
agent — the `tool` route of `route_query` — has no dedicated page. Its material
lives only in the "What it provides" tool bullet, and tool internals are described
inside `docs/mcp.md`, which conflates the internal tool engine with the external
MCP server surface.

## Decision

Keep `docs/mcp.md` as-is (it is a well-scoped MCP-server guide) and add a separate
`docs/tool-engine.md`. MCP is one external consumer of tools; the tool engine is
the web backend's internal `ToolAgentLoop` dispatch. Renaming mcp.md would
mis-file its MCP-protocol content.

## Goal

Populate `docs/tool-engine.md` as a standalone overview of the tool agent —
capabilities, routing, and the tool registry — plus a README pointer section and
docs-list entry. No behavior changes.

## Design

1. **New `docs/tool-engine.md`** with a README back-link:
   - *Capabilities* — `ToolAgentLoop` structured tool dispatch, routing/memory,
     the registered tool catalog.
   - *Routing into the tool engine* — action-command regex cues → `tool`;
     `ToolAgentLoop` with grounded-chat fallback; explicit `tool_agent` mode.
   - *Tool registry and discovery* — `ToolRegistry`, `seed_tools`,
     `register_from_openapi`, `discover_tools` TF-IDF, built-in seed tools.
   - *Relationship to MCP* — client-driven MCP selection is independent of the
     auto-router; `sync_tool_to_mcp` mirrors the registry into MCP. Links to
     `mcp.md` and `request-routing.md`.
2. **README edits:**
   - Add a `## Tool engine` pointer section after `## Search engine`.
   - Add `docs/tool-engine.md` to the Documentation list after API request routing.
   - Keep the short "What it provides" tool bullet in place.

Insertion points chosen to reduce conflict with the open chat-engine (#441) and
ingestion (#442) README edits. No code, API, or schema changes.
