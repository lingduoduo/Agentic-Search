# Tool registry inspection (Dev Console)

**Date:** 2026-07-21
**Status:** Approved

## Problem

The tool engine documents a `ToolRegistry` and a semantic discovery layer
(`discover_tools`, server/tool catalog), but nothing surfaces them live. The only
existing view is `GET /admin/tools` (admin-auth-gated), which returns a **flat**
list of registered tools via `tool_registry.all_summaries()` and powers
`web/src/components/ToolPanel.tsx`. It does **not** expose:

- the **catalog grouping** (`catalog_from_registry` → `local` server + one server
  per OpenAPI provider), or
- the **semantic discovery** ranking (`SemanticRouter.get_routing_details` /
  `discover_tools`) the tool-engine doc describes.

There is no Dev Console panel for tools at all.

## Goal

Add a read-only Dev Console inspection surface for the tool registry and its
discovery layer: an `/api/debug/tools` endpoint plus a `POST` discovery endpoint,
and a "Tools" panel that renders the catalog grouped by server and lets a user
test discovery for a query. No changes to the registry or semantic router
themselves; no tool invocation (that already lives in the admin ToolPanel).

## Design

### Backend — `src/internal/servers/web/debug_router.py`

Two handlers added to the existing `create_debug_router` factory (prefix
`/api/debug`, auto-mounted by the `if debug_panels:` block in
`src/internal/servers/web/app.py`; no new auth, no extra wiring). Both read the
process-wide module singleton `tool_registry` (`from src.tools.registry import
tool_registry`) — the same access pattern as `src/internal/servers/tools/api.py`.

1. `GET /api/debug/tools` (sync) →
   ```json
   {
     "registered": [ {"name","description","parameters","source","provider_id"}, ... ],
     "catalog":    [ {"name","description","tools":[{"name","description","source","server"}]}, ... ]
   }
   ```
   - `registered` = `tool_registry.all_summaries()`.
   - `catalog` = `catalog_from_registry(tool_registry)` (from
     `src.tools.semantic_router`), each `ServerDefinition`/`ToolDefinition`
     serialized to plain dicts (`dataclasses.asdict`).

2. `POST /api/debug/tools/discover` (sync), body `{"query": str}` →
   ```json
   { "stage1_servers": [...], "stage2_tools": [...], "final_tools": [...] }
   ```
   - Builds a `SemanticRouter` over `default_tool_catalog()` and returns
     `get_routing_details(query)`. Ranking is TF-IDF, so it needs no LLM.
   - Request model `DebugToolDiscoverRequest` with a `Field`/validator rejecting
     blank/whitespace queries → FastAPI 422, mirroring
     `DebugQueryTransformRequest`.

Empty registry (before seeding) yields `registered: []`, `catalog: []`, and empty
discovery stages — no error.

### Frontend — `web/src/components/debug/ToolsPanel.tsx`

Follows the `EvalResultsPanel.tsx` pattern (`useEffect` + `alive` guard):

- On mount, `getDebugTools()`; render the **catalog grouped by server** (server
  header + description, then its tools with a `source` badge) and the registered
  count.
- A query `<input>` + "Discover" button calls `discoverTools(query)` and renders
  the ranked `stage1_servers → stage2_tools → final_tools`.
- Add `getDebugTools()` and `discoverTools(query)` to `web/src/api.ts` (via the
  shared `requestJson` helper), response types to `web/src/types.ts` (reuse the
  existing `ToolView` for `registered`), and render `<ToolsPanel/>` in
  `web/src/components/debug/DevConsole.tsx`.

### Testing

- **Backend** (FastAPI TestClient with debug panels enabled, using the
  model-load-skip harness noted in the project memory / `tests/conftest.py`):
  - `GET /api/debug/tools`: `registered` includes the seeded `web_search`,
    `search`, `search_routing_tool`; `catalog` groups function tools under a
    `local` server.
  - `POST /api/debug/tools/discover`: a relevant query surfaces a matching tool in
    `final_tools`; blank query → 422.
- **Frontend**: a `ToolsPanel` fetch/render test mirroring existing debug-panel
  tests if that pattern exists; otherwise a minimal render test.

## Non-goals

- No tool invocation (exists in the admin ToolPanel).
- No new public/unauthenticated `/api/tools`.
- No changes to `ToolRegistry` or `SemanticRouter` internals.
