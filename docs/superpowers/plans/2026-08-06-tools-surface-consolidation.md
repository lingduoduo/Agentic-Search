# Tools surface consolidation — implementation plan

Spec: [2026-08-06-tools-surface-consolidation-design.md](../specs/2026-08-06-tools-surface-consolidation-design.md)

## Task 1 — `POST /admin/tools/discover`

Add to `create_tools_router` in `src/internal/servers/tools/api.py`, next to the
other `require_admin` routes. Body model `DiscoverRequest{query: str}`; delegate
to `SemanticRouter(default_tool_catalog()).get_routing_details(query)`. Import
inside the handler, matching how the debug router does it, so the router module
does not pull in the semantic router at import time.

→ verify: `pytest tests/unit/test_api_tools.py` — new cases for a ranked 200 and
for the unauthenticated rejection.

## Task 2 — `groupToolsByServer`

New `web/src/toolCatalog.ts`. Takes `ToolView[]`, returns `CatalogServer[]`:
group `openapi`/`mcp` by `provider_id`, everything else under `local`, preserve
input order within a server, omit empty servers. Mirrors
`catalog_from_registry`.

→ verify: new `web/src/__tests__/toolCatalog.test.ts` — mixed sources group
correctly, order preserved, a missing `provider_id` falls back to `local`.

## Task 3 — `ToolCatalog` presentational component

New `web/src/components/ToolCatalog.tsx`. Props: `servers`, `registeredCount`,
`discovery`, `onDiscover`, `error`, `note`. Renders the count, one block per
server with each tool's name / source badge / description, `agent_callable` and
`user_scoped` badges with `title` text explaining each, a discovery query box,
and the ranked results. No fetching, no `useEffect`.

→ verify: new `web/src/components/__tests__/ToolCatalog.test.tsx` — servers and
tools render, both badges appear only when applicable, `onDiscover` fires with
the typed query, results render, `note` renders when passed.

## Task 4 — render the catalog on `/tools`

In `ToolAgentView`, fetch `/admin/tools` on mount, group with
`groupToolsByServer`, and render `<ToolCatalog>` below the agent chat. Add
`discoverAdminTools(query)` to `web/src/api.ts` for the new endpoint. On 401/403,
render `ToolCatalog` with a `note` saying tool inventory requires admin rather
than an error banner — a non-admin using the agent is not in an error state.

→ verify: extend `web/src/components/__tests__/ToolAgentView.test.tsx` — catalog
renders when `/admin/tools` resolves; the admin note renders on 403; the agent
chat still works in both cases.

## Task 5 — repoint the dev console

Rename `components/debug/ToolsPanel.tsx` →
`components/debug/ToolCatalogPanel.tsx`, keeping its `/api/debug/tools` +
`/api/debug/tools/discover` fetching but rendering `<ToolCatalog>` instead of its
own markup. Update the import in `DevConsole.tsx` and move
`debug/__tests__/ToolsPanel.test.tsx` → `ToolCatalogPanel.test.tsx`.

→ verify: `npx vitest run` — the moved suite passes against the shared renderer.

## Task 6 — rename the admin panel and fix the label

`git mv web/src/components/ToolPanel.tsx web/src/components/ToolAdminPanel.tsx`
and the same for its test file; rename the exported symbol; update the import in
`App.tsx`. Change the wrench button's visible label from `Tools` to
`Manage tools`.

→ verify: `App.test.tsx` — the nav link is still the only control named exactly
"Tools"; the wrench control is found by "Manage tools". Update the existing
assertion that matches `/^tools$/i`.

## Task 7 — full verification

1. `pytest tests/unit/test_api_tools.py -q`
2. `python3 -m pytest -q` (whole suite; expect the one known-flaky
   `test_mcp_document_tools.py::test_parser_timeout_terminates_the_worker_process`,
   which fails on `main` too)
3. `cd web && npx vitest run`
4. `cd web && npm run typecheck`
5. `ruff check . && ruff format --check .`
6. `grep -rn "ToolPanel\|debug/ToolsPanel" web/src` → no stale references

## Task 8 — PR

Commit, push `feat/tools-surface-consolidation`, open a PR. Separate branch from
`refactor/simplify-cli` (PR #501) — different deliverable, no stacking.
