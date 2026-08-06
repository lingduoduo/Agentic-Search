# Tools surface consolidation — design

**Date:** 2026-08-06
**Scope:** the three "Tools" UIs, and making tool inventory visible on `/tools`

## Problem

Three separate UIs are all called "Tools", and the page actually named `/tools`
is the only one that shows nothing *about* tools.

| Where | Component | Backend | Shows |
|---|---|---|---|
| `/tools` route | `ToolAgentView` | `/tool/send-tool-message` | The LLM agent loop |
| Header wrench "Tools" | `components/ToolPanel.tsx` | `/admin/tools` | List, test-invoke, register/delete OpenAPI |
| `/assist` → Dev Console | `components/debug/ToolsPanel.tsx` | `/api/debug/tools` | Catalog by server, discovery ranking |

Four concrete defects:

1. **Two header controls read `Tools`.** `App.tsx:54` is a `NavLink` to `/tools`
   (the agent); `App.tsx:56-63` is a wrench button whose `title` is "Manage
   tools" but whose visible label is also `Tools`. They go to unrelated places.
2. **Two components differ by one letter** — `components/ToolPanel.tsx` versus
   `components/debug/ToolsPanel.tsx`. Nothing in either name says which is the
   admin mutation panel and which is the read-only catalog.
3. **`/tools` cannot show what tools exist.** To see the registry you must set
   `VITE_DEBUG_PANELS=1`, navigate to `/assist`, and open the console — the tool
   inventory lives behind a dev flag on a different page.
4. **Discovery ranking is dev-only.** `SemanticRouter` (TF-IDF, no LLM) is
   reachable only through `POST /api/debug/tools/discover`, and the whole debug
   router is mounted only when `AGENTIC_SEARCH_DEBUG_PANELS` is set
   ([app.py:411](../../../src/internal/servers/web/app.py#L411)). In any normal
   deployment that endpoint does not exist, so the ranking that would explain
   "why did the agent not pick this tool" is unavailable exactly where it helps.

Defect 4 also decides the data source: relocating `/api/debug/tools` to `/tools`
would leave the page blank whenever debug panels are off. `/admin/tools` is
always mounted and `require_admin`, so it is the correct source for a product
surface.

## Design

### 1. `POST /admin/tools/discover`

Add to the existing tools router ([api.py:80](../../../src/internal/servers/tools/api.py#L80)),
alongside the other `require_admin` routes. Body `{query}`; returns
`SemanticRouter(default_tool_catalog()).get_routing_details(query)` — the same
payload the debug endpoint returns, from the same live registry, but reachable
without the debug gate.

The debug endpoint stays as-is. It is a different audience (dev console, no auth
needed because the router is not mounted) and removing it would regress that
panel.

### 2. One catalog renderer, two containers

The catalog view is currently one implementation that only the dev console can
reach. Rather than write a second one for `/tools`, split it:

- **`components/ToolCatalog.tsx`** — presentational only. Props: `servers`,
  `registeredCount`, `discovery`, `onDiscover`, `error`. No fetching.
- **`/tools` container** — fetches `/admin/tools`, groups client-side, calls
  `/admin/tools/discover`.
- **`components/debug/ToolCatalogPanel.tsx`** — fetches `/api/debug/tools`
  (already grouped server-side) and `/api/debug/tools/discover`.

Both render `<ToolCatalog>`. One set of markup, two data sources, no duplicated
rendering.

Grouping on the client mirrors `catalog_from_registry`
([semantic_router.py:45](../../../src/internal/tools/semantic_router.py#L45)):
`openapi`/`mcp` entries group by `provider_id`, everything else goes to `local`,
registry order preserved within a server, empty servers omitted. Extracted as
`groupToolsByServer` in `web/src/toolCatalog.ts` with its own unit test, since it
now exists in two languages and can drift.

### 3. Surface `agent_callable` and `user_scoped`

`/admin/tools` already returns both flags; nothing renders them. `ToolCatalog`
shows a badge for each, because these two flags are the direct answer to "the
tool is registered, so why didn't the agent use it":

- `agent_callable: false` → "not offered to agents". Set for `rag_routing_tool`
  (returns a whole answer, not evidence) and the seeded `search` (built at
  process start with no request identity, so it carries no ACL — the tool agent
  substitutes a request-bound one).
- `user_scoped: true` → "requires a signed-in user", withheld from anonymous
  callers so a write cannot land in a shared bucket.

This is the highest-value information on the page and it costs one badge each.

### 4. Naming

- `components/ToolPanel.tsx` → **`components/ToolAdminPanel.tsx`** (mutations:
  register OpenAPI, delete provider, test-invoke).
- `components/debug/ToolsPanel.tsx` → **`components/debug/ToolCatalogPanel.tsx`**.
- The wrench button's visible label becomes **"Manage tools"**, matching its own
  `title`. The `/tools` nav link keeps "Tools".

Test files move with their components.

## Non-goals

- **Not wiring `SemanticRouter` into the agent loop.** It would change what
  `/tools` actually does — narrowing the tool list before generation, or adding a
  non-LLM fast path — and that is a behavior change to a product surface, not a
  visibility fix. This design only makes the ranking *visible*; deciding whether
  the agent should use it is a separate question.
- **Not changing what `/admin/tools/{name}/invoke` executes.** It invokes the
  registry directly, so test-invoking `search` uses the seeded unfiltered
  instance and can return more than an agent run would. Admin-gated, so it is
  defensible; the `agent_callable` badge now explains the discrepancy rather than
  silently leaving it.
- **Not exposing tool inventory to non-admins.** `/admin/tools` is
  `require_admin`; the `/tools` catalog renders for admins and shows a short note
  otherwise. Tool inventory reveals internal API surface and OpenAPI provider
  names, so admin-only is the conservative default.
- **Not fixing `check_router_auth`.** It logs non-public routes as "guarded"
  without verifying any auth dependency exists, so debug routes are labelled
  guarded while having none. Real weakness, unrelated to this work.

## Verification

1. `pytest tests/unit/test_api_tools.py` plus a new case for the discover route:
   200 with ranked tools for an admin, 401/403 without.
2. `cd web && npx vitest run` — new tests for `ToolCatalog` (renders servers,
   badges, discovery results), `groupToolsByServer`, the `/tools` container, and
   the renamed files' existing suites.
3. `cd web && npm run typecheck`.
4. `ruff check . && ruff format --check .`
5. Manual: `/tools` shows the inventory and ranks tools for a query with no LLM
   involved; the header no longer has two controls labelled "Tools".
