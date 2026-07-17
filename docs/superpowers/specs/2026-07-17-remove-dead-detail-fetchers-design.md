# Remove Dead Frontend Detail Fetchers — Design

**Date:** 2026-07-17
**Branch:** `chore/remove-dead-detail-fetchers`
**Status:** Approved (housekeeping)

## Motivation

Investigation of the dashboard's Tools/Connectors panels found three exported
frontend symbols with **zero callers**:

- `getTool(name)` — `GET /admin/tools/{name}` ([api.ts](../../../web/src/api.ts))
- `getConnector(id)` — `GET /admin/connectors/{id}`
- `ConnectorDetailView` (type) — used only as `getConnector`'s return type

The panels use the collection endpoints (`listTools` / `listConnectors`) and
never fetch a single item, so these detail fetchers are dead code in the UI.

## Scope

### In scope
- Delete `getTool` and `getConnector` from `web/src/api.ts`.
- Delete `ConnectorDetailView` from `web/src/types.ts` and its now-unused import
  in `api.ts`.

### Non-goals
- **No backend change.** `GET /admin/tools/{name}` is test-covered
  (`test_tool_admin_api.py`) and `GET /admin/connectors/{id}` is part of the
  admin REST surface (used by tests / external clients) — they are *not* dead
  server-side, only unconsumed by the UI.
- No other frontend refactor.

## Verification

- `cd web && npm run typecheck` — passes (nothing references the removed
  symbols; if anything did, the type-check would fail).
- `cd web && npm run build` — succeeds.

## Success criteria

1. `getTool`, `getConnector`, `ConnectorDetailView` are gone from `web/src`.
2. `npm run typecheck` and `npm run build` pass.
