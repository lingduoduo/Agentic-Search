# Intent Example-Query Chips — Design Spec

**Date:** 2026-06-21
**Status:** Approved

## Problem

The intent-routed search UI (see `2026-06-15-intent-routed-search-chat-design.md`) has a single input box with no mode selector — good for end users, but it gives newcomers and testers no quick way to exercise the three routing intents (search / chat / tool). To test routing you must know what kind of query triggers each path.

## Goal

Add a small set of clickable **example-query chips** under the search box — one per intent — so anyone can populate and run a representative query in one click. Plus broaden the frontend test coverage for the intent components.

## Scope

Frontend only. No backend changes.

- `web/src/components/SearchComposer.tsx`
- `web/src/App.tsx`
- `web/src/styles.css`
- `web/src/components/__tests__/SearchComposer.test.tsx`
- `web/src/components/__tests__/App.test.tsx`

## Design

### Example queries

Three representative queries, one per intent:

| Intent | Query |
|---|---|
| search | `find the onboarding checklist` |
| chat | `explain how FAISS indexing works` |
| tool | `summarize the latest sales figures and chart them` |

(The `tool` query only routes to the tool path when a local model + MCP tools are configured; it is still a valid query to test with otherwise.)

### Component

`SearchComposer` gains an optional prop:

```ts
onExampleSelect?: (query: string) => void;
```

A `.example-chips` row renders under the textarea: one `<button type="button" className="example-chip">` per example, labeled with an intent icon + short label (e.g. "🔍 Search example"). The chip's `title`/`aria-label` carries the full query.

- Click → `onExampleSelect(query)` when provided, else `onQueryChange(query)` (fill-only fallback).
- Chips are hidden while `isLoading` (no chip clicks mid-request).

### App wiring (stale-state-safe)

`handleSubmit` currently reads `query` from state. Extend it to accept an optional explicit query so a one-click run uses the chip's query, not stale state:

```ts
const handleSubmit = useCallback(async (eventOrQuery?: FormEvent | string) => {
  if (eventOrQuery && typeof eventOrQuery !== "string") eventOrQuery.preventDefault();
  const raw = typeof eventOrQuery === "string" ? eventOrQuery : query;
  const normalizedQuery = raw.trim();
  ...
}, [query, ...]);
```

The form `onSubmit` and Cmd+Enter paths are unchanged (they pass a `FormEvent` or nothing → use state `query`). The chip handler:

```ts
onExampleSelect={(q) => { setQuery(q); handleSubmit(q); }}
```

`setQuery(q)` keeps the textarea in sync for display; `handleSubmit(q)` runs the explicit query immediately.

## Testing

**`SearchComposer.test.tsx`**
- renders 3 example chips (one per intent).
- clicking a chip calls `onExampleSelect` with that example's query.
- chips are not rendered while `isLoading`.

**`App.test.tsx`**
- clicking an example chip runs the agent (`streamAgent` called) and renders the result with the matching `intent-*` layout class.
- undefined/unknown intent → no `.intent-badge` and no `intent-*` class (coverage broadening).

## Out of scope

- Configurable / admin-editable example sets.
- Auto-submit behavior changes for the normal typed-query path.
- Any backend or routing change.

## Success criteria

1. Three chips render under the search box; each populates and runs its query in one click.
2. Chips disappear during a request.
3. `npm run typecheck` clean; all existing + new vitest tests pass.
