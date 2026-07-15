# Generated Context Pack

# Intent Example Query Chips

## Sources

- [Specification: 2026-06-21-intent-example-query-chips-design.md](../specs/2026-06-21-intent-example-query-chips-design.md)
- [Plan: 2026-06-21-intent-example-query-chips.md](../plans/2026-06-21-intent-example-query-chips.md)

## Specification Context

### Goal

Add a small set of clickable **example-query chips** under the search box — one per intent — so anyone can populate and run a representative query in one click. Plus broaden the frontend test coverage for the intent components.

### Scope

Frontend only. No backend changes.

- `web/src/components/SearchComposer.tsx`
- `web/src/App.tsx`
- `web/src/styles.css`
- `web/src/components/__tests__/SearchComposer.test.tsx`
- `web/src/components/__tests__/App.test.tsx`

### Component

`SearchComposer` gains an optional prop:

```ts
onExampleSelect?: (query: string) => void;
```

A `.example-chips` row renders under the textarea: one `<button type="button" className="example-chip">` per example, labeled with an intent icon + short label (e.g. "🔍 Search example"). The chip's `title`/`aria-label` carries the full query.

- Click → `onExampleSelect(query)` when provided, else `onQueryChange(query)` (fill-only fallback).
- Chips are hidden while `isLoading` (no chip clicks mid-request).

### Testing

**`SearchComposer.test.tsx`**
- renders 3 example chips (one per intent).
- clicking a chip calls `onExampleSelect` with that example's query.
- chips are not rendered while `isLoading`.

**`App.test.tsx`**
- clicking an example chip runs the agent (`streamAgent` called) and renders the result with the matching `intent-*` layout class.
- undefined/unknown intent → no `.intent-badge` and no `intent-*` class (coverage broadening).

### Out of scope

- Configurable / admin-editable example sets.
- Auto-submit behavior changes for the normal typed-query path.
- Any backend or routing change.

## Implementation Plan Context

### Global Constraints

- No backend changes.
- Existing form-submit and Cmd+Enter paths must be unchanged (still use state `query`).
- Chips hidden while `isLoading`.

---

### Task 1: SearchComposer chips (TDD)

**Files:** `web/src/components/SearchComposer.tsx`, `web/src/components/__tests__/SearchComposer.test.tsx`

- [ ] Write failing tests: 3 chips render; clicking a chip calls `onExampleSelect` with the example query; no chips while `isLoading`.
- [ ] Run vitest → fail.
- [ ] Implement: `EXAMPLE_QUERIES` constant (search/chat/tool); optional `onExampleSelect?: (q: string) => void` prop; `.example-chips` row of `<button type="button">` rendered only when `!isLoading`; click → `onExampleSelect?.(q) ?? onQueryChange(q)`.
- [ ] Run vitest → pass.

### Task 2: App wiring + run-on-click (TDD)

**Files:** `web/src/App.tsx`, `web/src/components/__tests__/App.test.tsx`

- [ ] Write failing test: clicking an example chip calls `streamAgent` and renders the result with the matching `intent-*` layout class; add an undefined-intent coverage test (no `.intent-badge`, no `intent-*` class).
- [ ] Run vitest → fail.
- [ ] Implement: widen `handleSubmit` to `(eventOrQuery?: FormEvent | string)` using the explicit string when provided; pass `onExampleSelect={(q) => { setQuery(q); handleSubmit(q); }}` to `SearchComposer`.
- [ ] Run vitest → pass.

### Task 3: Styles + verify

**Files:** `web/src/styles.css`

- [ ] Add `.example-chips` (flex row, gap, wrap) and `.example-chip` (pill button) styles matching the existing visual language.
- [ ] Run `npm run typecheck` → clean.
- [ ] Run full vitest suite → all pass.
- [ ] Commit.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
