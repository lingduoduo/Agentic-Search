# Intent Example-Query Chips Implementation Plan

**Goal:** Add clickable example-query chips (one per intent) under the search box that populate and run a query in one click, plus broaden intent-component test coverage.

**Architecture:** Frontend-only. `SearchComposer` renders the chips and exposes an `onExampleSelect` callback; `App` wires it to a stale-state-safe `handleSubmit` that accepts an optional explicit query.

**Tech Stack:** React 19 + TypeScript + Vitest + React Testing Library.

## Global Constraints

- No backend changes.
- Existing form-submit and Cmd+Enter paths must be unchanged (still use state `query`).
- Chips hidden while `isLoading`.

---

## Task 1: SearchComposer chips (TDD)

**Files:** `web/src/components/SearchComposer.tsx`, `web/src/components/__tests__/SearchComposer.test.tsx`

- [ ] Write failing tests: 3 chips render; clicking a chip calls `onExampleSelect` with the example query; no chips while `isLoading`.
- [ ] Run vitest → fail.
- [ ] Implement: `EXAMPLE_QUERIES` constant (search/chat/tool); optional `onExampleSelect?: (q: string) => void` prop; `.example-chips` row of `<button type="button">` rendered only when `!isLoading`; click → `onExampleSelect?.(q) ?? onQueryChange(q)`.
- [ ] Run vitest → pass.

## Task 2: App wiring + run-on-click (TDD)

**Files:** `web/src/App.tsx`, `web/src/components/__tests__/App.test.tsx`

- [ ] Write failing test: clicking an example chip calls `streamAgent` and renders the result with the matching `intent-*` layout class; add an undefined-intent coverage test (no `.intent-badge`, no `intent-*` class).
- [ ] Run vitest → fail.
- [ ] Implement: widen `handleSubmit` to `(eventOrQuery?: FormEvent | string)` using the explicit string when provided; pass `onExampleSelect={(q) => { setQuery(q); handleSubmit(q); }}` to `SearchComposer`.
- [ ] Run vitest → pass.

## Task 3: Styles + verify

**Files:** `web/src/styles.css`

- [ ] Add `.example-chips` (flex row, gap, wrap) and `.example-chip` (pill button) styles matching the existing visual language.
- [ ] Run `npm run typecheck` → clean.
- [ ] Run full vitest suite → all pass.
- [ ] Commit.
