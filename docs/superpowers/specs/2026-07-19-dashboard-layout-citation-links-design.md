# Dashboard layout reorg + citation-link fix

**Date:** 2026-07-19
**Status:** Approved

## Problem

Two issues in the main search UI (`web/src/App.tsx` and friends):

1. **Layout.** After submitting a query, the answer is pushed far down the page.
   `AdminOverview` and `AnalyticsDashboard` render unconditionally once their data
   loads (`App.tsx:368-375`) and sit *between* the query box and the actual
   results (`results-layout`). The user wants results to appear immediately below
   the query.

2. **Citation links don't work in search mode.** The search agent emits citation
   labels in `[RxQyDz]` format (e.g. `[R1Q1D1]`, see `src/context/search.py:14`).
   But:
   - `AnswerPanel.tsx:16` only linkifies `[D\d+]`, so `[R1Q1D1]` markers never
     become links.
   - `_search_agent_documents` (`app.py:597`) re-indexes source cards to
     `[D1]…[Dn]`, so even the anchor targets (`source-[D1]`) would not match the
     `[R1Q1D1]` markers in the answer text.

   Result: search-mode citations are doubly broken — wrong regex *and* mismatched
   anchor ids. Chat/RAG mode uses `[D1]` on both sides and works today.

## Design

### Part 1 — Layout: results follow the query

In `App.tsx`, move the two always-on dashboard blocks — `AdminOverview` and
`AnalyticsDashboard` — from above `results-layout` to below it. New top-to-bottom
order:

```
topbar → SearchComposer (query) → error banner → [dev console]
→ results-layout (Answer / Sources / Session)   ← results now sit under the query
→ toggled panels (Connectors / Tools / History)
→ AdminOverview + AnalyticsDashboard             ← moved to the bottom
```

Pure JSX reorder. No logic or data-flow change. Update any ordering assertion in
`web/src/components/__tests__/App.test.tsx`.

### Part 2 — Fix citation links (search mode)

Make the answer's citation markers and the source-card anchor ids use the **same**
real `[RxQyDz]` label.

1. **Backend — `_search_agent_documents` (`app.py:597`).**
   Rewrite to iterate `output.context.rounds` (not the flattened `turns`) and
   assign each `ContextDocument` `id = "R{r}Q{q}D{d}"` using the *same* 1-based
   enumeration as `SearchAgentLoop._format_round_information` (round via
   `enumerate(rounds, 1)`, query via `enumerate(round_ctxs, 1)`, doc via
   `enumerate(ctx.results, 1)`). Then `document.citation` = `[R1Q1D1]`, exactly
   what the answer cites.

   **Dedup:** drop `_dedupe_documents` on this path so every cited label owns a
   card and no citation dangles. Tradeoff: if the model retrieves the same doc
   under two queries, it renders as two cards with distinct labels. This is
   faithful to what the agent saw. (Accepted by user.)

2. **Frontend — `AnswerPanel.tsx`.**
   Broaden `CITATION_RE` from `/(\[D\d+\])/` to `/(\[(?:R\d+Q\d+)?D\d+\])/` so it
   linkifies both `[D1]` (chat/RAG, unchanged) and `[R1Q1D1]` (search). The
   per-part `/^\[D\d+\]$/` test used inside `linkifyCitations` gets the same
   broadening. Anchors become `#source-[R1Q1D1]`, matching the card ids from
   change 1.

## Scope / non-goals

- Chat/RAG mode citation behavior is unchanged (`[D1]` still matches).
- No change to how the search agent generates labels.
- No change to `_reindex_documents` (used by other, non-search paths).

## Verification

- Backend: `_search_agent_documents` produces `ContextDocument`s whose `.citation`
  equals the `[RxQyDz]` labels present in the answer for a multi-round output.
- Frontend: `AnswerPanel` renders `<a href="#source-[R1Q1D1]">` for a `[R1Q1D1]`
  marker and still renders `<a href="#source-[D1]">` for `[D1]`.
- Layout: `App.test.tsx` confirms `results-layout` precedes the admin/analytics
  panels in DOM order.
- `pytest` (relevant web tests) and `npm run typecheck` + `npm test` pass.
