# Spec: Grounding-accurate answer status

## Objective

The web UI's status pill showed **"Grounded"** whenever an answer existed —
even for parametric answers with **0 citations** (e.g. a bare `direct_llm`
route). That misrepresents whether the answer is backed by retrieved sources.
**Make the status reflect actual grounding** so users can trust the label.

User: anyone using the search UI on `:7860`. Success: an answer with citations
reads "Grounded"; an answer with no citations reads "Answered"; in-flight reads
"Searching"; errors read "Needs attention"; idle reads "Ready".

## Tech Stack

React 19 + Vite + TypeScript (frontend, `web/`). Vitest + Testing Library.

## Commands

```bash
cd web
npm run dev          # Vite dev server (proxies /api → :7860)
npm run typecheck    # tsc -b
npm run test:unit    # vitest run
npm test             # typecheck + unit
```

## Project Structure

```
web/src/App.tsx                          → app shell; the status pill lives here
web/src/components/                       → presentational components
web/src/components/__tests__/App.test.tsx → App-level behavior tests
```

## Code Style

Derived state via `useMemo` with an explicit dependency list; comments explain
*why*, not *what*:

```tsx
const status = useMemo(() => {
  if (isLoading) return "Searching";
  if (error) return "Needs attention";
  // "Grounded" only when the answer is backed by retrieved sources; an
  // answer with no citations is parametric, so call it "Answered".
  if (answer) return citations.length > 0 ? "Grounded" : "Answered";
  return "Ready";
}, [answer, citations, error, isLoading]);
```

## Testing Strategy

Vitest + Testing Library in `web/src/**/__tests__`. Mock `streamAgent`; drive a
fake SSE stream and assert on rendered DOM (`.status-pill` text). Two cases:
citations present → "Grounded"; citations empty → "Answered".

## Boundaries

- **Always:** `npm test` (typecheck + unit) green before commit; keep the
  `(answer, citations, documents, intent, …)` response contract untouched.
- **Ask first:** changing the status vocabulary, adding backend fields, or
  touching the SSE event shape.
- **Never:** alter routing/dispatch logic or the answer text; this is a
  presentation-only change.

## Success Criteria

- [x] `citations.length > 0` → "Grounded"; `=== 0` → "Answered".
- [x] Loading/error/idle states unchanged.
- [x] New tests pin both cases; full frontend suite + typecheck green.
- [x] No backend or contract change.

## Open Questions

- Follow-up (separate PR): surface the chosen `route` / `route_degraded` in the
  UI so users see *why* an answer was or wasn't grounded. Needs the SSE `done`
  event to carry `route` (currently only `intent`).
