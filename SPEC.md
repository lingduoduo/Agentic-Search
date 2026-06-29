# Spec: Answer-grounding visibility

## Objective

Two related problems made the UI misrepresent *whether and why* an answer was
grounded:

1. **Status pill** showed "Grounded" for any answer, including parametric
   answers with **0 citations** (e.g. a bare `direct_llm` route).
2. **No route visibility** — users couldn't see *which strategy* answered
   (`direct_llm` / `agentic_rag` / `search_agent` / `tool_agent`) or whether it
   **degraded**, so an ungrounded answer looked unexplained.

**Make grounding legible:** the status reflects actual grounding, and the UI
surfaces the chosen route (and any degradation).

User: anyone using the search UI on `:7860`. Success: an answer with citations
reads "Grounded", without reads "Answered"; loading reads "Searching"; errors
"Needs attention"; idle "Ready". When an answer arrives, a route chip shows the
strategy (e.g. "via direct_llm"), with a degraded marker when one fired.

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
- [x] SSE `done` event carries `route` and `route_degraded` (from
      `result.hook_metadata`); the non-stream response already exposes them.
- [x] When an answer arrives, the UI shows a route chip ("via <route>"), with a
      degraded marker when `route_degraded` is set; cleared on new query/session.
- [x] Tests pin status + route rendering and the `done`-event field; frontend
      suite + typecheck + backend SSE tests green.
- [x] No response-contract break (additive `done`-event fields only).

## Open Questions

- None open. Route is surfaced read-only; no per-route UI actions are in scope.
