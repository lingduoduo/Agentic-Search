# Intent-Adaptive Layout Design Spec

**Date:** 2026-06-16
**Status:** Draft

---

## 1. Goals & Success Criteria

### Problem

`App.tsx` already sets an `intent-{search,chat,tool}` CSS class on `.results-layout` after each response, but no CSS rules consume it. All three intents render the same single-column stack: Answer → Sources → Session, regardless of which panel is most relevant for the current mode.

### Success Criteria

- **`intent-search`**: Sources panel is visually prominent (highlighted border, expanded height); session panel is de-emphasized
- **`intent-chat`**: Answer and Session panels appear side-by-side on wide screens; Sources panel spans full width below both
- **`intent-tool`**: Tool trace panel (Spec 3) spans full width at the top; Sources and Session are secondary
- Falls back gracefully to the current single-column layout when `intent` is undefined (no class applied)
- No new component logic — changes are CSS-only, except one class addition in `App.tsx`

---

## 2. Architecture

The `intent-*` class already lives on `.results-layout` in `App.tsx`. All layout changes are CSS-only targeting that class. One minimal JS change: add `session-panel` class to the Session `<section>` so it can be targeted independently from other `.panel` elements.

```
App.tsx  (1 line change)
  └── <section className="panel session-panel" ...>   ← add "session-panel"

styles.css  (new rules)
  ├── .intent-search  — sources border highlight + min-height
  ├── .intent-chat    — flex wrap, answer + session side-by-side, sources full-width
  └── .intent-tool    — tool-trace-panel full-width, elevated
```

---

## 3. `App.tsx` — add `session-panel` class

**File:** `web/src/App.tsx`

One character change on the Session section:

```tsx
// Before
<section className="panel" aria-label="Session">

// After
<section className="panel session-panel" aria-label="Session">
```

This gives the CSS rules a stable hook to target the session panel independently from other `.panel` elements (e.g. `ToolCallTracePanel`).

---

## 4. CSS Rules

**File:** `web/src/styles.css`

Append after the existing `.results-layout` block:

```css
/* ── Intent-adaptive layout ──────────────────────────────────────────────────── */

/* Search: sources panel gets visual prominence */
.intent-search .sources-panel {
  border: 2px solid var(--accent, #38bdf8);
  border-radius: 8px;
}
.intent-search .session-panel {
  opacity: 0.7;
}

/* Chat: answer + session side-by-side; sources full-width below */
.intent-chat.results-layout {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  align-items: flex-start;
}
.intent-chat .answer-column {
  flex: 1 1 calc(50% - 8px);
  min-width: 280px;
}
.intent-chat .session-panel {
  flex: 1 1 calc(50% - 8px);
  min-width: 280px;
  order: 1;
}
.intent-chat .sources-panel {
  flex: 1 1 100%;
  order: 2;
}

/* Tool: trace panel spans full width, sits above sources + session */
.intent-tool.results-layout {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  align-items: flex-start;
}
.intent-tool .answer-column {
  flex: 1 1 100%;
}
.intent-tool .tool-trace-panel {
  flex: 1 1 100%;
  order: 1;
}
.intent-tool .sources-panel {
  flex: 1 1 calc(50% - 8px);
  min-width: 240px;
  order: 2;
}
.intent-tool .session-panel {
  flex: 1 1 calc(50% - 8px);
  min-width: 240px;
  order: 2;
  opacity: 0.7;
}

/* Narrow screens: stack everything single-column for all intents */
@media (max-width: 720px) {
  .intent-chat.results-layout,
  .intent-tool.results-layout {
    display: grid;
  }
  .intent-chat .answer-column,
  .intent-chat .session-panel,
  .intent-chat .sources-panel,
  .intent-tool .tool-trace-panel,
  .intent-tool .sources-panel,
  .intent-tool .session-panel {
    flex: unset;
    order: unset;
    min-width: unset;
  }
}
```

---

## 5. Layout Reference

### `intent-search` (single column, sources highlighted)

```
┌─────────────────────────────────────┐
│ Answer panel                        │
├─────────────────────────────────────┤
│ Sources panel  ← blue border        │  ← prominent
├─────────────────────────────────────┤
│ Session panel  ← dimmed (70%)       │  ← de-emphasized
└─────────────────────────────────────┘
```

### `intent-chat` (side-by-side on wide screens)

```
┌──────────────────┬──────────────────┐
│ Answer           │ Session (bubbles)│  ← side-by-side
├──────────────────┴──────────────────┤
│ Sources (full width)                │  ← secondary
└─────────────────────────────────────┘
```

### `intent-tool` (trace panel at full width)

```
┌─────────────────────────────────────┐
│ Answer panel                        │
├─────────────────────────────────────┤
│ Tool Call Trace (full width)        │  ← hero
├──────────────────┬──────────────────┤
│ Sources          │ Session (dimmed) │  ← secondary
└──────────────────┴──────────────────┘
```

---

## 6. Interaction with Other Specs

| Spec | Dependency |
|---|---|
| Spec 1 (Streaming) | `intent` is now set from SSE `done` event — layout shifts after streaming completes. No conflict. |
| Spec 3 (Tool Trace) | `ToolCallTracePanel` uses `className="panel tool-trace-panel"`. The `.tool-trace-panel` CSS hook in this spec targets that class directly. Both specs must be implemented for `intent-tool` layout to work fully. |
| Specs 2 + 4 | No interaction — `AnswerPanel` and `SourceGrid` internals are independent of layout. |

---

## 7. Error Handling

| Scenario | Behavior |
|---|---|
| `intent` is `undefined` (first load, error state) | No `intent-*` class on container → current single-column grid unchanged |
| Spec 3 not yet implemented (`ToolCallTracePanel` absent) | `intent-tool` rules still apply to Answer, Sources, Session; no crash |
| Screen width ≤ 720 px | `@media` block reverts flex to grid; single-column stack for all intents |
| Browser doesn't support `flex` `order` | Falls back to DOM order (Answer → Sources/Session) — still usable |

---

## 8. Testing Strategy

**File:** `web/src/components/__tests__/App.test.tsx`

- Render `App` with mocked `runAgent` returning `intent: "search"` — assert `.results-layout` has class `intent-search`; assert `.sources-panel` has the highlighted border class in the DOM
- Render with `intent: "chat"` — assert `.results-layout` has class `intent-chat`
- Render with `intent: "tool"` — assert `.results-layout` has class `intent-tool`
- Render with no response (initial state) — assert no `intent-*` class present

CSS layout rules (flex order, min-width) are verified by visual review; Vitest DOM tests verify the class names only.

---

## 9. File Map

| Action | Path | Responsibility |
|---|---|---|
| **Modify** | `web/src/App.tsx` | Add `session-panel` class to Session `<section>` |
| **Modify** | `web/src/styles.css` | Add `intent-search`, `intent-chat`, `intent-tool` layout rules + `@media` fallback |
| **Modify** | `web/src/components/__tests__/App.test.tsx` | Assert correct `intent-*` class applied per response intent |

**Not changed:** All backend files, `AnswerPanel`, `SourceGrid`, `SessionTimeline`, `ToolCallTracePanel`, `SearchComposer`.
