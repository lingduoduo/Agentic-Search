# Generated Context Pack

# Intent-Adaptive Layout Design Spec

## Sources

- [Specification: 2026-06-16-intent-adaptive-layout-design.md](../specs/2026-06-16-intent-adaptive-layout-design.md)

## Specification Context

### 2. Architecture

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

### 8. Testing Strategy

**File:** `web/src/components/__tests__/App.test.tsx`

- Render `App` with mocked `runAgent` returning `intent: "search"` — assert `.results-layout` has class `intent-search`; assert `.sources-panel` has the highlighted border class in the DOM
- Render with `intent: "chat"` — assert `.results-layout` has class `intent-chat`
- Render with `intent: "tool"` — assert `.results-layout` has class `intent-tool`
- Render with no response (initial state) — assert no `intent-*` class present

CSS layout rules (flex order, min-width) are verified by visual review; Vitest DOM tests verify the class names only.

---

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
