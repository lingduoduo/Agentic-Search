# Generated Context Pack

# Intent-Adaptive Layout Design Spec

## Sources

- [Specification: 2026-06-16-intent-adaptive-layout-design.md](../specs/2026-06-16-intent-adaptive-layout-design.md)

## Specification Context

### 2. Architecture

The `intent-*` class already lives on `.results-layout` in `App.tsx`. All layout changes are CSS-only targeting that class. One minimal JS change: add `session-panel` class to the Session `<section>` so it can be targeted independently from other `.panel` elements.

---

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
