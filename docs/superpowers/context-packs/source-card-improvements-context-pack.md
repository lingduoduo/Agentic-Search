# Generated Context Pack

# Source Card Improvements Design Spec

## Sources

- [Specification: 2026-06-16-source-card-improvements-design.md](../specs/2026-06-16-source-card-improvements-design.md)

## Specification Context

### 2. Architecture

Single-file change: `web/src/components/SourceGrid.tsx`. Each card becomes a small controlled component (`SourceCard`) that owns its own `expanded` and `copied` state. The parent `SourceGrid` remains a thin mapper.

---

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
