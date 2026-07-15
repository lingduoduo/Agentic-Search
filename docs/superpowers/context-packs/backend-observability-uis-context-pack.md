# Generated Context Pack

# Backend Observability Uis

## Sources

- [Specification: 2026-06-29-backend-observability-uis-design.md](../specs/2026-06-29-backend-observability-uis-design.md)
- [Plan: 2026-06-29-backend-observability-uis.md](../plans/2026-06-29-backend-observability-uis.md)

## Specification Context

### 2. Features & Acceptance Criteria

A top-level nav toggles between the existing **Search** view and a new **Console** view. Console hosts four panels.

### 8. Open Questions (resolve during planning)

1. **Worker health source:** does `monitoring_worker` already persist snapshots to `AgenticSearchStore`, or do we add that write? (Affects F2 size.)
2. **Retrieval base URL:** lock to the backend-configured `search_url`, or let the Lab target an arbitrary host (handy for comparing demo vs server.py)? Arbitrary host = small SSRF surface, acceptable dev-only but worth a note.
3. **Nav placement:** simple top toggle vs. a left rail — pick the lower-footprint option that fits current `App.tsx`.

## Implementation Plan Context

### Resolved open questions (from spec §8)

Carried as decisions to confirm at the top of Phase 1/3:
1. **Worker health source** → T3.1 adds persistence if absent (assumed absent until checked).
2. **Retrieval base URL** → default to configured `search_url`; allow arbitrary host as an explicit dev-only input, with a code comment noting the (accepted, dev-only) SSRF surface.
3. **Nav placement** → simple top toggle (`ConsoleNav`), lowest footprint in current `App.tsx`.

If any of these three should flip, say so before Phase 1.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
