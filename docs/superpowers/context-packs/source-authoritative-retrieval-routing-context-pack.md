# Generated Context Pack

# Source Authoritative Retrieval Routing

## Sources

- [Specification: 2026-06-21-source-authoritative-retrieval-routing.md](../archive/specs/2026-06-21-source-authoritative-retrieval-routing.md)
- [Plan: 2026-06-21-source-authoritative-retrieval-routing.md](../archive/plans/2026-06-21-source-authoritative-retrieval-routing.md)

## Specification Context

### Goal

Make the **Source selection authoritative**, resolve the retrieval URL **server-side**,
and remove the raw URL box from the normal UI (keep it for local dev only).

### Scope

- `src/internal/servers/web/app.py`
- `web/src/App.tsx`, `web/src/components/SearchComposer.tsx`
- Tests: `tests/unit/test_execution_fallbacks.py`,
  `tests/unit/servers/web/test_web_experience_app.py`,
  `web/src/components/__tests__/SearchComposer.test.tsx`,
  `web/src/components/__tests__/App.test.tsx`

### Decision: what counts as "explicit"?

Any provider other than the default `retrieval` is treated as an explicit search command
(including `all`). The default `retrieval` keeps auto-routing so "explain X"-style queries
still go to chat — preserving the intent-routing feature for the common case.

## Implementation Plan Context

### Risk / rollback

- Behavior change is gated on `source_provider != "retrieval"`, so the default path is
  untouched. Rollback = revert the branch; no migrations or data changes.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
