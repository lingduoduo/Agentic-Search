# Generated Context Pack

# Source Authoritative Retrieval Routing

## Sources

- [Specification: 2026-06-21-source-authoritative-retrieval-routing.md](../specs/2026-06-21-source-authoritative-retrieval-routing.md)
- [Plan: 2026-06-21-source-authoritative-retrieval-routing.md](../plans/2026-06-21-source-authoritative-retrieval-routing.md)

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

### Testing

- Backend: explicit `source_provider="serpapi"` + a chat-looking query → `intent="search"`
  and `_run_hybrid_search` called with `source_provider="serpapi"`; classifier not called.
- Backend: default source still routes "explain FAISS" → `intent="chat"`.
- Backend: client `search_url` is ignored; server settings URL is used (SSRF).
- Frontend: URL field hidden by default, shown with `showUrlField`; request omits
  `search_url` in non-dev mode.

### Out of scope

- Redesigning the Source dropdown options (Google PSE remains disabled).
- Per-provider URL configuration UI.
- Any change to explicit-`mode` request paths.

## Implementation Plan Context

### Risk / rollback

- Behavior change is gated on `source_provider != "retrieval"`, so the default path is
  untouched. Rollback = revert the branch; no migrations or data changes.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
