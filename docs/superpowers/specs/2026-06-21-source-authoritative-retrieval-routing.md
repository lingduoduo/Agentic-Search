# Source-Authoritative Retrieval Routing — Design Spec

**Date:** 2026-06-21
**Status:** Approved

## Problem

The intent-routed search UI exposes a **Source** dropdown (Local Retrieval / SerpAPI /
Browser / All) and a free-text **Retrieval URL** box. Three defects make these controls
confusing and unsafe:

1. **The Source dropdown is a no-op in the default flow.** The frontend never sends an
   explicit `mode`, so every search goes through `_run_auto_routed`, which (a) was never
   passed `source_provider`, (b) decides search-vs-chat purely from query wording, and
   (c) hardcodes `provider = "retrieval"` even when it does search. Selecting SerpAPI and
   asking "explain how FAISS works" gets answered from the model and never touches SerpAPI.

2. **The raw Retrieval URL box is a footgun.** It is plain UI state seeded from a default
   constant. A stale default (port 8000 after the server moved to 8001) silently routed
   every request to a dead port → "Search returned no results". A user should not need to
   know an internal `host:port`.

3. **The client-supplied URL is an SSRF vector.** The backend fetched whatever
   `search_url` the client sent (e.g. cloud metadata endpoints / internal services).

Additionally, for `serpapi`/`google`/`serper` the `search_url` value is ignored entirely
(those providers call fixed external endpoints with API keys), so the box is meaningless
for any non-`retrieval` source.

## Goal

Make the **Source selection authoritative**, resolve the retrieval URL **server-side**,
and remove the raw URL box from the normal UI (keep it for local dev only).

## Scope

- `src/internal/servers/web/app.py`
- `web/src/App.tsx`, `web/src/components/SearchComposer.tsx`
- Tests: `tests/unit/test_execution_fallbacks.py`,
  `tests/unit/servers/web/test_web_experience_app.py`,
  `web/src/components/__tests__/SearchComposer.test.tsx`,
  `web/src/components/__tests__/App.test.tsx`

## Design

### Backend

- `_run_auto_routed` gains `source_provider: str = "retrieval"`. Define
  `explicit_source = source_provider != "retrieval"`.
  - When `explicit_source`: **skip the tool loop (Tier 1) and the classifier**, set
    `is_search = True`, and search **that** provider. (`provider = source_provider`,
    replacing the hardcoded `"retrieval"`.)
  - When the default `retrieval` source: behavior is unchanged (full three-tier
    auto-routing preserved).
- The call site passes `source_provider=_normalize_source_provider(request.source_provider)`.
- **URL resolution / SSRF:** the handler uses `settings.search_url` and ignores a
  client-supplied `request.search_url` unless `settings.allow_client_search_url` is true.
  That flag is populated from `AGENTIC_SEARCH_ALLOW_CLIENT_RETRIEVAL_URL` (dev only).
- The stale `SearchExperienceSettings.search_url` default is corrected to port 8001.

### Frontend

- `SearchComposer` gains `showUrlField?: boolean` (default `false`); the Retrieval URL
  `<label>` renders only when true.
- `App` computes `DEV_MODE` from `?dev=1`, passes `showUrlField={DEV_MODE}`, and only
  includes `search_url` in the request when `DEV_MODE` (otherwise omitted → backend
  resolves it).

### Decision: what counts as "explicit"?

Any provider other than the default `retrieval` is treated as an explicit search command
(including `all`). The default `retrieval` keeps auto-routing so "explain X"-style queries
still go to chat — preserving the intent-routing feature for the common case.

## Testing

- Backend: explicit `source_provider="serpapi"` + a chat-looking query → `intent="search"`
  and `_run_hybrid_search` called with `source_provider="serpapi"`; classifier not called.
- Backend: default source still routes "explain FAISS" → `intent="chat"`.
- Backend: client `search_url` is ignored; server settings URL is used (SSRF).
- Frontend: URL field hidden by default, shown with `showUrlField`; request omits
  `search_url` in non-dev mode.

## Out of scope

- Redesigning the Source dropdown options (Google PSE remains disabled).
- Per-provider URL configuration UI.
- Any change to explicit-`mode` request paths.

## Success criteria

1. Selecting SerpAPI searches SerpAPI regardless of query wording.
2. No raw Retrieval URL box in normal UI; `?dev=1` reveals it.
3. Client-supplied `search_url` ignored unless the dev env flag is set.
4. `pytest` + `npm run typecheck` + vitest all green.
