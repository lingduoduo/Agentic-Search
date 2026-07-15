# Auto Source Fan-out Search — Design Spec

**Date:** 2026-06-21
**Status:** Approved

## Problem

The search UI forces the user to pick a **Source** (Local Retrieval / SerpAPI /
Browser). That pushes an internal implementation decision onto the user: where the
information comes from (internal RAG vs the open web) is not something a searcher should
have to know or choose. It also produces confusing failures — selecting an unconfigured
source (e.g. Browser Retrieval with no server) returns a misleading "no results" as if the
search ran and found nothing.

Product principle: **the user searches for information; the system decides where to fetch
it** (internal RAG, online, or both) and the routing is invisible.

## Goal

On a `search` intent, always fan out to internal RAG **and** SerpAPI in parallel, merge
into one ranked list, and remove the Source picker from the normal UI. Unconfigured or
failing providers degrade silently. The picker survives only as a `?dev=1` affordance.

## Scope

- `src/internal/servers/web/app.py` (auto-router fan-out + degradation)
- `web/src/App.tsx`, `web/src/components/SearchComposer.tsx` (remove dropdown from normal UI)
- Tests (backend fan-out + degradation; frontend dropdown gating)

Out of scope: changing the `chat` / `tool` intents; browser in the default path;
per-provider UI configuration; reranker wiring.

## Design

### Core behavior

- Define a default search provider set: **`{retrieval, serpapi}`** (NOT the existing `all`,
  which also includes the slow browser).
- In `_run_auto_routed`, on a `search` decision, search this set via `_run_hybrid_search`,
  which already fans out across providers and dedupes + reranks + MMR-diversifies into one
  ranked list. Replace the single-provider `provider = source_provider` with the default
  set when no explicit provider is given.
- `chat` and `tool` intents are unchanged. `chat` still grounds on internal RAG only.

### Provider selection

- Default path: `retrieval` + `serpapi` only (both fast). Browser (~5–10s/query) is
  excluded from the default and reachable only via the dev affordance.
- The existing `source_provider` request field is retained but defaults to the new set; an
  explicit value (sent only in `?dev=1`) still forces a single provider for testing.

### Graceful degradation

The fan-out must never surface an unconfigured/failing provider as an error or a misleading
empty result:

- **SerpAPI key missing** → skip SerpAPI, return internal-only results (HTTP 200).
- **Internal index empty / retrieval server down** → return SerpAPI-only results.
- **Both unavailable** → return a clear, honest message ("No sources are reachable right
  now"), distinct from "searched and found nothing."
- Provider calls are independent: one raising or timing out never blocks or fails the
  other. The merge proceeds with whatever returned.
- A per-provider timeout (~5s) prevents a hung provider from stalling the whole search.

### UI

- Remove the Source `<select>` from `SearchComposer` in normal mode. Composer becomes:
  query box + example chips + Top K + Search.
- Gate the dropdown behind `?dev=1` (same `showUrlField` / `DEV_MODE` pattern already
  established). In dev, the user can force a single provider to test each in isolation.
- In normal mode the request omits `source_provider` (backend uses the default set).
- Per-result provenance is preserved: each document carries its `source` label, so source
  cards still show internal-vs-web origin without a control.

## Testing

**Backend**
- Search intent fans out to internal + SerpAPI and returns a merged/deduped list (mock both
  providers; assert documents from each appear).
- SerpAPI unconfigured → internal-only, 200, no error.
- Internal down → SerpAPI-only.
- Both down → clear "no sources reachable" message, not a silent empty.
- One provider raising/timing out does not fail the other.
- `chat` intent still grounds internal-only (regression guard).

**Frontend**
- Source dropdown absent in normal mode; present under `?dev=1`.
- Request omits `source_provider` in normal mode.

## Success criteria

1. A normal search returns internal + web results merged, with no source picker in the UI.
2. Missing SerpAPI key or down retrieval server degrades silently; only a total outage
   shows a message.
3. `?dev=1` still exposes single-provider selection for testing.
4. `pytest` + `npm run typecheck` + vitest all green.
