# Generated Context Pack

# Auto Source Fanout Search

## Sources

- [Specification: 2026-06-21-auto-source-fanout-search-design.md](../specs/2026-06-21-auto-source-fanout-search-design.md)
- [Plan: 2026-06-21-auto-source-fanout-search.md](../plans/2026-06-21-auto-source-fanout-search.md)

## Specification Context

### Goal

On a `search` intent, always fan out to internal RAG **and** SerpAPI in parallel, merge
into one ranked list, and remove the Source picker from the normal UI. Unconfigured or
failing providers degrade silently. The picker survives only as a `?dev=1` affordance.

### Scope

- `src/internal/servers/web/app.py` (auto-router fan-out + degradation)
- `web/src/App.tsx`, `web/src/components/SearchComposer.tsx` (remove dropdown from normal UI)
- Tests (backend fan-out + degradation; frontend dropdown gating)

Out of scope: changing the `chat` / `tool` intents; browser in the default path;
per-provider UI configuration; reranker wiring.

## Implementation Plan Context

### Task 1: `auto` provider + default fan-out set

**Files:**
- Modify: `src/internal/servers/web/app.py` (constants ~1242-1271; request model ~163-176; `explicit_source` line 309)
- Test: `tests/unit/test_execution_fallbacks.py`

**Interfaces:**
- Produces: `_DEFAULT_FANOUT_PROVIDERS = ["retrieval", "serpapi"]`; `_source_providers_for("auto") -> ["retrieval", "serpapi"]`; `AgentExperienceRequest.source_provider` default `"auto"`; `explicit_source = source_provider != "auto"`.

- [ ] **Step 1: Write the failing unit test**

In `tests/unit/test_execution_fallbacks.py`, append:

- [ ] **Step 2: Run tests to verify they fail**

…

### Task 2: Concurrent fan-out with graceful degradation + per-provider timeout

**Files:**
- Modify: `src/internal/servers/web/app.py` (`_HybridSearchResult` ~217-220; `_documents_from_search_pages` ~1574; `_run_hybrid_search` multi-provider block ~1486-1553; single-provider returns ~1454, 1481)
- Test: `tests/unit/servers/web/test_web_experience_app.py`

**Interfaces:**
- Consumes: `_source_providers_for` (Task 1), `search_tool`, `_documents_from_search_pages`, `_dedupe_documents`, `_rerank_documents`, `mmr_rerank`, `_reindex_documents`, `_run_browser_search`, `fetch_pages_concurrently`.

…

### Task 3: Auto-router surfaces status as user message

**Files:**
- Modify: `src/internal/servers/web/app.py` (`_run_auto_routed` search branch ~435-465)
- Test: `tests/unit/test_execution_fallbacks.py`

**Interfaces:**
- Consumes: `_HybridSearchResult.status` (Task 2), `_search_only_answer`.
- Produces: search-intent answer text that distinguishes `unreachable` ("No sources are reachable right now…") from `empty`/`ok`.

- [ ] **Step 1: Write failing tests**

In `tests/unit/test_execution_fallbacks.py`, append:

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:. python -m pytest tests/unit/test_execution_fallbacks.py -k "unreachable or empty_uses" -q`

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
