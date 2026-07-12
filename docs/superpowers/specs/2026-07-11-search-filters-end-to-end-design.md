# Access filters end-to-end through search_tool + SearchAgentLoop — Design

Date: 2026-07-11
Status: Approved
Branch/PR: feat/search-filters-end-to-end

## Problem

PR #405 closed the SEARCH-route access-filter leak with a **stopgap**: when
per-user `filters` are present, route through the filter-aware
`_auto_search_pipeline` instead of the unfiltered direct-retrieval / SearchAgentLoop
paths. That is correct but costs authenticated multi-tenant users the direct-first
fast path and the local-model loop.

This change does the **proper** fix: thread `filters` all the way into the
internal-corpus retrieval those paths use, so they enforce access control
themselves and no longer need to be bypassed. Supersedes #405's guard.

## Key finding

`SearchClient.retrieve` / `retrieve_one` (`src/context/retrieval/client.py`)
**already** accept `filters` and put them in the `/retrieve` request body. The gap
is only that the callers never pass them down. So the change is additive
plumbing, not new retrieval logic.

## Approach

Thread an optional `filters: dict | None = None` (internal corpus only; web
providers have no ACL metadata and never receive it):

1. **Tools** (`src/tools/search.py`): `retrieval_search(..., filters=None)` →
   `client.retrieve_one(..., filters=filters)`. `search_tool(..., filters=None)`
   forwards to `retrieval_search` for the `retrieval` provider only.
2. **Loop** (`src/agents/search/search.py`): add `SearchAgentLoopConfig.filters`;
   `_retrieve_many` passes `config.filters` to the vector-DB client and `None` to
   the web client.
3. **Web** (`src/internal/servers/web/app.py`):
   - `_run_direct_search(..., filters=None)` → `search_tool(..., filters=filters)`.
   - `_run_search_agent(..., filters=None)` → `SearchAgentLoopConfig(filters=...)`.
   - `_run_search_direct_or_escalate` passes `filters` to both the direct-search
     and the agent-escalation calls. The explicit `mode=search_tool` /
     `mode=search_agent` sites also pass `filters`.
   - **Remove #405's guard** — the direct/loop paths are now filtered, so the
     special-case bypass is unnecessary. (This branch is cut from `main`, which
     never had the guard; the interaction note is for whoever merges both.)

Backends that don't implement metadata filtering (the demo TF-IDF `/retrieve`)
simply ignore the extra `filters` field (pydantic tolerates it) — no leak,
because the demo is a single unfiltered corpus. Enforcement is the backend's job
(the full `RetrievalService` honors filters); the web tier's job is to *send*
them, which it now does on every SEARCH sub-path.

## Non-goals

- No change to `SearchClient` (already supports filters).
- No new filtering in the demo server (single corpus; nothing to enforce).
- No web-provider filtering (no ACL metadata exists there).

## Success criteria

- `search_tool(provider="retrieval", filters=F)` forwards `F`; `provider="google"`
  never receives filters.
- `SearchAgentLoop._retrieve_many` passes `config.filters` to the vector-DB
  retriever and `None` to the web retriever.
- `_run_direct_search(filters=F)` and `_run_search_agent(filters=F)` propagate `F`.
- Existing web/loop tests stay green (test doubles updated for the new kwarg).
- `ruff` clean; full `pytest tests/unit` green.

## Relationship to PR #405

Supersedes the #405 stopgap. Recommend closing #405 in favor of this PR (both fix
the same leak; merging both would leave the guard shadowing this plumbing).
