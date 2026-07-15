# SEARCH-route access filters + real search-flow LLM — Design

Date: 2026-07-11
Status: Approved
Branch/PR: fix/search-route-access-filters

Two independently-found web-backend bugs, fixed together (both in the query/search surface).

## W1 — per-user access filters dropped on the SEARCH auto-route (security)

`_run_agent_impl` builds per-user access filters via `build_user_only_filters(...)`
and threads `filters` into `_run_auto_routed` → `_run_search_direct_or_escalate`.
On the SEARCH strategy those filters are applied to only *one* of three
sub-paths — the no-local-model degrade branch (`_auto_search_pipeline(filters=...)`).
The other two ignore them:

- direct-retrieval (`_run_direct_search` → `search_tool`) takes no `filters`;
- escalation (`_run_search_agent` → `SearchAgentLoop`) takes no `filters`
  (`SearchAgentLoopConfig` has no filter field).

So an authenticated/multi-tenant query that routes to SEARCH can retrieve
documents belonging to other users. Worse, *whether* filtering happens flips on
`manager`/`tokenizer` availability: with a local model the SEARCH route escalates
to the **unfiltered** loop; without one it degrades to the **filtered** pipeline —
same query, different access scope. The CHAT and `hybrid_search` paths correctly
pass `filters` to `answer_with_retrieval` / `run_expanded_search`.

### Why not thread filters end-to-end

`search_tool` and the `SearchAgentLoop` accept no `filters`, and the retrieval
HTTP API would need to honor them per-request. Plumbing filters through the tools
layer, the retrieval client, and the loop is a feature-sized change whose
enforcement also depends on the retrieval backend — out of scope for a security
fix that should land now.

### Fix (surgical, correctness-first)

At the top of `_run_search_direct_or_escalate`, when `filters` is truthy, route
through the filter-aware `_auto_search_pipeline(filters=filters)` instead of the
unfiltered direct/loop paths. Closes the leak for every SEARCH sub-path in the
auto-router (the path every bundled-UI query uses).

**Deliberate tradeoff:** authenticated multi-tenant users lose the direct-first
fast path and the local-model loop on the SEARCH route (they always use the
filtered pipeline). Correctness over feature. Full filter plumbing through
`search_tool`/`SearchAgentLoop` is the documented follow-up. Unauthenticated /
single-user setups (`filters is None`) are byte-for-byte unchanged.

**Out of scope (noted):** explicit `mode=search_agent` / `mode=search_tool`
requests bypass this gate. Those are dev/API-only surfaces (the bundled UI sends
no `mode`); a follow-up can extend the guard there.

## W2 — `/search/search-flow-classification` always returns "search"

`search_backend.py` hardcoded a local `_NoOpLLM` whose `complete()` always
returns `"search"`, so with `SEARCH_CLASS == "search"` the classifier returned
`True` for every query ≤200 chars (>200 short-circuits to chat). The endpoint's
output depended only on length, never content — its whole purpose is broken.

### Fix

Build the env-configured LLM (`_build_flow_classifier_llm`, mirroring the
retrieval service's `_build_llm`) and pass it to `classify_is_search_flow`. When
no LLM is configured (no `GEN_AI_API_KEY`/`OPENAI_API_KEY`) return `None` →
default to chat (`is_search_flow=False`) instead of guessing.

## Success criteria

- W1: with `filters` set, `_run_search_direct_or_escalate` calls
  `_auto_search_pipeline` (with the filters) and neither `_run_direct_search` nor
  `_run_search_agent`; with `filters=None` the direct-first path is unchanged.
- W2: a configured LLM makes the result content-driven (search vs chat); no LLM →
  chat. Existing endpoint tests stay green and hermetic (no real network call).
- `ruff` clean; `pytest tests/unit/test_search_route_access_filters.py` +
  `tests/unit/servers/query_and_chat/test_query_and_chat.py` green.
