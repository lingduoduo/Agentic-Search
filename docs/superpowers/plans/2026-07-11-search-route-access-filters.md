# SEARCH-route access filters + real search-flow LLM — Plan

Spec: [2026-07-11-search-route-access-filters-design.md](../specs/2026-07-11-search-route-access-filters-design.md)

## Steps

1. **W1 guard** — in `_run_search_direct_or_escalate` (`src/internal/servers/web/app.py`),
   short-circuit to `_auto_search_pipeline(filters=filters, ...)` when `filters` is
   truthy, before the direct-first/escalation logic.
   → verify: unit test asserts filters-present → pipeline (with filters), and
   `_run_direct_search`/`_run_search_agent` not called; filters=None → direct path.

2. **W2 real LLM** — in `src/internal/servers/query_and_chat/search_backend.py`,
   add `_build_flow_classifier_llm()` (env LLM or None) and use it in
   `/search/search-flow-classification`; default to chat when None. Drop the
   hardcoded `_NoOpLLM`.
   → verify: unit tests — configured LLM drives content-based result; no LLM → chat;
   existing endpoint tests stay green and hermetic.

3. **Lint + tests** → `ruff check --fix && ruff format`;
   `pytest tests/unit/test_search_route_access_filters.py tests/unit/servers/query_and_chat/test_query_and_chat.py -q` green.

4. **Land** — commit, rebase `--onto origin/main` (drop the unrelated local-main
   commit), push, open PR with a specific title.
