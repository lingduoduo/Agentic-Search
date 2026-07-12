# Access filters end-to-end — Plan

Spec: [2026-07-11-search-filters-end-to-end-design.md](../specs/2026-07-11-search-filters-end-to-end-design.md)

## Steps

1. **Tools** — `retrieval_search` + `search_tool` gain `filters`, forwarded to the
   client for the `retrieval` provider only (`src/tools/search.py`).
   → verify: unit — retrieval provider forwards filters; google provider omits them.

2. **Loop** — `SearchAgentLoopConfig.filters`; `_retrieve_many` threads it to the
   vector-DB client, `None` to web (`src/agents/search/search.py`).
   → verify: unit — vector gets filters, web gets None.

3. **Web** — `_run_direct_search` + `_run_search_agent` gain `filters`; the
   escalate gate and the explicit search_tool/search_agent modes pass it
   (`src/internal/servers/web/app.py`).
   → verify: unit — direct-search forwards filters; search-agent config carries them.

4. **Fix test doubles** — add the new kwarg to fake `search_tool` /
   `_run_direct_search` / `SearchClient.retrieve` stubs that broke.
   → verify: `pytest tests/unit` green.

5. **Lint + land** — ruff; rebase `--onto origin/main`; push; PR noting it
   supersedes #405.
