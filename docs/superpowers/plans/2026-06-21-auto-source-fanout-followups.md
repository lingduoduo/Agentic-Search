# Auto Source Fan-out Follow-ups — Implementation Plan

Spec: `docs/superpowers/specs/2026-06-21-auto-source-fanout-followups-design.md`
Branch: `feat/auto-source-fanout-followups`

## Steps

1. **Exclude `auto` from the browser sidecar** → in `src/internal/servers/web/app.py`
   `_run_direct_search`, change `source_provider not in {"browser", "all"}` to
   `source_provider not in {"browser", "all", "auto"}`.
   Verify: new test asserts `_run_direct_search(source_provider="auto")` does not call
   `_run_browser_search`; `source_provider="all"` does (when `browser_search_url` set).

2. **Harden the no-LLM test** → in `tests/unit/servers/web/test_web_experience_app.py`,
   import `AppSettings` and pass `app_settings=AppSettings()` into the `create_web_app` call
   in `test_agent_no_llm_no_model_returns_400`, keeping the `load_dotenv` stub + delenv.
   Verify: the test passes even though `.env` contains a real `OPENAI_API_KEY`.

3. **Add `_run_direct_search` browser-leg coverage test** → new test mocking `search_tool`
   and `_run_browser_search`, asserting the `auto` vs `all` sidecar behavior from Step 1.

4. **Fix stale docstring** → `tests/unit/test_execution_fallbacks.py`
   `test_default_source_still_auto_routes_to_chat` docstring `('retrieval')` → `('auto')`.

5. **Verify + lint** → `PYTHONPATH=src:. python -m pytest tests/unit -q` green;
   `ruff check . --fix && ruff format .` clean; `cd web && npm run typecheck` clean
   (no frontend changes, sanity only).

6. **Push + open PR** with spec + plan on the branch.

## Risk / rollback

- The only behavior change is `auto` no longer triggering the browser sidecar in
  `_run_direct_search` — a dev/API-only path; the UI fan-out is unaffected. Rollback =
  revert the branch.
