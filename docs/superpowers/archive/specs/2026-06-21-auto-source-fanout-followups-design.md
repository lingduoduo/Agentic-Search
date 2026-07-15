# Auto Source Fan-out — Follow-ups Design Spec

**Date:** 2026-06-21
**Status:** Approved

## Problem

PR #314 (auto source fan-out) merged with three deferred follow-ups recorded during
subagent-driven execution and the final whole-branch review:

1. **`test_agent_no_llm_no_model_returns_400` is environment-fragile.** It stubs the web
   app's `load_dotenv` but not the config loader (`load_app_settings`), which independently
   reads the LLM key. With a real `OPENAI_API_KEY`/`GEN_AI_API_KEY` in `.env`, the "no LLM"
   path builds an LLM and returns 200 instead of 400. (Verified failing on `main` without
   any new code.)

2. **Browser sidecar fires for `auto` in the direct-search path.** `_run_direct_search`
   adds a browser sidecar when `source_provider not in {"browser", "all"}` — which is true
   for `"auto"`. An explicit `mode="search_tool"` API call with the default `auto` provider
   therefore pulls in the slow browser, contradicting "browser is out of the default
   fan-out." (Not reachable from the UI, which never sets `mode`.)

3. **Minor coverage/cosmetic gaps:** a stale docstring referencing the old `"retrieval"`
   default, and no test exercising the `"all"` provider's browser leg in the fan-out.

## Goal

Make the no-LLM test robust to a real key in `.env`; keep the browser provider out of the
`auto` path everywhere (fan-out *and* direct-search); close the named cosmetic/coverage
gaps. Behavior of the shipped feature is unchanged except that `auto` no longer triggers
the browser sidecar.

## Scope

- `src/internal/servers/web/app.py` (`_run_direct_search` sidecar guard)
- `tests/unit/servers/web/test_web_experience_app.py` (test hardening + `all`/browser test)
- `tests/unit/test_execution_fallbacks.py` (stale docstring)

Out of scope: any change to the fan-out merge, status logic, frontend, or SerpAPI latency.

## Design

### 1. Harden the no-LLM test

Inject an explicit `app_settings=AppSettings()` (whose `llm.api_key` defaults to `None`)
into `create_web_app`, alongside the existing `load_dotenv` stub and `OPENAI_API_KEY`
delenv. With `resolved.llm.api_key is None` and no `OPENAI_API_KEY` in the environment,
`create_web_app` builds no LLM and the chat path returns 400 — regardless of `.env`.

### 2. Exclude `auto` from the browser sidecar

In `_run_direct_search`, change the sidecar guard from
`source_provider not in {"browser", "all"}` to
`source_provider not in {"browser", "all", "auto"}`. `auto` expands to
`["retrieval", "serpapi"]` and must not pull browser. (`_run_hybrid_search` — the actual
UI fan-out path — already never adds a browser sidecar; this aligns the direct-search path.)

### 3. Minor cleanups

- Fix the stale docstring in `test_default_source_still_auto_routes_to_chat`
  (`"Default source ('retrieval')"` → `"(auto)"`).
- Add a test that `_run_direct_search(source_provider="all")` includes the browser leg when
  `browser_search_url` is set (closes the named coverage gap), and that
  `source_provider="auto"` does **not** invoke the browser sidecar.

## Testing

- No-LLM test passes with a real key present in `.env`.
- New test: `auto` direct-search does not call `_run_browser_search`; `all` direct-search
  does (when `browser_search_url` set).
- Full `pytest` + frontend vitest + typecheck remain green.

## Success criteria

1. `test_agent_no_llm_no_model_returns_400` passes regardless of `.env` LLM keys.
2. `auto` never triggers the browser provider in either search path.
3. No behavior change to the merged feature beyond the sidecar exclusion.
