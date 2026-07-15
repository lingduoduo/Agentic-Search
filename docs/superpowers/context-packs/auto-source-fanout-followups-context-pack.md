# Generated Context Pack

# Auto Source Fanout Followups

## Sources

- [Specification: 2026-06-21-auto-source-fanout-followups-design.md](../specs/2026-06-21-auto-source-fanout-followups-design.md)
- [Plan: 2026-06-21-auto-source-fanout-followups.md](../plans/2026-06-21-auto-source-fanout-followups.md)

## Specification Context

### Goal

Make the no-LLM test robust to a real key in `.env`; keep the browser provider out of the
`auto` path everywhere (fan-out *and* direct-search); close the named cosmetic/coverage
gaps. Behavior of the shipped feature is unchanged except that `auto` no longer triggers
the browser sidecar.

### Scope

- `src/internal/servers/web/app.py` (`_run_direct_search` sidecar guard)
- `tests/unit/servers/web/test_web_experience_app.py` (test hardening + `all`/browser test)
- `tests/unit/test_execution_fallbacks.py` (stale docstring)

Out of scope: any change to the fan-out merge, status logic, frontend, or SerpAPI latency.

### 1. Harden the no-LLM test

Inject an explicit `app_settings=AppSettings()` (whose `llm.api_key` defaults to `None`)
into `create_web_app`, alongside the existing `load_dotenv` stub and `OPENAI_API_KEY`
delenv. With `resolved.llm.api_key is None` and no `OPENAI_API_KEY` in the environment,
`create_web_app` builds no LLM and the chat path returns 400 — regardless of `.env`.

### Testing

- No-LLM test passes with a real key present in `.env`.
- New test: `auto` direct-search does not call `_run_browser_search`; `all` direct-search
  does (when `browser_search_url` set).
- Full `pytest` + frontend vitest + typecheck remain green.

## Implementation Plan Context

### Risk / rollback

- The only behavior change is `auto` no longer triggering the browser sidecar in
  `_run_direct_search` — a dev/API-only path; the UI fan-out is unaffected. Rollback =
  revert the branch.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
