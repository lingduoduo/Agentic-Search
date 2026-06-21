# Plan — Source-Authoritative Retrieval Routing

Spec: `docs/superpowers/specs/2026-06-21-source-authoritative-retrieval-routing.md`

## Steps

1. **Settings: fix default port + add SSRF dev flag** → verify: `SearchExperienceSettings`
   default `search_url` is 8001; `allow_client_search_url` parsed from
   `AGENTIC_SEARCH_ALLOW_CLIENT_RETRIEVAL_URL`.

2. **Handler: resolve URL server-side** → verify: client `search_url` ignored unless
   `allow_client_search_url`; updated `test_agent_endpoint_runs_pipeline_and_persists_chat`
   asserts the server URL is used.

3. **Auto-router: thread `source_provider`, force search on explicit source, use chosen
   provider** → verify: new test `test_explicit_source_forces_search_against_that_provider`
   passes; classifier not called for explicit source.

4. **Preserve default behavior** → verify: `test_default_source_still_auto_routes_to_chat`
   and existing fallback tests still pass.

5. **Frontend: `showUrlField` prop + dev-gated URL box + omit `search_url` in normal mode**
   → verify: `npm run typecheck` clean; SearchComposer tests (hidden by default / shown in
   dev) and App test (no `search_url` in non-dev) pass.

6. **Full suite + lint** → verify: `pytest` green, vitest green,
   `ruff check`/`ruff format` clean.

7. **Commit, push, open PR** with spec + plan on the branch.

## Risk / rollback

- Behavior change is gated on `source_provider != "retrieval"`, so the default path is
  untouched. Rollback = revert the branch; no migrations or data changes.
