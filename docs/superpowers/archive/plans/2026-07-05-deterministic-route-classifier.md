# Deterministic route classifier — plan

Spec: `docs/superpowers/specs/2026-07-05-deterministic-route-classifier-design.md`

## Steps

1. **RED** — add failing tests → verify: both fail for "feature missing".
   - `tests/unit/test_llm_providers.py::test_complete_forwards_temperature`
   - `tests/unit/servers/web/test_agent_router.py::test_classify_route_uses_deterministic_decoding`
2. **GREEN** — implement minimal fix → verify: new tests pass.
   - `providers.py`: forward `temperature` kwarg into `complete()` body.
   - `intent_routing.py`: `classify_route` passes `temperature=0.0`.
3. **Regression** — run web/routing/llm/fallback suites → verify: all green.
4. **Lint** — `ruff check` on changed files → verify: clean.
5. **Ship** — commit on `fix/deterministic-route-classifier`, push, open PR with
   spec + plan.

## Status

Steps 1–4 complete (236 passed, lint clean). Step 5 in progress.
