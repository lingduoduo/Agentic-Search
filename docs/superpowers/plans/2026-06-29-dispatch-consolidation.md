# Agent Dispatch Consolidation — Plan

Spec: `docs/superpowers/specs/2026-06-29-dispatch-consolidation-design.md`

**Goal:** One place builds/runs each agent loop; one place assembles the response.
`_run_auto_routed` and the explicit-mode chain in `app.py` both dispatch through
shared runners and `_finalize_response`. Behavior preserved except additive
convergences listed in the spec. TDD; commit per task.

## Tasks

1. **Runner tests (red).** New `tests/unit/servers/web/test_loop_runners.py`:
   assert each of `_run_search_agent`, `_run_agentic_rag`, `_run_tool_agent`
   returns `(answer, citations, documents, intent, extra)` with the right `extra`
   keys, driven by fake loops. → verify: tests fail (helpers absent).

2. **Extract runners (green).** Add `_run_search_agent`, `_run_agentic_rag`,
   `_run_tool_agent` to `app.py` per spec signatures. → verify: task-1 tests pass.

3. **Extract `_finalize_response` + rewire call sites.** Route `_run_auto_routed`
   loop branches and every explicit-mode branch (incl. `search_tool`,
   `hybrid_search`, default `answer_with_retrieval`) through the runners + tail.
   Keep degradation in `_run_auto_routed`, 400 guards at explicit sites. Delete the
   now-dead inline construction/extraction. → verify: `pytest tests/unit/servers/web/ -v`.

4. **Update converged-contract tests.** Fix `test_web_experience_app.py`,
   `test_tool_trace.py`, `test_agent_router.py` for the additive convergences.
   → verify: `pytest` (full) green; `ruff check` + `ruff format --check` clean.

## Done when

Each loop constructed/run once; doc-extraction once; response assembly once;
public API + streaming + routing unchanged; full suite + lint green.
