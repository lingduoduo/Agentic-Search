# Search-Agent Deterministic Auto-Search — Plan

Spec: [2026-06-28-search-agent-deterministic-auto-search-design.md](../specs/2026-06-28-search-agent-deterministic-auto-search-design.md)
Status: shipped (consolidated in PR #347).

**Goal:** guarantee retrieval fires at least once at the format-error dead-end,
gated behind `auto_search_on_deadend` (default `True`). Honest scope — retrieval
runs once; the answer still requires an `<answer>` tag (no-fabricate invariant).

## Tasks

1. **Config flag.** `SearchAgentLoopConfig.auto_search_on_deadend: bool = True`.
2. **Trigger (in `SearchAgentLoop.run`, the `if not actions:` branch).** Before
   the format-recovery path, when `auto_search_on_deadend and state.search_rounds
   == 0 and question and consecutive_format_errors + 1 >=
   max_consecutive_format_errors`: set `actions = [(cfg.search_tag, question)]`,
   reset the format-error counter, and record an `auto_search` event. Otherwise
   fall through to the existing `_handle_no_action` path.
3. **Tests (honest).**
   - `…auto_searches_when_model_emits_no_action`: genuinely tag-less model ⇒
     `search_rounds == 1`, `auto_search` event, `final_answer is None`.
   - `…auto_search_disabled_preserves_format_recovery`: flag off ⇒ no retrieval.
   - Pin the two legacy dead-end tests to `auto_search_on_deadend=False`.
4. **Regression + lint:** `tests/unit/test_agent_loop.py`, reward/sft/grpo; `ruff`.

## Done when

Flag defaults on; tag-less dead-end fires exactly one search and records the
event; the no-fabricate invariant is intact (tag-less ⇒ `final_answer is None`);
flag off reproduces prior behavior; suites + `ruff` green.
