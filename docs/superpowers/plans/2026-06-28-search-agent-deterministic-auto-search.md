# Search-Agent Deterministic Auto-Search Implementation Plan

Spec: [docs/superpowers/specs/2026-06-28-search-agent-deterministic-auto-search-design.md](../specs/2026-06-28-search-agent-deterministic-auto-search-design.md)
Date: 2026-06-28

## Global Constraints

- Surgical change: touch only `SearchAgentLoop` config + run loop, plus tests.
- Backward compatible: gated behind `auto_search_on_deadend` (default `True`);
  set `False` to restore the exact prior behavior.
- No change to the downstream search-execution path, the reward metrics, or any
  other agent loop.
- TDD: write the two failing unit tests first, then implement until green, then
  run the full search-loop suite for regressions.

## File Map

- `src/agents/search.py`
  - `SearchAgentLoopConfig`: add `auto_search_on_deadend: bool = True`.
  - `SearchAgentLoop.run`: inject first-turn auto-search before the
    `if not actions:` format-recovery branch.
- `tests/unit/test_agent_loop.py`: two new tests.

## Execution Order

### Task 1: Write failing tests

Add to `tests/unit/test_agent_loop.py` (reusing `DummyTokenizerWithEncode`,
`DummyServerManager`, `FakeSearchClient`, `SearchResult`):

- `test_search_agent_loop_auto_searches_when_model_emits_no_action`
  - Responses: turn 1 = tag-less prose; turn 2 = `<answer>…</answer>`.
  - `FakeSearchClient` keyed on `("What is FAISS?",)`.
  - Assert: `loop._search_client.calls == [["What is FAISS?"]]`,
    `output.context.num_rounds == 1`, `output.metrics["search_rounds"] == 1.0`,
    `output.final_answer` is the turn-2 answer, and a control-flow event with
    `action == "auto_search"` is present.
- `test_search_agent_loop_auto_search_disabled_preserves_format_recovery`
  - Config: `auto_search_on_deadend=False`, `force_answer_on_deadend=False`,
    all responses tag-less.
  - Assert: `loop._search_client.calls == []` and
    `output.metrics["search_rounds"] == 0.0`.

Verify: `pytest tests/unit/test_agent_loop.py -k auto_search` → both fail
(flag/behavior absent).

### Task 2: Add the config flag

In `SearchAgentLoopConfig`, beside `force_answer_on_deadend`:

```python
# Auto-issue a search on the user's question when the first turn produces no
# recognized action tag, so retrieval always fires at least once. RL rollouts
# may set this False to preserve dead-end penalties during training.
auto_search_on_deadend: bool = True
```

### Task 3: Inject the first-turn auto-search

In `SearchAgentLoop.run`, after `working_messages.append({"role": "assistant", …})`
and before the existing `if actions:` block:

```python
if (
    not actions
    and cfg.auto_search_on_deadend
    and state.search_rounds == 0
    and question
):
    actions = [(cfg.search_tag, question)]
    recorder.record(
        turn=num_turns,
        component="planner",
        action="auto_search",
        status="decided",
        details={"query": question, "reason": "no_action_first_turn"},
    )
```

`question` and `state` are already in scope. The synthesized action flows through
the unchanged `_collect_requested_queries_and_urls` path.

Verify: `pytest tests/unit/test_agent_loop.py -k auto_search` → both pass.

### Task 4: Regression check

- `pytest tests/unit/test_agent_loop.py -q`
- `pytest tests/unit/test_loop_controller.py tests/unit/test_reward.py tests/unit/test_sft.py tests/unit/test_grpo.py -q`
- `ruff check src/agents/search.py tests/unit/test_agent_loop.py`

## Final Acceptance Checklist

- [ ] New flag `auto_search_on_deadend` defaults to `True`.
- [ ] Tag-less first turn triggers exactly one search on the user's question.
- [ ] `auto_search` control-flow event recorded.
- [ ] `auto_search_on_deadend=False` reproduces prior format-recovery behavior.
- [ ] Existing search-loop, reward, sft, grpo unit tests still pass.
- [ ] `ruff` clean on touched files.
