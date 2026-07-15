# Remove Dead AgentState Fields — Design

Date: 2026-07-09
Status: Approved (brainstorming)
Branch/PR: chore/remove-dead-agentstate-fields
Related: [[project_chat_orchestration]] (gap #6)

## Problem

`AgentState` (`src/agents/core/state.py:232-233`) declares two fields that are
never read or written anywhere in the repo:

```python
    short_term_memory: list[dict[str, str]] = field(default_factory=list)
    long_term_memory: dict[str, Any] = field(default_factory=dict)
```

Verified by grep: the only occurrences are the definitions themselves — no
readers, no writers, no tests. They are vestigial.

## Scope note (verified)

The gap-#6 "dead code" list from the chat-orchestration investigation was partly
overstated; only these two fields are genuinely dead. Explicitly **out of scope**
(NOT dead — do not touch):
- `StopReason.BUDGET_EXHAUSTED` — a *tested* return value of
  `LoopController.should_continue_searching`; the call site ignoring it is an
  integration gap, not dead code.
- `early_stops` metric — feeds the `early_stop_bonus` reward term and has a test.
- `LoopSnapshot.model_emitted_answer` — written-but-unread "reserved for Phase 2"
  scaffolding; left in place per the scope decision.

## Approach

Delete the two `AgentState` fields. Nothing else changes:
- `AgentState` uses `slots=True`; removing fields is clean.
- The `Any` import stays (used ~10 other places in the module).
- No behavior change (the fields were never accessed).

## Success criteria

- The two fields are gone from `AgentState`.
- A regression guard asserts they don't creep back.
- Existing state / agent-loop tests stay green.

## Testing

A tiny guard test (no construction needed):

```python
def test_dead_memory_fields_removed():
    from src.agents.core.state import AgentState
    fields = AgentState.__dataclass_fields__
    assert "short_term_memory" not in fields
    assert "long_term_memory" not in fields
```

Plus the existing suite (`test_agent_loop`, any state tests) as the behavior guard.

## Risks

- None of note — pure deletion of unreferenced fields.
