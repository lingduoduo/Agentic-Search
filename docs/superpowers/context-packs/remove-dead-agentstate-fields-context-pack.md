# Generated Context Pack

# Remove Dead Agentstate Fields

## Sources

- [Specification: 2026-07-09-remove-dead-agentstate-fields-design.md](../specs/2026-07-09-remove-dead-agentstate-fields-design.md)
- [Plan: 2026-07-09-remove-dead-agentstate-fields.md](../plans/2026-07-09-remove-dead-agentstate-fields.md)

## Specification Context

### Scope note (verified)

The gap-#6 "dead code" list from the chat-orchestration investigation was partly
overstated; only these two fields are genuinely dead. Explicitly **out of scope**
(NOT dead — do not touch):
- `StopReason.BUDGET_EXHAUSTED` — a *tested* return value of
  `LoopController.should_continue_searching`; the call site ignoring it is an
  integration gap, not dead code.
- `early_stops` metric — feeds the `early_stop_bonus` reward term and has a test.
- `LoopSnapshot.model_emitted_answer` — written-but-unread "reserved for Phase 2"
  scaffolding; left in place per the scope decision.

### Testing

A tiny guard test (no construction needed):

```python
def test_dead_memory_fields_removed():
    from src.agents.core.state import AgentState
    fields = AgentState.__dataclass_fields__
    assert "short_term_memory" not in fields
    assert "long_term_memory" not in fields
```

Plus the existing suite (`test_agent_loop`, any state tests) as the behavior guard.

### Risks

- None of note — pure deletion of unreferenced fields.

## Implementation Plan Context

### Global Constraints

- Branch off `main` (never commit to `main`); branch `chore/remove-dead-agentstate-fields`.
- Pure deletion — no behavior change; keep the `Any` import (used elsewhere).
- Out of scope: `BUDGET_EXHAUSTED`, `early_stops`, `model_emitted_answer` (not dead).
- Match repo ruff formatting.

---

### Task 1: Delete the fields + add the guard test

**Files:**
- Modify: `src/agents/core/state.py`
- Test: `tests/unit/test_agent_state_fields.py`

- [ ] **Step 1: Write the failing guard test**

Create `tests/unit/test_agent_state_fields.py`:

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/unit/test_agent_state_fields.py -v`
Expected: FAIL — the fields still exist.

- [ ] **Step 3: Delete the fields**

In `src/agents/core/state.py`, remove these two lines from `AgentState`:

- [ ] **Step 4: Run the guard test + regression**

Run: `python3 -m pytest tests/unit/test_agent_state_fields.py tests/unit/test_agent_loop.py -q`
Expected: PASS.

- [ ] **Step 5: Grep-verify no references remain**

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
