# Generated Context Pack

# Remove Dead Agentstate Fields

## Sources

- [Specification: 2026-07-09-remove-dead-agentstate-fields-design.md](../archive/specs/2026-07-09-remove-dead-agentstate-fields-design.md)
- [Plan: 2026-07-09-remove-dead-agentstate-fields.md](../archive/plans/2026-07-09-remove-dead-agentstate-fields.md)

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

## Implementation Plan Context

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
