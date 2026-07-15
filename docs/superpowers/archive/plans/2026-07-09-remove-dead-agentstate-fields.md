# Remove Dead AgentState Fields Implementation Plan

> Steps use checkbox (`- [ ]`) syntax.

**Goal:** Delete the two unused `AgentState` fields (`short_term_memory`, `long_term_memory`).

**Architecture:** One deletion in `src/agents/core/state.py` + a regression-guard test.

## Global Constraints

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

```python
"""Guard: dead AgentState memory fields stay removed."""

from __future__ import annotations

from src.agents.core.state import AgentState


def test_dead_memory_fields_removed():
    fields = AgentState.__dataclass_fields__
    assert "short_term_memory" not in fields
    assert "long_term_memory" not in fields
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/unit/test_agent_state_fields.py -v`
Expected: FAIL — the fields still exist.

- [ ] **Step 3: Delete the fields**

In `src/agents/core/state.py`, remove these two lines from `AgentState`:

```python
    short_term_memory: list[dict[str, str]] = field(default_factory=list)
    long_term_memory: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 4: Run the guard test + regression**

Run: `python3 -m pytest tests/unit/test_agent_state_fields.py tests/unit/test_agent_loop.py -q`
Expected: PASS.

- [ ] **Step 5: Grep-verify no references remain**

Run: `grep -rn "short_term_memory\|long_term_memory" src/ tests/`
Expected: no output (zero references).

- [ ] **Step 6: Commit**

```bash
git add src/agents/core/state.py tests/unit/test_agent_state_fields.py
git commit -m "chore(state): remove unused AgentState memory fields

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

- Delete the two fields (spec §Approach) → Step 3. ✓
- Guard against re-introduction (spec §Testing) → Step 1 test. ✓
- No references remain → Step 5 grep. ✓
- Out-of-scope items untouched → only `state.py` `AgentState` edited. ✓
