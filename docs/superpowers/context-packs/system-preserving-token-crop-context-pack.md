# Generated Context Pack

# System Preserving Token Crop

## Sources

- [Specification: 2026-07-09-system-preserving-token-crop-design.md](../specs/2026-07-09-system-preserving-token-crop-design.md)
- [Plan: 2026-07-09-system-preserving-token-crop.md](../plans/2026-07-09-system-preserving-token-crop.md)

## Specification Context

### Non-goals

- No message-level trimming / re-render loop (approach B) — higher blast radius
  on a shared method; deferred.
- No per-loop changes; no config flag (under-budget output is unchanged, over-budget
  strictly improves).
- No change to the tokenizer or chat template.

### Testing (no tokenizer for the core)

Unit tests on the pure `_crop_prompt_ids` (plain int lists):
1. under budget unchanged; `budget <= 0` unchanged.
2. no system → tail-crop.
3. system + over budget → `result[:len(system)] == system_ids`, `len(result) == budget`,
   and `result[len(system):] == full[-(budget-len(system)):]`.
4. system larger than budget → `system[-budget:]`.

One integration test: instantiate `AgentLoopBase` with a minimal dummy tokenizer
(fallback path, `encode(s) = list(s.encode())`, no `apply_chat_template`),
`prompt_length` tiny, a system message + long later messages; assert the encoded
system content is a prefix of `_build_prompt_ids_sync(...)`.

### Risks

- The mid-message tail cut can slightly garble the boundary between the system
  prefix and the tail — but the pre-existing behavior already cut mid-token, and
  the system instruction surviving is the higher-order correctness win.

## Implementation Plan Context

### Global Constraints

- Branch off `main` (never commit to `main`); branch `feat/system-preserving-token-crop`.
- Under-budget output must stay byte-identical to today (zero regression for normal prompts).
- Change only `base.py`; no per-loop changes, no config flag.
- Match repo ruff formatting.

---

### Task 1: `_crop_prompt_ids` helper + wiring + tests

**Files:**
- Modify: `src/agents/core/base.py`
- Test: `tests/unit/test_prompt_crop.py`

**Interfaces:**
- Produces: module-level `_crop_prompt_ids(full_ids: list[int], system_ids: list[int], budget: int) -> list[int]`; method `AgentLoopBase._encode_system_prefix(messages) -> list[int]`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_prompt_crop.py`:

```python
"""Unit tests for system-preserving prompt-id truncation."""

from __future__ import annotations

from src.agents.core.base import AgentLoopBase, AgentLoopConfig, _crop_prompt_ids
from tests.unit.test_agent_loop import DummyServerManager, DummyTokenizerWithEncode


def test_under_budget_unchanged():
    full = [1, 2, 3]
    assert _crop_prompt_ids(full, [9], 10) == full
    assert _crop_prompt_ids(full, [9], 0) == full  # budget <= 0 → unchanged


def test_no_system_tail_crop():
    full = list(range(10))
    assert _crop_prompt_ids(full, [], 4) == [6, 7, 8, 9]


def test_system_preserved_over_budget():
    system = [100, 101]
    full = list(range(20))  # far over budget
    out = _crop_prompt_ids(full, system, 6)
    assert len(out) == 6
    assert out[:2] == system
    assert out[2:] == full[-(6 - 2):]  # recent tail fills the rest


def test_system_larger_than_budget_degenerate():
    system = [1, 2, 3, 4, 5]
    full = list(range(50))
    out = _crop_prompt_ids(full, system, 3)
    assert out == system[-3:]


def test_build_prompt_ids_sync_keeps_system_prefix():
    loop = AgentLoopBase(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=DummyServerManager([]),

_[Section compacted.]_

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
