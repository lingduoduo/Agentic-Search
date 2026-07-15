# Generated Context Pack

# System Preserving Token Crop

## Sources

- [Specification: 2026-07-09-system-preserving-token-crop-design.md](../archive/specs/2026-07-09-system-preserving-token-crop-design.md)
- [Plan: 2026-07-09-system-preserving-token-crop.md](../archive/plans/2026-07-09-system-preserving-token-crop.md)

## Specification Context

### Overview

Date: 2026-07-09
Status: Approved (brainstorming)
Branch/PR: feat/system-preserving-token-crop
Related: [[project_chat_orchestration]] (gap #2)

## Implementation Plan Context

### Task 1: `_crop_prompt_ids` helper + wiring + tests

**Files:**
- Modify: `src/agents/core/base.py`
- Test: `tests/unit/test_prompt_crop.py`

**Interfaces:**
- Produces: module-level `_crop_prompt_ids(full_ids: list[int], system_ids: list[int], budget: int) -> list[int]`; method `AgentLoopBase._encode_system_prefix(messages) -> list[int]`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_prompt_crop.py`:

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_prompt_crop.py -v`
Expected: FAIL — `ImportError` for `_crop_prompt_ids`.

- [ ] **Step 3: Add the helper + system-prefix encoder**

In `src/agents/core/base.py`, add a module-level helper (near the top, after the
regex constants):

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
