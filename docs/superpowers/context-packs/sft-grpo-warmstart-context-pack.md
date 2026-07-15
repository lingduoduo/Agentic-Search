# Generated Context Pack

# Sft Grpo Warmstart

## Sources

- [Specification: 2026-06-17-sft-grpo-warmstart-design.md](../specs/2026-06-17-sft-grpo-warmstart-design.md)
- [Plan: 2026-06-17-sft-grpo-warmstart.md](../plans/2026-06-17-sft-grpo-warmstart.md)

## Specification Context

### Out of Scope

- LoRA / PEFT adapters
- DPO or preference-pair training
- Auto-triggering from the beat worker
- New dataclasses — `SFTExample` (already in `src/training/sft.py`) is reused as-is

---

### 2. Architecture

Signal is preserved across phases: thumbs-up sessions are used in Phase 1 as imitation targets and in Phase 2 as `human_signal=+1.0` via `load_feedback_examples`.

---

### `test_sft_examples.py`

- Thumbs-up session with assistant turn → `SFTExample` with correct `prompt_messages` and `completion`
- Session with no assistant turn is skipped
- JSONL row `{"question": "Q", "response": "R"}` → correct `SFTExample`
- JSONL row missing `response` key is skipped
- Both sources merged: total count = DB examples + JSONL examples
- `ValueError` when total < `min_ratings`

## Implementation Plan Context

### Task 1: `load_sft_examples` in `src/training/data.py`

**Files:**
- Modify: `src/training/data.py` (append after the `load_feedback_examples` function)
- Test: `tests/unit/test_sft_examples.py`

- [x] **Step 1: Write the failing tests**

Create `tests/unit/test_sft_examples.py`:

- [x] **Step 2: Run tests — verify they fail**

Expected: all 8 fail with `ImportError: cannot import name 'load_sft_examples'`

- [x] **Step 3: Add `load_sft_examples` to `src/training/data.py`**

Add `import json` and `import logging` to the imports block (after `from pathlib import Path`), then append after `load_feedback_examples`:

- [x] **Step 4: Run tests — verify they pass**

Expected: 8 passed

- [x] **Step 5: Commit**

---

### Task 2: `SFTConfig` and `SFTTrainer` in `src/training/sft.py`

**Files:**
- Modify: `src/training/sft.py`
- Test: `tests/unit/test_sft_trainer.py`

### Task 3: CLI script `examples/run_sft_grpo.py`

**Files:**
- Create: `examples/run_sft_grpo.py`

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
