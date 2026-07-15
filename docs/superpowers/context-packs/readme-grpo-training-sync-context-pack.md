# Generated Context Pack

# Readme Grpo Training Sync

## Sources

- [Specification: 2026-07-09-readme-grpo-training-sync-design.md](../specs/2026-07-09-readme-grpo-training-sync-design.md)
- [Plan: 2026-07-09-readme-grpo-training-sync.md](../plans/2026-07-09-readme-grpo-training-sync.md)

## Specification Context

### Overview

Date: 2026-07-09
Status: Approved
Branch/PR: docs/readme-grpo-training-sync (PR #390)
Related: Simulated-Judge GRPO Demo,
Reward Dimensions Consolidation

## Implementation Plan Context

### Task 1: README edits + verification

**Files:**
- Modify: `README.md`
- Verify: `tests/unit/test_readme_examples.py`

- [x] **Step 1: Examples section** — add the `run_bamboogle_grpo_train.py` command beside `run_grpo_training_pipeline`.
- [x] **Step 2: RL Training feature list** — add the four-reward-dimensions bullet and the `SimulatedPreferenceJudge` (RLAIF) bullet.
- [x] **Step 3: Training table** — add rows for the demo entry point and `src/training/judge.py`.
- [x] **Step 4: Reward-components section** — document the 4-dimension rollup + partition invariant; add `retriever_aware` to the preset list.
- [x] **Step 5: Repository tree** — add `judge.py`; annotate `reward.py`.

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
