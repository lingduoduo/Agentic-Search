# Generated Context Pack

# Readme Grpo Training Sync

## Sources

- [Specification: 2026-07-09-readme-grpo-training-sync-design.md](../specs/2026-07-09-readme-grpo-training-sync-design.md)
- [Plan: 2026-07-09-readme-grpo-training-sync.md](../plans/2026-07-09-readme-grpo-training-sync.md)

## Specification Context

### Non-goals

- No code changes. Documentation only.
- No restructuring of the README. Only additive/corrective edits to existing
  training/reward sections.
- No new runnable example commands beyond the one demo that now exists.

### Testing

- Run `tests/unit/test_readme_examples.py` — expect pass (docs-only change).
- Grep-verify each new reference exists in the codebase (demo file, reward
  symbols, judge class).

## Implementation Plan Context

### Global Constraints

- Branch off `main` (never commit to `main`); PR #390.
- Documentation only — no code, no behavior change.
- Match existing README style (tables, fenced bash blocks, `code` symbol refs).
- Every new symbol reference must resolve to real code.
- `tests/unit/test_readme_examples.py` must stay green.

---

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
