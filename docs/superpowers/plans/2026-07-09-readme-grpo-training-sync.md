# README GRPO Training Sync Implementation Plan

> Documentation-only change. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Sync the README's training/reward documentation with the merged GRPO work (#387, #388).

**Architecture:** Six surgical, additive edits to `README.md` — no code touched.

**Tech Stack:** Markdown.

## Global Constraints

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
- [x] **Step 6: Verify** — `python3 -m pytest tests/unit/test_readme_examples.py -q` (green) and grep-confirm each new reference resolves (`run_bamboogle_grpo_train`, `REWARD_DIMENSIONS`/`group_reward_components`/`reward_dimensions`, `SimulatedPreferenceJudge`).
- [x] **Step 7: Commit** on branch `docs/readme-grpo-training-sync`.

---

## Self-Review

- Spec's six edits (§Approach) → all applied in Task 1 Steps 1–5. ✓
- Accuracy criteria (§Correctness) → Step 6 (test green + grep-verified). ✓
- Docs-only, no code — matches Global Constraints. ✓
