# README GRPO Training Sync — Design

Date: 2026-07-09
Status: Approved
Branch/PR: docs/readme-grpo-training-sync (PR #390)
Related: [Simulated-Judge GRPO Demo](2026-07-09-simulated-grpo-demo-design.md),
[Reward Dimensions Consolidation](2026-07-09-reward-dimensions-consolidation-design.md)

## Problem

The README predates the merged GRPO work (#387, #388). It does not mention the
new `examples/run_bamboogle_grpo_train.py` policy-training demo, the
`SimulatedPreferenceJudge` (RLAIF stand-in), or the four reward dimensions
(`REWARD_DIMENSIONS` / `group_reward_components` / `reward_dimensions()` /
`dim_*` keys). Its reward-preset list is also missing `retriever_aware`.

Goal: bring the README's training/reward documentation in line with what is now
on `main`, with surgical edits that match the existing style.

## Non-goals

- No code changes. Documentation only.
- No restructuring of the README. Only additive/corrective edits to existing
  training/reward sections.
- No new runnable example commands beyond the one demo that now exists.

## Approach (six surgical edits)

1. **Examples section** — add the `run_bamboogle_grpo_train.py` command next to
   the existing `run_grpo_training_pipeline` smoke test, noting it updates a
   policy (plain generation, no retrieval server, CPU/MPS).
2. **RL Training feature list** — add a bullet for the four reward dimensions
   and a bullet for the `SimulatedPreferenceJudge` RLAIF path + demo.
3. **Training section table** — add rows for the demo entry point and
   `src/training/judge.py`.
4. **Reward-components section** — document the 4-dimension rollup and the
   pre-scale partition invariant; complete the preset list with `retriever_aware`.
5. **Repository tree** — add `judge.py`; annotate `reward.py` with the dimensions.

## Correctness / accuracy criteria

- Every new symbol reference resolves to real code: `run_bamboogle_grpo_train`,
  `REWARD_DIMENSIONS`, `group_reward_components`, `reward_dimensions`,
  `SimulatedPreferenceJudge`.
- `tests/unit/test_readme_examples.py` stays green (it imports functions
  directly; it does not shell-execute README bash blocks, so new command blocks
  carry no execution risk).
- Statements about the reward view match the merged behavior: purely additive,
  pre-scale, `sum(dims) == terminal_reward + shaping_total == total / reward_scale`.

## Testing

- Run `tests/unit/test_readme_examples.py` — expect pass (docs-only change).
- Grep-verify each new reference exists in the codebase (demo file, reward
  symbols, judge class).
