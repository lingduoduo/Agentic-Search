# Generated Context Pack

# Simulated Grpo Demo

## Sources

- [Specification: 2026-07-09-simulated-grpo-demo-design.md](../archive/specs/2026-07-09-simulated-grpo-demo-design.md)
- [Plan: 2026-07-09-simulated-grpo-demo.md](../archive/plans/2026-07-09-simulated-grpo-demo.md)

## Specification Context

### Component: `examples/run_bamboogle_grpo_train.py`

Single new script. Uses only existing machinery.

Flow:
1. `load_bamboogle(limit=args.limit)` → list of prompts (from
   `src/training/eval/bamboogle.py`).
2. `judge = SimulatedPreferenceJudge()`; adapt to the pointwise `JudgeFn` seam:
   `judge_fn = lambda pred, _gt: judge.score(pred)`.
3. `trainer = LLMGRPOTrainer.from_pretrained(args.model, judge_fn=judge_fn,
   config=LLMGRPOConfig(num_rollouts=args.num_rollouts,
   max_new_tokens=args.max_new_tokens, temperature=args.temperature),
   lr=args.lr, device=args.device, local_files_only=not
   args.allow_remote_model_downloads)`.
4. For `args.steps` iterations: take the next `args.batch_prompts` prompts (cycling

…

## Implementation Plan Context

### Task 1: Pure helpers (`make_judge_fn`, `cycle_prompt_batches`)

**Files:**
- Create: `examples/run_bamboogle_grpo_train.py`
- Test: `tests/unit/test_run_bamboogle_grpo_train.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `make_judge_fn(judge) -> Callable[[str, str], float]` — returns a fn that ignores its second arg and returns `judge.score(pred)`. `judge` is any object with a `.score(str) -> float` method (e.g. `SimulatedPreferenceJudge`).

…

### Task 2: Demo script CLI + training loop

**Files:**
- Modify: `examples/run_bamboogle_grpo_train.py` (add `_build_arg_parser`, `_run`, `main`)
- Test: `tests/unit/test_run_bamboogle_grpo_train.py` (add a `--help` smoke test)

**Interfaces:**
- Consumes: `make_judge_fn`, `cycle_prompt_batches` from Task 1.
- Consumes (existing, do not modify):
  - `src.training.judge.SimulatedPreferenceJudge()` with `.score` / `.as_batch_judge_fn`.
  - `src.training.eval.bamboogle.load_bamboogle(limit: int) -> list[dict]`; each item has `question: str` (and `golden_answers` we ignore).

…

### Task 3: Manual end-to-end verification

**Files:** none (verification only).

- [ ] **Step 1: Confirm `--help` is light**

Run: `python3 -m examples.run_bamboogle_grpo_train --help`
Expected: usage text prints; exits 0.

- [ ] **Step 2: Run a short real training run**

Run:
Expected (success criteria from the spec):
- Runs to completion, printing a metrics row per step.
- `mean_reward` values are in `[0, 1]`; `mean_kl` is > 0 by step 2 (policy diverging from the frozen reference); `clip_fraction` is reported.
- No exceptions.

- [ ] **Step 3: (Optional) confirm weights changed**

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
