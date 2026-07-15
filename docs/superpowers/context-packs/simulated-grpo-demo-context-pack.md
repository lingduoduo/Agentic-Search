# Generated Context Pack

# Simulated Grpo Demo

## Sources

- [Specification: 2026-07-09-simulated-grpo-demo-design.md](../specs/2026-07-09-simulated-grpo-demo-design.md)
- [Plan: 2026-07-09-simulated-grpo-demo.md](../plans/2026-07-09-simulated-grpo-demo.md)

## Specification Context

### Non-goals

- No trained reward model / pairwise comparison data (the judge is pointwise and
  reference-free; that stays as-is).
- No classic PPO with a value critic — GRPO only (critic-free group baseline).
- No retrieval server, no `SearchAgentLoop` rollouts.
- No durability: no checkpointing, resume, or `train_loop.py` wiring. (Possible
  future follow-up, explicitly out of scope here.)
- No changes to `reward.py`, `judge.py`, or the trainers.

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
   the loaded list), call `trainer.step(prompts, ground_truths=[""] * len(prompts))`
   (ground truth ignored by the judge).
5. Print `step | mean_reward | mean_advantage | mean_kl | clip_fraction | loss`
   plus a rolling mean of `mean_reward`.

### Two pure helpers (unit-tested)

- `make_judge_fn(judge) -> JudgeFn`: returns `lambda pred, _gt: judge.score(pred)`.
- `cycle_prompt_batches(prompts, steps, batch_size) -> list[list[str]]` (or a
  generator): yields `steps` batches of size `batch_size`, cycling `prompts` when
  exhausted. Keeps the demo runnable with a small `--limit`.

### Testing

Automated (fast, **no model load** — respects the web-test model-load gotcha):
- `make_judge_fn`: returned fn ignores the ground-truth arg and returns
  `judge.score(pred)` for a few sample answers.
- `cycle_prompt_batches`: yields exactly `steps` batches of `batch_size`, cycling
  correctly when `steps * batch_size > len(prompts)`; handles `len(prompts) == 1`.

Manual (`/verify`, not default pytest): one real `--steps 2` run on a tiny model
(`Qwen/Qwen2.5-0.5B-Instruct`, `--device cpu`) asserting criteria 1–4 by
observation.

### Risks

- **Noisy signal**: reward may not rise in 10 steps. Mitigated by framing success
  as "real updates + sane diagnostics + weights changed", not monotonic reward.
- **Reward hacks form, not correctness**: the judge rewards length/uniqueness/no-
  hedging, so the policy may learn to pad answers. This is expected and is exactly
  the known limitation of the reference-free judge — the demo illustrates the
  mechanism, not a production reward. Documented in the script docstring.

## Implementation Plan Context

### Global Constraints

- Never commit to `main`; work on branch `feat/simulated-grpo-demo` (already created).
- No changes to `reward.py`, `judge.py`, or any trainer — reuse only.
- Heavy imports (`torch`, `transformers`, `src.training.ppo.*`) must be **inside functions**, never at module top level, so `--help` stays light (matches commit `0328b12`).
- The judge is reference-free and pointwise: `ground_truths` are passed as `""` and ignored.
- Match existing example conventions (`examples/run_bamboogle_synthetic_grpo.py`, `examples/run_feedback_grpo.py`).
- Tests for pure helpers must NOT load a model or import torch (respects the web-test model-load gotcha).

---

### Task 1: Pure helpers (`make_judge_fn`, `cycle_prompt_batches`)

**Files:**
- Create: `examples/run_bamboogle_grpo_train.py`
- Test: `tests/unit/test_run_bamboogle_grpo_train.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `make_judge_fn(judge) -> Callable[[str, str], float]` — returns a fn that ignores its second arg and returns `judge.score(pred)`. `judge` is any object with a `.score(str) -> float` method (e.g. `SimulatedPreferenceJudge`).
  - `cycle_prompt_batches(prompts: list[str], steps: int, batch_size: int) -> list[list[str]]` — returns exactly `steps` lists, each of length `batch_size`, drawing from `prompts` in order and wrapping around when exhausted. Raises `ValueError` if `prompts` is empty or `batch_size < 1` or `steps < 1`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_run_bamboogle_grpo_train.py`:

```python
"""Unit tests for the simulated-judge GRPO demo helpers.

These test only the pure helpers and must not load a model or import torch.
"""

from __future__ import annotations

import pytest

from examples.run_bamboogle_grpo_train import (
    cycle_prompt_batches,
    make_judge_fn,
)


class _StubJudge:
    """Minimal judge: score is the word count, ground truth ignored."""

    def score(self, answer: str) -> float:
        return float(len(answer.split()))


def test_make_judge_fn_ignores_ground_truth_and_returns_score():
    judge_fn = make_judge_fn(_StubJudge())
    assert judge_fn("two words", "IGNORED GOLD") == 2.0
    assert judge_fn("one two three", "") == 3.0


def test_make_judge_fn_matches_real_judge_pointwise():

_[Section compacted.]_

### Task 2: Demo script CLI + training loop

**Files:**
- Modify: `examples/run_bamboogle_grpo_train.py` (add `_build_arg_parser`, `_run`, `main`)
- Test: `tests/unit/test_run_bamboogle_grpo_train.py` (add a `--help` smoke test)

**Interfaces:**
- Consumes: `make_judge_fn`, `cycle_prompt_batches` from Task 1.
- Consumes (existing, do not modify):
  - `src.training.judge.SimulatedPreferenceJudge()` with `.score` / `.as_batch_judge_fn`.
  - `src.training.eval.bamboogle.load_bamboogle(limit: int) -> list[dict]`; each item has `question: str` (and `golden_answers` we ignore).
  - `src.training.ppo.llm_grpo_trainer.LLMGRPOTrainer.from_pretrained(model_name_or_path, judge_fn, lr=..., config=..., device=..., **hf_kwargs) -> LLMGRPOTrainer` and `.step(prompts: list[str], ground_truths: list[str]) -> dict[str, float]` returning keys `loss`, `mean_kl`, `mean_reward`, `clip_fraction`, `mean_advantage`.
  - `src.training.ppo.llm_grpo_trainer.LLMGRPOConfig(num_rollouts=..., max_new_tokens=..., temperature=...)`.
- Produces: `main() -> None` CLI entrypoint.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_run_bamboogle_grpo_train.py`:

```python
def test_help_runs_without_torch(monkeypatch):
    """`--help` must exit cleanly and not require heavy imports at module top."""
    import runpy
    import sys

    monkeypatch.setattr(
        sys, "argv", ["run_bamboogle_grpo_train", "--help"]
    )
    with pytest.raises(SystemExit) as exc:
        runpy.run_module(
            "examples.run_bamboogle_grpo_train", run_name="__main__"
        )
    assert exc.value.code == 0
```

_[Section compacted.]_

### Task 3: Manual end-to-end verification

**Files:** none (verification only).

- [ ] **Step 1: Confirm `--help` is light**

Run: `python3 -m examples.run_bamboogle_grpo_train --help`
Expected: usage text prints; exits 0.

- [ ] **Step 2: Run a short real training run**

Run:
```bash
python3 -m examples.run_bamboogle_grpo_train \
  --model Qwen/Qwen2.5-0.5B-Instruct --device cpu \
  --allow_remote_model_downloads --steps 2 --num_rollouts 2 \
  --batch_prompts 1 --limit 4 --max_new_tokens 32
```
Expected (success criteria from the spec):
- Runs to completion, printing a metrics row per step.
- `mean_reward` values are in `[0, 1]`; `mean_kl` is > 0 by step 2 (policy diverging from the frozen reference); `clip_fraction` is reported.
- No exceptions.

- [ ] **Step 3: (Optional) confirm weights changed**

If a quick check is desired, add a throwaway snippet (do NOT commit) that snapshots one policy parameter before the loop and asserts it differs after — or simply rely on `mean_kl > 0` as the proxy that `optimizer.step` mutated the policy.

---

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
