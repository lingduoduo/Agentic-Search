# Simulated-Judge GRPO Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a runnable example script that optimizes a policy with GRPO against the reference-free `SimulatedPreferenceJudge`, closing the sample → generate → judge → update loop.

**Architecture:** A single new example, `examples/run_bamboogle_grpo_train.py`, drives the existing `LLMGRPOTrainer` (plain prompt→response, no retrieval). It loads Bamboogle prompts, adapts the pointwise judge into the `judge_fn` seam, and runs N real gradient steps, printing per-step diagnostics. Two pure helpers (`make_judge_fn`, `cycle_prompt_batches`) are unit-tested without loading a model; heavy imports (`torch`/`transformers`/trainer) are lazy so `--help` stays light.

**Tech Stack:** Python 3, PyTorch, HuggingFace Transformers, existing `src/training/ppo/llm_grpo_trainer.py`, `src/training/judge.py`, `src/training/eval/bamboogle.py`.

## Global Constraints

- Never commit to `main`; work on branch `feat/simulated-grpo-demo` (already created).
- No changes to `reward.py`, `judge.py`, or any trainer — reuse only.
- Heavy imports (`torch`, `transformers`, `src.training.ppo.*`) must be **inside functions**, never at module top level, so `--help` stays light (matches commit `0328b12`).
- The judge is reference-free and pointwise: `ground_truths` are passed as `""` and ignored.
- Match existing example conventions (`examples/run_bamboogle_synthetic_grpo.py`, `examples/run_feedback_grpo.py`).
- Tests for pure helpers must NOT load a model or import torch (respects the web-test model-load gotcha).

---

## File Structure

- Create: `examples/run_bamboogle_grpo_train.py` — the demo script + two pure helpers.
- Create: `tests/unit/test_run_bamboogle_grpo_train.py` — unit tests for the two helpers + a `--help` smoke test.

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
    from src.training.judge import SimulatedPreferenceJudge

    judge = SimulatedPreferenceJudge()
    judge_fn = make_judge_fn(judge)
    answer = "Paris is the capital of France."
    assert judge_fn(answer, "whatever") == judge.score(answer)


def test_cycle_prompt_batches_no_wrap():
    prompts = ["a", "b", "c", "d"]
    batches = cycle_prompt_batches(prompts, steps=2, batch_size=2)
    assert batches == [["a", "b"], ["c", "d"]]


def test_cycle_prompt_batches_wraps_when_exhausted():
    prompts = ["a", "b", "c"]
    batches = cycle_prompt_batches(prompts, steps=3, batch_size=2)
    # a,b | c,a | b,c  — continuous cycle across step boundaries
    assert batches == [["a", "b"], ["c", "a"], ["b", "c"]]


def test_cycle_prompt_batches_single_prompt():
    batches = cycle_prompt_batches(["only"], steps=2, batch_size=2)
    assert batches == [["only", "only"], ["only", "only"]]


@pytest.mark.parametrize(
    "prompts,steps,batch",
    [([], 1, 1), (["a"], 0, 1), (["a"], 1, 0)],
)
def test_cycle_prompt_batches_rejects_bad_args(prompts, steps, batch):
    with pytest.raises(ValueError):
        cycle_prompt_batches(prompts, steps=steps, batch_size=batch)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_run_bamboogle_grpo_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'examples.run_bamboogle_grpo_train'` (file does not exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `examples/run_bamboogle_grpo_train.py` with the module docstring and the two helpers only (no argparse/main yet). Keep all heavy imports out of module scope.

```python
"""Optimize a policy with GRPO against the reference-free SimulatedPreferenceJudge.

For each step this samples ``--num_rollouts`` completions per prompt directly
from the model (no retrieval), scores each completion with the pointwise
``SimulatedPreferenceJudge`` (a stand-in for an LLM-as-judge), computes
group-relative GRPO advantages, and updates the policy with a PPO-clip + KL
objective via the existing ``LLMGRPOTrainer``. It closes the sample -> generate
-> judge -> update loop that the merged synthetic-data demo (#387) stopped short
of.

The judge is reference-free: it scores answer *form* (length, unique-word ratio,
no hedging), not correctness. Expect the policy to optimize form — this demo
illustrates the GRPO mechanism against a simulated reward, not a production
reward. Ground-truth answers are not used.

Quick start (local CPU, self-contained, slow):
    python3 -m examples.run_bamboogle_grpo_train \\
        --model Qwen/Qwen2.5-0.5B-Instruct --device cpu \\
        --allow_remote_model_downloads --steps 10
"""

from __future__ import annotations

import argparse
from typing import Any, Callable


def make_judge_fn(judge: Any) -> Callable[[str, str], float]:
    """Adapt a pointwise judge to the ``(pred, ground_truth) -> float`` seam.

    ``judge`` must expose ``score(answer: str) -> float``. The ground-truth
    argument is ignored because the judge is reference-free.
    """

    def _judge_fn(pred: str, _ground_truth: str) -> float:
        return float(judge.score(pred))

    return _judge_fn


def cycle_prompt_batches(
    prompts: list[str],
    steps: int,
    batch_size: int,
) -> list[list[str]]:
    """Return ``steps`` batches of ``batch_size`` prompts, cycling ``prompts``.

    Prompts are drawn in order and wrap around continuously across step
    boundaries so a small prompt pool can feed many steps.
    """
    if not prompts:
        raise ValueError("prompts must be non-empty.")
    if steps < 1:
        raise ValueError("steps must be >= 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    batches: list[list[str]] = []
    cursor = 0
    n = len(prompts)
    for _ in range(steps):
        batch = [prompts[(cursor + i) % n] for i in range(batch_size)]
        cursor = (cursor + batch_size) % n
        batches.append(batch)
    return batches
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_run_bamboogle_grpo_train.py -v`
Expected: PASS (all 8 test cases).

- [ ] **Step 5: Commit**

```bash
git add examples/run_bamboogle_grpo_train.py tests/unit/test_run_bamboogle_grpo_train.py
git commit -m "feat(grpo): simulated-judge GRPO demo helpers + tests

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

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

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_run_bamboogle_grpo_train.py::test_help_runs_without_torch -v`
Expected: FAIL — no `main`/`__main__` block yet, so `--help` is not handled (argparse `SystemExit(0)` never raised).

- [ ] **Step 3: Write minimal implementation**

Add to `examples/run_bamboogle_grpo_train.py` below the helpers. Heavy imports go **inside** `_run`.

```python
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize a policy with GRPO against SimulatedPreferenceJudge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model id or path")
    parser.add_argument("--steps", type=int, default=10, help="GRPO update steps")
    parser.add_argument("--num_rollouts", type=int, default=4, help="completions per prompt")
    parser.add_argument("--batch_prompts", type=int, default=2, help="prompts per step")
    parser.add_argument("--limit", type=int, default=8, help="Bamboogle prompts to load")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="cpu", help="cpu / mps / cuda")
    parser.add_argument("--allow_remote_model_downloads", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _run(args: argparse.Namespace) -> None:
    import torch

    from src.training.eval.bamboogle import load_bamboogle
    from src.training.judge import SimulatedPreferenceJudge
    from src.training.ppo.llm_grpo_trainer import LLMGRPOConfig, LLMGRPOTrainer

    torch.manual_seed(args.seed)

    examples = load_bamboogle(limit=args.limit)
    prompts = [ex["question"] for ex in examples]
    if not prompts:
        raise SystemExit("No Bamboogle prompts loaded; check --limit / network.")

    judge = SimulatedPreferenceJudge()
    judge_fn = make_judge_fn(judge)

    trainer = LLMGRPOTrainer.from_pretrained(
        args.model,
        judge_fn=judge_fn,
        lr=args.lr,
        config=LLMGRPOConfig(
            num_rollouts=args.num_rollouts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        ),
        device=args.device,
        local_files_only=not args.allow_remote_model_downloads,
    )

    batches = cycle_prompt_batches(prompts, steps=args.steps, batch_size=args.batch_prompts)
    reward_history: list[float] = []
    print(
        f"step | mean_reward | rolling | mean_adv | mean_kl | clip_frac | loss"
    )
    for step, batch in enumerate(batches, 1):
        metrics = trainer.step(batch, ground_truths=[""] * len(batch))
        reward_history.append(metrics["mean_reward"])
        rolling = sum(reward_history) / len(reward_history)
        print(
            f"{step:4d} | {metrics['mean_reward']:11.4f} | {rolling:7.4f} | "
            f"{metrics['mean_advantage']:8.4f} | {metrics['mean_kl']:7.4f} | "
            f"{metrics['clip_fraction']:9.4f} | {metrics['loss']:.4f}"
        )


def main() -> None:
    args = _build_arg_parser().parse_args()
    _run(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_run_bamboogle_grpo_train.py::test_help_runs_without_torch -v`
Expected: PASS (`SystemExit(0)` from argparse `--help`).

- [ ] **Step 5: Run the full helper test file**

Run: `pytest tests/unit/test_run_bamboogle_grpo_train.py -v`
Expected: PASS (all tests from Task 1 + the `--help` smoke test).

- [ ] **Step 6: Commit**

```bash
git add examples/run_bamboogle_grpo_train.py tests/unit/test_run_bamboogle_grpo_train.py
git commit -m "feat(grpo): CLI + training loop for simulated-judge GRPO demo

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

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

## Self-Review

**1. Spec coverage:**
- Script `run_bamboogle_grpo_train.py` (Approach A, plain-generation GRPO) → Task 1 + Task 2. ✓
- `make_judge_fn` / `cycle_prompt_batches` helpers → Task 1. ✓
- Lazy imports so `--help` stays light → Task 2 (imports inside `_run`) + `test_help_runs_without_torch`. ✓
- CLI flags & defaults table → Task 2 `_build_arg_parser` (all 11 flags present with spec defaults). ✓
- Per-step diagnostics + rolling mean → Task 2 `_run` print loop. ✓
- Success criteria 1–5 → Task 3 manual verification (+ `mean_kl > 0` proxy for weights-changed). ✓
- Fast tests with no model load → Task 1 tests (pure helpers, stub judge). ✓
- "Reward hacks form" risk documented → Task 2 module docstring. ✓

**2. Placeholder scan:** No TBD/TODO; all code shown in full. Task 3 Step 3 is explicitly optional and marked do-not-commit. ✓

**3. Type consistency:** `make_judge_fn(judge)` and `cycle_prompt_batches(prompts, steps, batch_size)` signatures identical across Task 1 (definition), its tests, and Task 2 (call sites). `LLMGRPOTrainer.from_pretrained` / `.step` / `LLMGRPOConfig` kwargs match the verified source interface. Metric keys (`mean_reward`, `mean_advantage`, `mean_kl`, `clip_fraction`, `loss`) match `llm_grpo_trainer.py:459-465`. ✓
