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
    python3 -m examples.run_bamboogle_simulated_grpo \\
        --model Qwen/Qwen2.5-0.5B-Instruct --device cpu \\
        --allow_remote_model_downloads --steps 10
"""

from __future__ import annotations

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
