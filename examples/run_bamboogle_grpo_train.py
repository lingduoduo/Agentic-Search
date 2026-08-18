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

    ``judge`` must expose ``score(answer, gold)``. It used to expose
    ``score(answer)`` and the ground truth was discarded here, which meant the
    training signal could not depend on correctness at all -- a fluent wrong
    answer outscored a terse right one. Reference-free judges are still usable
    via ``--judge simulated``; they are adapted at the call site instead.
    """

    def _judge_fn(pred: str, ground_truth: str) -> float:
        return float(judge.score(pred, ground_truth))

    return _judge_fn


def _build_judge_llm():
    """Build a judge LLM from the same GEN_AI_* variables the app uses.

    Returns None when no key is configured, which makes ``LLMJudge`` behave as
    the deterministic gold judge rather than failing — the no-network smoke
    path has to keep working.
    """
    import os

    if not (os.environ.get("GEN_AI_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        return None
    from src.internal.llm.interfaces import LLMConfig
    from src.internal.llm.providers import OpenAICompatibleLLM

    return OpenAICompatibleLLM(
        LLMConfig(
            model_provider=os.environ.get("GEN_AI_MODEL_PROVIDER", "openai"),
            model_name=os.environ.get("GEN_AI_MODEL_VERSION", "gpt-4o-mini"),
            api_key=os.environ.get("GEN_AI_API_KEY")
            or os.environ.get("OPENAI_API_KEY"),
            api_base=os.environ.get("GEN_AI_API_BASE"),
        )
    )


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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize a policy with GRPO against SimulatedPreferenceJudge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model id or path")
    parser.add_argument("--steps", type=int, default=10, help="GRPO update steps")
    parser.add_argument(
        "--num_rollouts", type=int, default=4, help="completions per prompt"
    )
    parser.add_argument("--batch_prompts", type=int, default=2, help="prompts per step")
    parser.add_argument(
        "--limit", type=int, default=8, help="Bamboogle prompts to load"
    )
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="cpu", help="cpu / mps / cuda")
    parser.add_argument("--allow_remote_model_downloads", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--judge",
        choices=("gold", "llm", "simulated"),
        default="gold",
        help="gold (default): deterministic reference-based scoring, no "
        "network. llm: LLM-as-judge over (answer, gold) via GEN_AI_*, falling "
        "back to gold per item on any parse or provider failure. simulated: "
        "the reference-free shape heuristic — ignores the gold entirely and is "
        "kept only for comparison.",
    )
    return parser


def _run(args: argparse.Namespace) -> None:
    import torch

    from src.training.eval.bamboogle import load_bamboogle
    from src.training.judge import (
        LLMJudge,
        SimulatedPreferenceJudge,
    )
    from src.training.rl.llm_grpo_trainer import LLMGRPOConfig, LLMGRPOTrainer

    torch.manual_seed(args.seed)

    examples = load_bamboogle(limit=args.limit)
    prompts = [ex["question"] for ex in examples]
    if not prompts:
        raise SystemExit("No Bamboogle prompts loaded; check --limit / network.")

    if args.judge == "simulated":
        # Reference-free: scores answer shape, ignores the gold. Kept reachable
        # for comparison against the old behaviour, never the default.
        simulated = SimulatedPreferenceJudge()
        judge_fn = lambda pred, _gold: float(simulated.score(pred))  # noqa: E731
    else:
        judge = LLMJudge(llm=_build_judge_llm() if args.judge == "llm" else None)
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

    batches = cycle_prompt_batches(
        prompts, steps=args.steps, batch_size=args.batch_prompts
    )
    reward_history: list[float] = []
    print(
        f"{'step':>4} | {'mean_reward':>11} | {'rolling':>7} | "
        f"{'mean_adv':>8} | {'mean_kl':>7} | {'clip_frac':>9} | loss"
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
