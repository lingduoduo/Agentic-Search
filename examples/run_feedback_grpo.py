"""Feedback-driven GRPO fine-tuning.

Loads thumbs-up/down signals from an AgenticSearchStore SQLite DB, runs
on-policy rollouts through SearchAgentLoop, and saves a checkpoint.

Usage::

    python3 -m examples.run_feedback_grpo \\
      --db_path data/feedback.sqlite3 \\
      --model Qwen/Qwen2.5-0.5B-Instruct \\
      --num_rollouts 2 --min_ratings 1 --device cpu
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feedback-driven GRPO fine-tuning")
    parser.add_argument(
        "--db_path",
        default=os.environ.get("AGENTIC_SEARCH_WEB_DB_PATH", ":memory:"),
        help="SQLite DB path (default: $AGENTIC_SEARCH_WEB_DB_PATH or :memory:)",
    )
    parser.add_argument("--model", required=True, help="HuggingFace model id or path")
    parser.add_argument(
        "--output_dir",
        default="data/checkpoints/feedback_grpo/",
        help="Checkpoint destination",
    )
    parser.add_argument(
        "--min_ratings",
        type=int,
        default=10,
        help="Abort if fewer rated sessions found",
    )
    parser.add_argument(
        "--human_feedback_weight",
        type=float,
        default=0.5,
        help="Weight for human feedback reward component",
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=4,
        help="G rollouts per prompt",
    )
    parser.add_argument(
        "--search_url",
        default="http://localhost:8001/retrieve",
        help="Retrieval server URL",
    )
    parser.add_argument(
        "--device",
        default="mps",
        choices=["cpu", "mps", "cuda"],
        help="Training device",
    )
    return parser.parse_args()


async def _train(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.agents.search import SearchAgentLoop
    from src.training.data import load_feedback_examples
    from src.training.grpo import GRPOAdvantageConfig, GRPOTrainerConfig
    from src.training.ppo.search_agent_grpo_trainer import SearchAgentGRPOTrainer
    from src.training.reward import SearchRewardConfig, SearchRewardFunction
    from src.training.reward import simple_sparse_correctness_reward

    print(f"Loading feedback examples from {args.db_path!r} …")
    examples = load_feedback_examples(args.db_path, min_ratings=args.min_ratings)
    print(f"  {len(examples)} rated sessions loaded")

    prompts = [ex.question for ex in examples]
    ground_truths = [ex.ground_truth for ex in examples]
    metadata = [dict(ex.metadata) for ex in examples]

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    policy = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    reward_config = SearchRewardConfig(
        human_feedback_weight=args.human_feedback_weight,
        correctness_weight=0.0,
    )
    reward_fn = SearchRewardFunction(reward_config)

    def loop_factory():
        return SearchAgentLoop(search_url=args.search_url)

    trainer = SearchAgentGRPOTrainer(
        policy=policy,
        tokenizer=tokenizer,
        loop_factory=loop_factory,
        judge_fn=simple_sparse_correctness_reward,
        reward_fn=reward_fn,
        config=GRPOTrainerConfig(num_rollouts=args.num_rollouts),
        advantage_config=GRPOAdvantageConfig(),
        device=device,
    )

    print("Running rollouts and gradient step …")
    metrics = await trainer.step_async(prompts, ground_truths, metadata=metadata)
    print("  Metrics:", metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Checkpoint saved to {output_dir}")


def main() -> None:
    args = _parse_args()
    asyncio.run(_train(args))


if __name__ == "__main__":
    main()
