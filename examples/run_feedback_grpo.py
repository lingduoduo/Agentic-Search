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
    from examples._grpo_common import run_feedback_grpo_step

    print(f"Loading feedback examples from {args.db_path!r} …")
    print("Running rollouts and gradient step …")
    metrics = await run_feedback_grpo_step(
        model_path=args.model,
        db_path=args.db_path,
        output_dir=args.output_dir,
        min_ratings=args.min_ratings,
        human_feedback_weight=args.human_feedback_weight,
        num_rollouts=args.num_rollouts,
        search_url=args.search_url,
        device=args.device,
    )
    print("  Metrics:", metrics)


def main() -> None:
    args = _parse_args()
    asyncio.run(_train(args))


if __name__ == "__main__":
    main()
