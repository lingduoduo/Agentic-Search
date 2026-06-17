"""Two-phase training: SFT warm-start followed by GRPO with human feedback.

Phase 1 (SFT): imitate thumbs-up sessions + optional JSONL pairs, save checkpoint.
Phase 2 (GRPO): load SFT checkpoint, run on-policy rollouts with human feedback signal.

Skip Phase 1 with --sft_epochs 0 to run pure GRPO from a base model.

Usage::

    python3 -m examples.run_sft_grpo \\
      --db_path data/feedback.sqlite3 \\
      --model Qwen/Qwen2.5-0.5B-Instruct \\
      --sft_epochs 1 --min_ratings 1 --device cpu
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT warm-start + GRPO fine-tuning")
    p.add_argument(
        "--db_path",
        default=os.environ.get("AGENTIC_SEARCH_WEB_DB_PATH", ":memory:"),
        help="SQLite DB path",
    )
    p.add_argument("--jsonl_path", default=None, help="Optional JSONL SFT pairs file")
    p.add_argument("--model", required=True, help="HuggingFace model id or local path")
    p.add_argument("--sft_epochs", type=int, default=3, help="0 to skip SFT phase")
    p.add_argument(
        "--sft_output_dir",
        default="data/checkpoints/sft_warmstart/",
        help="Intermediate SFT checkpoint directory",
    )
    p.add_argument(
        "--grpo_output_dir",
        default="data/checkpoints/sft_grpo/",
        help="Final GRPO checkpoint directory",
    )
    p.add_argument(
        "--sft_lr",
        type=float,
        default=2e-5,
        help="Learning rate for SFT phase",
    )
    p.add_argument(
        "--min_ratings",
        type=int,
        default=1,
        help="Abort if fewer feedback/SFT examples found",
    )
    p.add_argument(
        "--human_feedback_weight",
        type=float,
        default=0.5,
        help="GRPO human signal reward weight",
    )
    p.add_argument(
        "--num_rollouts", type=int, default=4, help="Number of rollouts per GRPO prompt"
    )
    p.add_argument(
        "--search_url",
        default="http://localhost:8001/retrieve",
        help="Retrieval server URL",
    )
    p.add_argument(
        "--device", default="mps", choices=["cpu", "mps", "cuda"], help="Compute device"
    )
    return p.parse_args()


async def _train(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.agents.search import SearchAgentLoop
    from src.training.data import load_feedback_examples, load_sft_examples
    from src.training.grpo import GRPOAdvantageConfig
    from src.training.ppo.llm_grpo_trainer import LLMGRPOConfig
    from src.training.ppo.search_agent_grpo_trainer import SearchAgentGRPOTrainer
    from src.training.reward import SearchRewardConfig, SearchRewardFunction
    from src.training.reward import simple_sparse_correctness_reward
    from src.training.sft import SFTConfig, SFTTrainer

    device = torch.device(args.device)

    # ── Phase 1: SFT ─────────────────────────────────────────────────────────
    grpo_model_path = args.model  # default: start GRPO from base model

    if args.sft_epochs > 0:
        print("[Phase 1] Loading SFT examples …")
        sft_examples = load_sft_examples(
            args.db_path, args.jsonl_path, min_ratings=args.min_ratings
        )
        print(f"  {len(sft_examples)} SFT examples loaded")

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        policy = AutoModelForCausalLM.from_pretrained(args.model).to(device)
        sft_config = SFTConfig(epochs=args.sft_epochs, lr=args.sft_lr)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=sft_config.lr)

        trainer = SFTTrainer(
            policy,
            tokenizer,
            optimizer,
            sft_config,
            device=device,
        )
        history = trainer.train(sft_examples)
        print(
            f"  SFT complete. Final loss: {history[-1]:.4f}"
            if history
            else "  SFT complete."
        )
        trainer.save(args.sft_output_dir)
        print(f"  SFT checkpoint saved to {args.sft_output_dir}")
        grpo_model_path = args.sft_output_dir  # GRPO starts from SFT checkpoint

    # ── Phase 2: GRPO ────────────────────────────────────────────────────────
    print("[Phase 2] Loading feedback examples for GRPO …")
    feedback_examples = load_feedback_examples(
        args.db_path, min_ratings=args.min_ratings
    )
    print(f"  {len(feedback_examples)} rated sessions loaded")

    prompts = [ex.question for ex in feedback_examples]
    ground_truths = [ex.ground_truth for ex in feedback_examples]
    metadata = [dict(ex.metadata) for ex in feedback_examples]

    reward_fn = SearchRewardFunction(
        SearchRewardConfig(
            human_feedback_weight=args.human_feedback_weight,
            correctness_weight=0.0,
        )
    )

    def loop_factory():
        return SearchAgentLoop(search_url=args.search_url)

    grpo_trainer = SearchAgentGRPOTrainer.from_pretrained(
        grpo_model_path,
        judge_fn=simple_sparse_correctness_reward,
        loop_factory=loop_factory,
        reward_fn=reward_fn,
        config=LLMGRPOConfig(num_rollouts=args.num_rollouts),
        advantage_config=GRPOAdvantageConfig(),
        device=args.device,
    )

    print("  Running GRPO step …")
    metrics = await grpo_trainer.step_async(prompts, ground_truths, metadata=metadata)
    print(f"  GRPO metrics: {metrics}")

    grpo_output_dir = Path(args.grpo_output_dir)
    grpo_output_dir.mkdir(parents=True, exist_ok=True)
    grpo_trainer.policy.save_pretrained(grpo_output_dir)
    grpo_trainer.tokenizer.save_pretrained(grpo_output_dir)
    print(f"  GRPO checkpoint saved to {grpo_output_dir}")


def main() -> None:
    args = _parse_args()
    asyncio.run(_train(args))


if __name__ == "__main__":
    main()
