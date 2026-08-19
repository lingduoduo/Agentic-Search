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

    from src.model.post_training.data import load_sft_examples
    from src.model.post_training.sft.trainer import SFTConfig, SFTTrainer

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
    from examples._grpo_common import run_feedback_grpo_step

    print("[Phase 2] Running GRPO over feedback examples …")
    metrics = await run_feedback_grpo_step(
        model_path=grpo_model_path,
        db_path=args.db_path,
        output_dir=args.grpo_output_dir,
        min_ratings=args.min_ratings,
        human_feedback_weight=args.human_feedback_weight,
        num_rollouts=args.num_rollouts,
        search_url=args.search_url,
        device=args.device,
    )
    print(f"  GRPO metrics: {metrics}")


def main() -> None:
    args = _parse_args()
    asyncio.run(_train(args))


if __name__ == "__main__":
    main()
