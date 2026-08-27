"""Shared GRPO step for the feedback-driven example trainers.

run_feedback_grpo.py and run_sft_grpo.py (Phase 2) both run the same on-policy
GRPO step over thumbs-up/down feedback examples; this is that step.
"""

from __future__ import annotations

from pathlib import Path


async def run_feedback_grpo_step(
    *,
    model_path: str,
    db_path: str,
    output_dir: str | Path,
    min_ratings: int,
    human_feedback_weight: float,
    num_rollouts: int,
    search_url: str,
    device: str,
) -> dict:
    """Load feedback examples, run one GRPO step from ``model_path``, save a
    checkpoint to ``output_dir``, and return the step metrics."""
    from src.agents.search import SearchAgentLoop
    from src.model.post_training.data import load_feedback_examples
    from src.model.post_training.grpo.algorithms import GRPOAdvantageConfig
    from src.model.post_training.grpo.trainers import (
        LLMGRPOConfig,
        SearchAgentGRPOTrainer,
    )
    from src.model.post_training.reward import (
        SearchRewardConfig,
        SearchRewardFunction,
        simple_sparse_correctness_reward,
    )

    feedback_examples = load_feedback_examples(db_path, min_ratings=min_ratings)
    print(f"  {len(feedback_examples)} rated sessions loaded")

    prompts = [ex.question for ex in feedback_examples]
    ground_truths = [ex.ground_truth for ex in feedback_examples]
    metadata = [dict(ex.metadata) for ex in feedback_examples]

    reward_fn = SearchRewardFunction(
        SearchRewardConfig(
            human_feedback_weight=human_feedback_weight,
            correctness_weight=0.0,
        )
    )

    def loop_factory():
        return SearchAgentLoop(search_url=search_url)

    trainer = SearchAgentGRPOTrainer.from_pretrained(
        model_path,
        judge_fn=simple_sparse_correctness_reward,
        loop_factory=loop_factory,
        reward_fn=reward_fn,
        config=LLMGRPOConfig(num_rollouts=num_rollouts),
        advantage_config=GRPOAdvantageConfig(),
        device=device,
    )

    metrics = await trainer.step_async(prompts, ground_truths, metadata=metadata)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.policy.save_pretrained(out)
    trainer.tokenizer.save_pretrained(out)
    print(f"  GRPO checkpoint saved to {out}")
    return metrics
