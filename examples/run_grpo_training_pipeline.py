"""Small runnable reference for PPO / GRPO training helpers.

This is intentionally deterministic and model-free. It exercises the public
reward and group-advantage helpers without starting a retrieval server or
loading an LLM. If PyTorch is installed, it also runs the trajectory packing
and PPO/GRPO policy-loss helpers.
"""

from __future__ import annotations

from typing import Any

from src.agent_loop import (
    AgentLoopOutput,
    SearchRewardConfig,
    SearchRewardFunction,
)
from src.agent_loop import simple_sparse_correctness_reward


GROUND_TRUTH = "John William Henry II"


def _output(answer: str, *, search_rounds: float, repeated: float = 0.0):
    return AgentLoopOutput(
        prompt_ids=[1, 2],
        response_ids=[3, 4, 5],
        response_mask=[1, 1, 1],
        num_turns=int(search_rounds) + 1,
        metrics={
            "rounds_used": search_rounds,
            "search_rounds": search_rounds,
            "repeated_search_queries": repeated,
            "subquestion_coverage_ratio": 1.0,
            "fetched_pages": 0.0,
            "unnecessary_fetch_count": 0.0,
            "answer_when_evidence_insufficient": 0.0,
            "search_budget_exhausted_without_answer": 0.0,
            "final_evidence_sufficient": 1.0,
            "answer_allowed": 1.0,
            "search_quality_score": 1.0,
        },
        action_trace=f"<answer>{answer}</answer>",
        final_answer=answer,
    )


def _outputs() -> list[AgentLoopOutput]:
    return [
        _output("John William Henry II", search_rounds=2),
        _output("John William Henry II was older.", search_rounds=3),
        _output("Jed Hoyer", search_rounds=4, repeated=1),
    ]


def _std_normalized_advantages(rewards: list[float]) -> list[float]:
    if len(rewards) <= 1:
        return [0.0] * len(rewards)
    mean = sum(rewards) / len(rewards)
    centered = [reward - mean for reward in rewards]
    std = (sum(value * value for value in centered) / len(centered)) ** 0.5
    return [value / (std + 1e-8) for value in centered]


def _score_with_optional_grpo_helpers(
    outputs: list[AgentLoopOutput],
    reward_fn: SearchRewardFunction,
) -> tuple[list[float], list[float], list[dict[str, float]], str]:
    try:
        from src.agent_loop import (
            GRPOAdvantageConfig,
            GRPORolloutSample,
            score_prompt_group,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        components = [
            reward_fn.reward_components(
                output,
                ground_truth=GROUND_TRUTH,
                judge_fn=simple_sparse_correctness_reward,
            )
            for output in outputs
        ]
        rewards = [component["total"] for component in components]
        return rewards, _std_normalized_advantages(rewards), components, "manual"

    samples = [
        GRPORolloutSample(
            group_id="question-1",
            rollout_index=index,
            sampling_params={"temperature": 0.7 + index * 0.1},
            output=output,
        )
        for index, output in enumerate(outputs)
    ]
    scored = score_prompt_group(
        samples,
        ground_truth=GROUND_TRUTH,
        judge_fn=simple_sparse_correctness_reward,
        reward_fn=reward_fn,
        advantage_config=GRPOAdvantageConfig.std_normalized(),
    )
    return (
        [rollout.reward for rollout in scored],
        [rollout.advantage for rollout in scored],
        [rollout.reward_components for rollout in scored],
        "src.agent_loop.score_prompt_group",
    )


def _optional_policy_loss() -> dict[str, Any]:
    try:
        from src.agent_loop import (
            compute_trajectory_policy_loss,
            trajectory_log_prob_pack,
        )
        from src.agent_loop.generation import RolloutTrajectory
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            return {"skipped": "torch is not installed"}
        raise

    trajectory = RolloutTrajectory(
        batch_index=0,
        prompt_token_ids=[1, 2],
        response_token_ids=[3, 4, 5],
        response_with_observation_mask=[1, 1, 1],
        trajectory_turns=1,
        steps=[],
        final_answer=GROUND_TRUTH,
        finished_without_answer=False,
        tokens=[1, 2, 3, 4, 5],
        attention_mask=[1, 1, 1, 1, 1],
        response_mask=[0, 0, 1, 1, 1],
        old_log_probs=[0.0, 0.0, -0.35, -0.3, -0.2],
    )
    pack = trajectory_log_prob_pack(trajectory)
    loss = compute_trajectory_policy_loss(
        new_log_probs=[0.0, 0.0, -0.31, -0.28, -0.18],
        old_log_probs=pack["old_log_probs"],
        advantages=[1.0] * len(pack["tokens"]),
        response_mask=pack["response_mask"],
        clip_epsilon=0.2,
        kl_beta=0.01,
    )
    return {
        "packed_lengths": {key: len(value) for key, value in pack.items()},
        "loss": loss,
    }


def run_demo() -> dict[str, Any]:
    reward_fn = SearchRewardFunction(
        SearchRewardConfig.simple_sparse_with_search_penalty(per_search_penalty=-0.02)
    )
    rewards, advantages, reward_components, scoring_backend = (
        _score_with_optional_grpo_helpers(_outputs(), reward_fn)
    )

    return {
        "rewards": rewards,
        "advantages": advantages,
        "reward_components": reward_components,
        "scoring_backend": scoring_backend,
        "policy_loss": _optional_policy_loss(),
    }


def main() -> None:
    result = run_demo()
    print("Rewards:", [round(value, 4) for value in result["rewards"]])
    print("Advantages:", [round(value, 4) for value in result["advantages"]])
    print("Scoring backend:", result["scoring_backend"])
    print("Policy loss:", result["policy_loss"])


if __name__ == "__main__":
    main()
