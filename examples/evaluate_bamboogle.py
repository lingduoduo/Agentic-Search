"""CLI: evaluate an agentic search agent on the Bamboogle benchmark.

Usage examples
--------------
Minimal (judge_fn only, 20 examples):
    python -m examples.evaluate_bamboogle \\
        --search_url http://localhost:8000/retrieve \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --vllm_url http://localhost:8080

With shaped reward (second-pass curriculum):
    python -m examples.evaluate_bamboogle \\
        --search_url http://localhost:8000/retrieve \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --vllm_url http://localhost:8080 \\
        --reward_preset second_pass \\
        --limit 125 \\
        --output results/bamboogle_second_pass.jsonl

Reward presets: sparse_final_only | simple_sparse | second_pass | third_pass
"""

from __future__ import annotations

import argparse
import sys

from src.training.eval.bamboogle import evaluate_bamboogle

_REWARD_PRESETS = {
    "sparse_final_only",
    "simple_sparse",
    "second_pass",
    "third_pass",
}


def _build_agent(args: argparse.Namespace):
    """Construct a BaseAgent-compatible agent from CLI arguments."""

    from src.agents.graph_base import BaseAgent, AgentState
    from src.agents.search import SearchAgentLoop
    from src.agents.base import AgentLoopConfig

    # Inline agent that wraps SearchAgentLoop.run() inside invoke().
    class _SearchAgent(BaseAgent):
        def run(self, state: AgentState) -> None:
            import asyncio
            from uuid import uuid4

            from src.internal.chat.queue_manager import AgentThought, QueueEvent

            tid = state["task_id"]
            question = state["messages"][0]["content"]

            loop = SearchAgentLoop(
                tokenizer=self._tokenizer,
                server_manager=self._server_manager,
                config=AgentLoopConfig(),
            )

            async def _run():
                return await loop.run(
                    messages=[{"role": "user", "content": question}],
                    sampling_params={
                        "temperature": 0.0,
                        "max_tokens": args.max_tokens,
                    },
                )

            output = asyncio.run(_run())
            self.queue_manager.publish(
                tid,
                AgentThought(
                    id=uuid4(),
                    task_id=tid,
                    event=QueueEvent.AGENT_MESSAGE,
                    answer=output.final_answer or "",
                ),
            )
            self.queue_manager.publish(
                tid,
                AgentThought(id=uuid4(), task_id=tid, event=QueueEvent.AGENT_END),
            )

    raise NotImplementedError(
        "Wire up your model server here and return a BaseAgent instance.\n"
        "See src/agents/graph_base.py for the BaseAgent interface."
    )


def _build_reward_fn(preset: str | None):
    if preset is None:
        return None
    from src.training.reward import SearchRewardFunction, SearchRewardConfig

    presets = {
        "sparse_final_only": SearchRewardConfig.sparse_final_only,
        "simple_sparse": SearchRewardConfig.simple_sparse_with_search_penalty,
        "second_pass": SearchRewardConfig.second_pass,
        "third_pass": SearchRewardConfig.third_pass_with_format,
    }
    if preset not in presets:
        print(f"Unknown preset '{preset}'. Choose from: {', '.join(presets)}")
        sys.exit(1)
    return SearchRewardFunction(presets[preset]())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an agentic search agent on the Bamboogle benchmark."
    )
    parser.add_argument("--search_url", default="http://localhost:8000/retrieve")
    parser.add_argument("--vllm_url", default="http://localhost:8080")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--reward_preset", choices=list(_REWARD_PRESETS), default=None)
    parser.add_argument("--output", default="bamboogle_results.jsonl")
    args = parser.parse_args()

    agent = _build_agent(args)
    reward_fn = _build_reward_fn(args.reward_preset)

    evaluate_bamboogle(
        agent,
        reward_fn=reward_fn,
        limit=args.limit,
        output_path=args.output,
        verbose=True,
    )


if __name__ == "__main__":
    main()
