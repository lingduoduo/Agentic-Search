"""Evaluate the SearchAgentLoop on the Bamboogle two-hop QA benchmark.

Quick start (local CPU, no vLLM needed — slow but self-contained):
    python3 -m examples.run_bamboogle_eval --model Qwen/Qwen2.5-1.5B-Instruct --local

Server-backed (fast):
    python3 -m examples.run_bamboogle_eval \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --vllm_url http://localhost:8080 \\
        --search_url http://localhost:8000/retrieve \\
        --reward_preset second_pass \\
        --limit 125

Reward presets: sparse_final_only | simple_sparse | second_pass | third_pass
"""

from __future__ import annotations

import argparse
import asyncio
from types import SimpleNamespace
from typing import Any


def _build_server_manager(args: argparse.Namespace, tokenizer: Any) -> Any:
    from examples.run_agentic_search import LocalServerManager, VLLMServerManager

    if args.local:
        return LocalServerManager(
            model_path=args.model,
            device=args.device,
            allow_unsafe_mps=args.allow_unsafe_mps,
            generation_timeout_seconds=args.generation_timeout_seconds,
            generation_heartbeat_seconds=args.generation_heartbeat_seconds,
        )
    return VLLMServerManager(
        tokenizer=tokenizer, base_url=args.vllm_url, model=args.model
    )


def _build_agent(args: argparse.Namespace, tokenizer: Any, server_manager: Any) -> Any:
    from src.agents.search import SearchAgentLoop, SearchAgentLoopConfig
    from src.training.evaluation import SearchEvaluationConfig

    class _SyncAgent:
        def invoke(self, state: dict) -> SimpleNamespace:
            question = state["messages"][0]["content"]
            loop = SearchAgentLoop(
                tokenizer=tokenizer,
                server_manager=server_manager,
                search_config=SearchAgentLoopConfig(
                    search_url=args.search_url,
                    topk=args.topk,
                    max_turns=args.max_turns,
                    evaluation_config=SearchEvaluationConfig(
                        min_results_per_query=1,
                        min_total_results=2,
                        min_content_length=10,
                    ),
                ),
            )
            output = asyncio.run(
                loop.run(
                    [{"role": "user", "content": question}],
                    sampling_params={
                        "temperature": args.temperature,
                        "max_tokens": args.max_tokens,
                    },
                )
            )
            return SimpleNamespace(
                answer=output.final_answer or "",
                # Pass the real AgentLoopOutput so reward_fn gets full
                # search metadata (citations, rounds_used, fetch history).
                metadata={"loop_output": output},
            )

    return _SyncAgent()


def _build_reward_fn(preset: str | None) -> Any:
    if preset is None:
        return None
    from src.training.reward import SearchRewardConfig, SearchRewardFunction

    presets = {
        "sparse_final_only": SearchRewardConfig.sparse_final_only,
        "simple_sparse": SearchRewardConfig.simple_sparse_with_search_penalty,
        "second_pass": SearchRewardConfig.second_pass,
        "third_pass": SearchRewardConfig.third_pass_with_format,
    }
    return SearchRewardFunction(presets[preset]())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SearchAgentLoop on Bamboogle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument(
        "--local", action="store_true", help="Run model in-process (no vLLM)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for local inference: cpu, mps, cuda, or auto",
    )
    parser.add_argument(
        "--allow_unsafe_mps",
        action="store_true",
        help="Allow MPS generation on macOS (may segfault on old torch/transformers; safe on torch>=2.3)",
    )
    parser.add_argument("--vllm_url", default="http://localhost:8080")
    parser.add_argument("--search_url", default="http://localhost:8000/retrieve")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max_turns", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument(
        "--reward_preset",
        choices=["sparse_final_only", "simple_sparse", "second_pass", "third_pass"],
        default=None,
        help="SearchRewardFunction preset for shaped reward scoring alongside EM/contains",
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="Number of examples (max 125)"
    )
    parser.add_argument("--output", default="bamboogle_results.jsonl")
    parser.add_argument(
        "--generation_timeout_seconds",
        type=float,
        default=0.0,
        help="Local generation wall-clock timeout in seconds; 0 disables (default for benchmarks)",
    )
    parser.add_argument(
        "--generation_heartbeat_seconds",
        type=float,
        default=10.0,
        help="How often local generation prints a still-running heartbeat",
    )
    parser.add_argument(
        "--print_output",
        action="store_true",
        help="Print each question, raw prediction, and gold answers (useful for smoke-testing)",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    from src.training.eval.bamboogle import evaluate_bamboogle

    server_manager = _build_server_manager(args, tokenizer)
    agent = _build_agent(args, tokenizer, server_manager)
    reward_fn = _build_reward_fn(args.reward_preset)

    _summary, rows = evaluate_bamboogle(
        agent,
        reward_fn=reward_fn,
        limit=args.limit,
        output_path=args.output,
        verbose=True,
    )

    if args.print_output:
        print("\n--- per-example predictions ---")
        for r in rows:
            print(f"Q : {r.question}")
            print(f"A : {r.prediction!r}")
            print(f"Gold: {r.golden_answers}")
            print(f"EM={r.exact_match:.0f}  contains={r.contains_match:.0f}")
            print()


if __name__ == "__main__":
    main()
