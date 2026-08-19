"""Generate synthetic AI-feedback on Bamboogle prompts and feed it to GRPO.

For each Bamboogle prompt this samples ``--num_rollouts`` agent answers, scores
them with the reference-free :class:`SimulatedPreferenceJudge` (a stand-in for
an LLM-as-judge), computes GRPO group-relative advantages via the existing
``score_prompt_group`` seam, dumps a synthetic-preference dataset to JSONL, and
prints a judge-vs-gold agreement report.

The judge is the only simulated piece: prompts are real Bamboogle questions and
answers are real model rollouts.  Swap ``SimulatedPreferenceJudge`` for a real
LLM judge behind the same ``as_batch_judge_fn()`` interface to go live.

Quick start (local CPU, self-contained, slow):
    python3 -m examples.run_bamboogle_synthetic_grpo \\
        --model Qwen/Qwen2.5-1.5B-Instruct --local \\
        --search_url http://localhost:8000/retrieve --num_rollouts 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.model.post_training.grpo.rollouts import ScoredGRPORollout
    from src.model.post_training.grpo.judge import SimulatedPreferenceJudge


def build_synthetic_record(
    prompt: str,
    gold: list[str],
    judge: SimulatedPreferenceJudge,
    scored: list[ScoredGRPORollout],
) -> dict[str, Any]:
    """Build one synthetic-preference JSONL record for a prompt group."""
    from src.model.post_training.eval.bamboogle import contains_match, exact_match

    rollouts = []
    for s in scored:
        answer = s.output.final_answer or ""
        rollouts.append(
            {
                "answer": answer,
                "judge_score": judge.score(answer),
                "reward": s.reward,
                "advantage": s.advantage,
                "exact_match": exact_match(answer, gold),
                "contains_match": contains_match(answer, gold),
            }
        )
    return {"prompt": prompt, "gold": gold, "rollouts": rollouts}


def _build_loop_factory(args: argparse.Namespace, tokenizer: Any):
    from examples.run_bamboogle_eval import _build_server_manager
    from src.agents.search import SearchAgentLoop, SearchAgentLoopConfig
    from src.agents.components.result_evaluation import SearchEvaluationConfig

    server_manager = _build_server_manager(args, tokenizer)

    def factory() -> SearchAgentLoop:
        return SearchAgentLoop(
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

    return factory, server_manager


async def _run(args: argparse.Namespace) -> None:
    from src.model.post_training.eval.bamboogle import load_bamboogle
    from src.model.post_training.grpo.rollouts import (
        sample_prompt_group,
        score_prompt_group,
    )
    from src.model.post_training.grpo.judge import (
        SimulatedPreferenceJudge,
        judge_gold_agreement,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=not args.allow_remote_model_downloads,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    judge = SimulatedPreferenceJudge()
    loop_factory, server_manager = _build_loop_factory(args, tokenizer)
    examples = load_bamboogle(limit=args.limit)

    pairs: list[tuple[float, bool]] = []
    out_path = Path(args.output)
    try:
        with out_path.open("w") as fh:
            for ex in examples:
                question = ex["question"]
                gold = ex.get("golden_answers") or ex.get("answers") or []
                samples = await sample_prompt_group(
                    loop_factory,
                    messages=[{"role": "user", "content": question}],
                    sampling_params={
                        "temperature": args.temperature,
                        "max_tokens": args.max_tokens,
                    },
                    num_rollouts=args.num_rollouts,
                )
                scored = score_prompt_group(
                    samples,
                    ground_truth=gold[0] if gold else "",
                    judge_fn=lambda pred, g: 0.0,
                    batch_judge_fn=judge.as_batch_judge_fn(),
                )
                record = build_synthetic_record(question, gold, judge, scored)
                fh.write(json.dumps(record) + "\n")
                for r in record["rollouts"]:
                    pairs.append((r["judge_score"], r["contains_match"] > 0))
                print(f"[{question[:60]}...] {len(scored)} rollouts scored")
    finally:
        await server_manager.aclose()

    report = judge_gold_agreement(pairs)
    print(f"\nSynthetic dataset written to {out_path}")
    print("Judge-vs-gold agreement:")
    for k, v in report.items():
        print(f"  {k:20s}: {v:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic AI-feedback on Bamboogle for GRPO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--local", action="store_true", help="Run model in-process")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow_unsafe_mps", action="store_true")
    parser.add_argument("--allow_remote_model_downloads", action="store_true")
    parser.add_argument("--server_url", default="http://localhost:8080")
    parser.add_argument("--search_url", default="http://localhost:8000/retrieve")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max_turns", type=int, default=8)
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=5, help="Number of prompts")
    parser.add_argument("--output", default="bamboogle_synthetic.jsonl")
    parser.add_argument("--generation_timeout_seconds", type=float, default=0.0)
    parser.add_argument("--generation_heartbeat_seconds", type=float, default=10.0)
    args = parser.parse_args()

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
