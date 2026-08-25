"""End-to-end retriever-aware GRPO: train a policy to choose web vs vector-DB.

Wires the agent-framework GRPO pieces into one command:

  from_pretrained policy  →  on-policy SearchAgentLoop rollouts (web/vdb/rerank)
                          →  SearchRewardConfig.retriever_aware()  (web priced 5× vdb)
                          →  train_loop  (checkpoint/resume, step timeout+skip)
                          →  action_eval (baseline vs trained: fewer rounds @ = correctness)

Dependency note
---------------
This example requires the full stack merged into the working tree:
  - web/vdb retriever action in SearchAgentLoop      (PR #325)
  - train_loop + trainer save/load                    (PR #326)
  - retriever_aware() reward + action_eval            (PR #327)
`train_loop` lives in ``src.model.post_training.grpo.train_loop`` (PR #326); until that
merges, this script will not import.

Usage
-----
Start two retrieval backends first (vector-DB + web/cached-web), then::

    python3 -m examples.run_retriever_aware_grpo \\
      --model Qwen/Qwen2.5-1.5B-Instruct --device mps \\
      --search_url     http://localhost:8001/retrieve \\
      --web_search_url http://localhost:8002/retrieve \\
      --data data/qa.jsonl --steps 50 \\
      --ckpt_dir data/checkpoints/retriever_aware/

``--data`` is a JSONL of ``{"question": ..., "answer": ...}`` rows. With no
``--data`` a tiny built-in demo set is used so the wiring can be exercised.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

_DEMO_DATA = [
    {"question": "What is FAISS used for?", "answer": "similarity search"},
    {"question": "What does BM25 rank?", "answer": "documents"},
    {"question": "What is RRF in retrieval?", "answer": "reciprocal rank fusion"},
    {"question": "What does a cross-encoder reranker do?", "answer": "rescore"},
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retriever-aware GRPO training + eval")
    p.add_argument("--model", required=True, help="HuggingFace model id or local path")
    p.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    p.add_argument(
        "--search_url",
        default="http://localhost:8001/retrieve",
        help="Vector-DB retrieval backend",
    )
    p.add_argument(
        "--web_search_url",
        default=None,
        help="Web retrieval backend; omit to make web degrade to vector-DB",
    )
    p.add_argument("--data", default=None, help="JSONL of {question, answer} rows")
    p.add_argument("--steps", type=int, default=20, help="GRPO training steps")
    p.add_argument("--num_rollouts", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max_turns", type=int, default=6)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--ckpt_dir", default="data/checkpoints/retriever_aware/")
    p.add_argument("--ckpt_every", type=int, default=10)
    p.add_argument("--step_timeout_s", type=float, default=600.0)
    p.add_argument("--resume_from", default=None, help="Checkpoint dir to resume from")
    p.add_argument("--eval_limit", type=int, default=8, help="Eval questions per side")
    return p.parse_args()


def _load_data(path: str | None) -> tuple[list[str], list[str]]:
    if not path:
        return ([d["question"] for d in _DEMO_DATA], [d["answer"] for d in _DEMO_DATA])
    rows = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    return ([r["question"] for r in rows], [r["answer"] for r in rows])


class PolicyServerManager:
    """In-process generation backed by the trainer's *live* policy (on-policy).

    Implements the ``server_manager`` duck type expected by ``AgentLoopBase``:
    ``async generate(request_id, prompt_ids, sampling_params) -> list[int]``
    returning only the newly generated response tokens.
    """

    def __init__(self, policy: Any, tokenizer: Any, device: Any) -> None:
        self._policy = policy
        self._tokenizer = tokenizer
        self._device = device

    async def generate(self, request_id, prompt_ids, sampling_params) -> list[int]:
        del request_id
        return await asyncio.get_event_loop().run_in_executor(
            None, self._generate_sync, prompt_ids, sampling_params
        )

    def _generate_sync(self, prompt_ids, sampling_params) -> list[int]:
        import torch

        inputs = torch.tensor([list(prompt_ids)], device=self._device)
        temperature = float(sampling_params.get("temperature", 0.7))
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(sampling_params.get("max_tokens", 512)),
            "do_sample": temperature > 0.0,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if temperature > 0.0:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = float(sampling_params.get("top_p", 1.0))
        with torch.no_grad():
            out = self._policy.generate(inputs, **kwargs)
        # Return only the response tokens (strip the prompt prefix).
        return out[0, inputs.shape[1] :].tolist()


async def _eval_side(loop_factory, prompts, ground_truths, judge_fn) -> list[dict]:
    """Run one greedy rollout per eval prompt → action_eval samples."""
    samples: list[dict] = []
    for question, gt in zip(prompts, ground_truths):
        loop = loop_factory()
        output = await loop.run(
            [{"role": "user", "content": question}], {"temperature": 0.0}
        )
        samples.append(
            {
                "correctness": judge_fn(output.final_answer or "", gt),
                "metrics": output.metrics,
            }
        )
    return samples


async def _run(args: argparse.Namespace) -> None:
    import torch

    from src.agents.search import SearchAgentLoop, SearchAgentLoopConfig
    from src.model.post_training.eval.action_eval import (
        aggregate_action_metrics,
        compare_action_evals,
        format_comparison_table,
    )
    from src.model.post_training.grpo.trainers import (
        LLMGRPOConfig,
        SearchAgentGRPOTrainer,
    )
    from src.model.post_training.grpo.train_loop import TrainLoopConfig, train_loop
    from src.model.post_training.reward import (
        SearchRewardConfig,
        SearchRewardFunction,
        simple_sparse_correctness_reward,
    )

    prompts, ground_truths = _load_data(args.data)
    eval_prompts = prompts[: args.eval_limit]
    eval_gts = ground_truths[: args.eval_limit]
    device = torch.device(args.device)
    judge = simple_sparse_correctness_reward

    # Trainer owns the policy; build it first, then point rollouts at that policy.
    trainer = SearchAgentGRPOTrainer.from_pretrained(
        args.model,
        judge_fn=judge,
        loop_factory=lambda: None,  # replaced below once the policy exists
        reward_fn=SearchRewardFunction(SearchRewardConfig.retriever_aware()),
        config=LLMGRPOConfig(
            num_rollouts=args.num_rollouts, max_new_tokens=args.max_new_tokens
        ),
        lr=args.lr,
        device=args.device,
    )
    server = PolicyServerManager(trainer.policy, trainer.tokenizer, device)

    def loop_factory() -> SearchAgentLoop:
        return SearchAgentLoop(
            tokenizer=trainer.tokenizer,
            server_manager=server,
            search_config=SearchAgentLoopConfig(
                max_turns=args.max_turns,
                search_url=args.search_url,
                web_search_url=args.web_search_url,
            ),
        )

    trainer.loop_factory = loop_factory

    # ── Baseline eval (before training) ──────────────────────────────────────
    print(f"[eval] baseline over {len(eval_prompts)} questions …")
    baseline = aggregate_action_metrics(
        await _eval_side(loop_factory, eval_prompts, eval_gts, judge)
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"[train] {args.steps} GRPO steps (web priced 5× vdb) …")
    history = await train_loop(
        trainer,
        prompts,
        ground_truths,
        TrainLoopConfig(
            max_steps=args.steps,
            ckpt_dir=args.ckpt_dir,
            ckpt_every=args.ckpt_every,
            step_timeout_s=args.step_timeout_s,
        ),
        resume_from=args.resume_from,
        on_metrics=lambda m: print(
            f"  step {m['step']:>4}  loss={m.get('loss', float('nan')):.4f}  "
            f"reward={m.get('mean_reward', float('nan')):.4f}"
        ),
    )
    print(f"[train] {len(history)} steps completed")

    # ── Trained eval (after training) ────────────────────────────────────────
    print("[eval] trained policy …")
    trained = aggregate_action_metrics(
        await _eval_side(loop_factory, eval_prompts, eval_gts, judge)
    )

    print("\n" + format_comparison_table(compare_action_evals(baseline, trained)))


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
