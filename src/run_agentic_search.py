"""Entry point for all agentic search flows.

Three modes are supported, selected via --mode:

  single   One-shot generation — no search, no tool calls.
  search   Deep-research loop (SearchAgentLoop): plan → adaptive decision →
           subquestions → parallel search → fetch → evaluate → answer.
  tool     Tool-calling loop (ToolAgentLoop): model emits structured tool calls
           that are executed in parallel and injected back.

Server managers
---------------
Two server-manager implementations are provided:

  VLLMServerManager   Calls any OpenAI-compatible completions endpoint
                      (vLLM, Ollama, LiteLLM, …).  Decodes prompt token IDs
                      back to text, sends the text prompt, then re-tokenises
                      the completion.  Works out of the box with vLLM's
                      default serving mode.

  LocalServerManager  Loads a HuggingFace model in-process with greedy or
                      sampling decoding.  Useful for offline testing without
                      a separate inference server.

Quick start
-----------
# With a running vLLM server and Google search server:
python3 -m src.run_agentic_search \\
    --mode search \\
    --question "Compare dense vs sparse retrieval" \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --vllm_url http://localhost:8080 \\
    --search_url http://localhost:8000/retrieve

# Local model, single-turn:
python3 -m src.run_agentic_search \\
    --mode single \\
    --question "What is FAISS?" \\
    --model BAAI/bge-base-en-v1.5 \\
    --local
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Server managers
# ---------------------------------------------------------------------------

class VLLMServerManager:
    """Calls an OpenAI-compatible /v1/completions endpoint.

    The manager accepts prompt token IDs, decodes them to text with the
    tokenizer, sends a completion request, then encodes the returned text
    back to token IDs so the agent loop can parse XML tags from the result.
    """

    def __init__(
        self,
        tokenizer: Any,
        base_url: str,
        model: str,
        timeout_seconds: int = 120,
    ) -> None:
        self.tokenizer = tokenizer
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
    ) -> list[int]:
        import aiohttp

        prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        payload = {
            "model": self.model,
            "prompt": prompt_text,
            "max_tokens": sampling_params.get("max_tokens", 512),
            "temperature": sampling_params.get("temperature", 0.7),
            "top_p": sampling_params.get("top_p", 1.0),
            "stop": sampling_params.get("stop", None),
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.base_url}/v1/completions", json=payload
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        completion_text = data["choices"][0]["text"]
        return list(self.tokenizer.encode(completion_text))


class LocalServerManager:
    """Runs generation in-process using a loaded HuggingFace model.

    Intended for offline testing.  Loads the model lazily on the first call.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        use_fp16: bool = False,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.use_fp16 = use_fp16
        self._model: Any = None
        self._tokenizer: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        logger.info("Loading model %s onto %s …", self.model_path, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self._model.eval()
        self._model.to(self.device)
        if self.use_fp16 and self.device.startswith("cuda"):
            self._model = self._model.half()

    async def generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
    ) -> list[int]:
        import torch

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._generate_sync, prompt_ids, sampling_params
        )

    def _generate_sync(
        self, prompt_ids: list[int], sampling_params: dict[str, Any]
    ) -> list[int]:
        import torch

        self._ensure_loaded()
        max_new = sampling_params.get("max_tokens", 512)
        temp = float(sampling_params.get("temperature", 0.7))
        inputs = torch.tensor([prompt_ids], dtype=torch.long).to(self.device)
        with torch.no_grad():
            out = self._model.generate(
                inputs,
                max_new_tokens=max_new,
                do_sample=temp > 0,
                temperature=temp if temp > 0 else 1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return out[0][len(prompt_ids):].tolist()


# ---------------------------------------------------------------------------
# Flow runners
# ---------------------------------------------------------------------------

async def run_single_turn(
    tokenizer: Any,
    server_manager: Any,
    question: str,
    sampling_params: dict[str, Any],
    max_tokens: int,
) -> None:
    from src.agent_loop import SingleTurnAgentLoop, AgentLoopConfig

    loop = SingleTurnAgentLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        config=AgentLoopConfig(response_length=max_tokens),
    )
    output = await loop.run(
        messages=[{"role": "user", "content": question}],
        sampling_params=sampling_params,
    )
    answer = tokenizer.decode(output.response_ids, skip_special_tokens=True)
    _print_result(answer=answer, output=output, rounds=None)


async def run_search_agent(
    tokenizer: Any,
    server_manager: Any,
    question: str,
    sampling_params: dict[str, Any] | None = None,
    *,
    search_url: str = "http://localhost:8000/retrieve",
    topk: int = 5,
    max_turns: int = 8,
    max_search_limit: int = 0,
    require_evidence: bool = True,
    max_answer_rejections: int = 3,
    allow_internal_knowledge: bool = True,
    intent_pipeline: Any | None = None,
    intent_min_confidence: float = 0.6,
) -> None:
    """Run the deep-research SearchAgentLoop and print results.

    Can be called directly as a library function or via the CLI.

    Example::

        from src.run_agentic_search import run_search_agent, VLLMServerManager
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        server_manager = VLLMServerManager(tokenizer, "http://localhost:8080", "meta-llama/Llama-3.1-8B-Instruct")
        await run_search_agent(
            tokenizer, server_manager,
            question="Compare dense vs sparse retrieval.",
            sampling_params={"temperature": 0.7},
            search_url="http://localhost:8000/retrieve",
            topk=5,
            max_turns=8,
        )
    """
    from src.agent_loop import (
        SearchAgentLoop,
        SearchAgentLoopConfig,
        SearchEvaluationConfig,
    )

    sampling_params = sampling_params or {"temperature": 0.7, "max_tokens": 512}
    effective_search_limit = max_search_limit or max_turns
    if intent_pipeline is not None:
        resolved_topk, effective_search_limit, require_evidence, allow_internal_knowledge, intent_metadata = (
            intent_pipeline.resolve_search_settings(
                question,
                topk=topk,
                max_search_limit=effective_search_limit,
                require_evidence=require_evidence,
                allow_internal_knowledge=allow_internal_knowledge,
                min_confidence=intent_min_confidence,
            )
        )
        resolved_topk = resolved_topk
    else:
        resolved_topk = topk
        intent_metadata: dict[str, Any] = {"intent_routing_used": False}
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        search_config=SearchAgentLoopConfig(
            search_url=search_url,
            topk=resolved_topk,
            max_turns=max_turns,
            max_search_limit=effective_search_limit,
            require_sufficient_evidence_before_answer=require_evidence,
            max_answer_rejections=max_answer_rejections,
            allow_internal_knowledge_answer=allow_internal_knowledge,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1,
                min_total_results=2,
                min_content_length=10,
            ),
        ),
    )
    t0 = time.perf_counter()
    output = await loop.run(
        messages=[{"role": "user", "content": question}],
        sampling_params=sampling_params,
    )
    elapsed = time.perf_counter() - t0

    answer = tokenizer.decode(output.response_ids, skip_special_tokens=True)
    _print_result(
        answer=answer,
        output=output,
        rounds=output.context,
        elapsed=elapsed,
        intent_metadata=intent_metadata if intent_metadata.get("intent_routing_used") else None,
    )


async def run_tool_agent(
    tokenizer: Any,
    server_manager: Any,
    question: str,
    sampling_params: dict[str, Any] | None = None,
    *,
    search_url: str = "http://localhost:8000/retrieve",
    topk: int = 5,
    max_turns: int = 8,
    tool_format: str = "hermes",
) -> None:
    """Run the ToolAgentLoop with a built-in search tool and print results.

    Can be called directly as a library function or via the CLI.
    """
    from src.agent_loop import (
        FunctionTool,
        ToolAgentLoop,
        ToolAgentLoopConfig,
    )
    import aiohttp

    sampling_params = sampling_params or {"temperature": 0.7, "max_tokens": 512}

    async def search(query: str) -> str:
        """Search the retrieval server and return a text summary of results."""
        payload = {"queries": [query], "topk": topk}
        timeout = aiohttp.ClientTimeout(total=10)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(search_url, json=payload) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
            rows = data.get("result", [[]])[0]
            if not rows:
                return "No results found."
            parts = []
            for i, item in enumerate(rows, 1):
                doc = item.get("document", item)
                parts.append(f"[{i}] {doc.get('contents', '')}")
            return "\n".join(parts)
        except Exception as exc:
            return f"Search failed: {exc}"

    search_tool = FunctionTool(
        fn=search,
        name="search",
        description="Search for information on a topic.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    )

    loop = ToolAgentLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        tools=[search_tool],
        config=ToolAgentLoopConfig(
            tool_parser_format=tool_format,
            max_assistant_turns=max_turns,
            max_parallel_calls=4,
        ),
    )
    t0 = time.perf_counter()
    output = await loop.run(
        messages=[{"role": "user", "content": question}],
        sampling_params=sampling_params,
    )
    elapsed = time.perf_counter() - t0

    answer = tokenizer.decode(output.response_ids, skip_special_tokens=True)
    _print_result(answer=answer, output=output, rounds=None, elapsed=elapsed)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _print_result(
    answer: str,
    output: Any,
    rounds: Any = None,
    elapsed: float | None = None,
    intent_metadata: dict[str, Any] | None = None,
) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("ANSWER")
    print(sep)
    print(answer.strip())

    if rounds is not None and rounds.num_searches > 0:
        print(f"\n{sep}")
        print("SEARCH TRACE")
        print(sep)
        for r_idx, round_ctxs in enumerate(rounds.rounds, 1):
            print(f"  Round {r_idx}:")
            for ctx in round_ctxs:
                task = f"[{ctx.task_id}] " if ctx.task_id else ""
                print(f"    {task}{ctx.query!r}  → {len(ctx.results)} result(s)")
        if rounds.tasks:
            print(f"  Subquestions: {rounds.tasks}")

    print(f"\n{sep}")
    print("METRICS")
    print(sep)
    for key, val in output.metrics.items():
        if not key.startswith("build_prompt") and not key.startswith("generate_turn"):
            print(f"  {key:<30} {val}")
    if elapsed is not None:
        print(f"  {'wall_time_seconds':<30} {elapsed:.2f}")
    if intent_metadata:
        print(f"\n{sep}")
        print("INTENT ROUTING")
        print(sep)
        for key, val in intent_metadata.items():
            print(f"  {key:<30} {val}")
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an agentic search flow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core
    parser.add_argument("--question", type=str, required=True, help="Research question")
    parser.add_argument(
        "--mode",
        choices=["single", "search", "tool"],
        default="search",
        help="Agent loop to use",
    )

    # Model
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--local", action="store_true", help="Run model locally (no vLLM server)")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8080", help="vLLM base URL")
    parser.add_argument("--device", type=str, default="cpu", help="Device for local model")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 for local model")

    # Search
    parser.add_argument("--search_url", type=str, default="http://localhost:8000/retrieve")
    parser.add_argument("--topk", type=int, default=5)

    # Loop tuning
    parser.add_argument("--max_turns", type=int, default=8)
    parser.add_argument("--max_search_limit", type=int, default=0, help="0 = same as max_turns")
    parser.add_argument("--max_answer_rejections", type=int, default=3)
    parser.add_argument("--no_evidence_gate", action="store_true", help="Allow answer without sufficient evidence")
    parser.add_argument("--require_search", action="store_true", help="Disable internal-knowledge direct answers")
    parser.add_argument("--intent_examples", type=str, default=None, help="JSON file of intent-labeled examples")
    parser.add_argument("--intent_min_confidence", type=float, default=0.6, help="Minimum confidence for intent-based routing")
    parser.add_argument("--tool_format", choices=["hermes", "llama3", "json"], default="hermes")

    # Sampling
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--top_p", type=float, default=1.0)

    # Misc
    parser.add_argument("--verbose", action="store_true")
    return parser


async def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Load tokenizer
    from transformers import AutoTokenizer
    logger.info("Loading tokenizer %s …", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Build server manager
    if args.local:
        server_manager = LocalServerManager(
            model_path=args.model,
            device=args.device,
            use_fp16=args.fp16,
        )
    else:
        server_manager = VLLMServerManager(
            tokenizer=tokenizer,
            base_url=args.vllm_url,
            model=args.model,
        )

    sampling_params = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
    }

    intent_pipeline = None
    if args.intent_examples:
        from src.agent_loop.intent_classifier import IntentPipeline, load_training_data

        training_data = load_training_data(args.intent_examples)
        if training_data:
            intent_pipeline = IntentPipeline()
            intent_pipeline.train(training_data, epochs=10)

    print(f"\nMode    : {args.mode}")
    print(f"Model   : {args.model}")
    print(f"Question: {args.question}\n")

    if args.mode == "single":
        await run_single_turn(
            tokenizer, server_manager, args.question, sampling_params, args.max_tokens
        )
    elif args.mode == "search":
        await run_search_agent(
            tokenizer, server_manager, args.question, sampling_params,
            search_url=args.search_url,
            topk=args.topk,
            max_turns=args.max_turns,
            max_search_limit=args.max_search_limit,
            require_evidence=not args.no_evidence_gate,
            max_answer_rejections=args.max_answer_rejections,
            allow_internal_knowledge=not args.require_search,
            intent_pipeline=intent_pipeline,
            intent_min_confidence=args.intent_min_confidence,
        )
    elif args.mode == "tool":
        await run_tool_agent(
            tokenizer, server_manager, args.question, sampling_params,
            search_url=args.search_url,
            topk=args.topk,
            max_turns=args.max_turns,
            tool_format=args.tool_format,
        )


if __name__ == "__main__":
    asyncio.run(main())
