"""Entry point for all agentic search flows.

Three modes are supported, selected via --mode:

  single   One-shot generation — no search, no tool calls.
  search   SearchAgentLoop: plan → adaptive decision →
           subquestions → parallel search → fetch → evaluate → answer.
  tool     Tool-calling loop (ToolAgentLoop): model emits structured tool calls
           that are executed in parallel and injected back.

Server managers
---------------
Two server-manager implementations are provided:

  OpenAIServerManager Calls any OpenAI-compatible completions endpoint
                      (mlx-lm, vLLM, Ollama, LiteLLM, …).  Decodes prompt
                      token IDs back to text, sends the text prompt, then
                      re-tokenises the completion.

  LocalServerManager  Loads a HuggingFace model in-process with greedy or
                      sampling decoding.  Useful for offline testing without
                      a separate inference server.

Quick start
-----------
# With a running OpenAI-compatible server (e.g. mlx-lm, vLLM) and Google search server:
python3 -m examples.run_agentic_search \\
    --mode search \\
    --question "Compare dense vs sparse retrieval" \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --server_url http://localhost:8080 \\
    --search_url http://localhost:8000/retrieve

# Local model, single-turn:
python3 -m examples.run_agentic_search \\
    --mode single \\
    --question "What is FAISS?" \\
    --model Qwen/Qwen2.5-1.5B-Instruct \\
    --local --device cpu --max_tokens 256
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from src.model.serving import (  # re-export for back-compat
    OpenAIServerManager as OpenAIServerManager,
    LocalServerManager as LocalServerManager,
    build_server_manager as build_server_manager,
    _has_accelerate as _has_accelerate,
    _parse_major_minor as _parse_major_minor,
    _resolve_local_device as _resolve_local_device,
    _validate_local_runtime_device as _validate_local_runtime_device,
    _validate_local_runtime_stack as _validate_local_runtime_stack,
    _validate_local_generation_config as _validate_local_generation_config,
)

logger = logging.getLogger(__name__)


def _friendly_model_load_error(model: str, exc: Exception) -> str | None:
    """Return a concise user-facing message for common HF model-load failures."""

    message = str(exc)
    lowered = message.lower()

    if (
        "gated repo" in lowered
        or "cannot access gated repo" in lowered
        or "401 client error" in lowered
    ):
        return (
            f"Cannot load model '{model}' because it is gated on Hugging Face.\n"
            "You need access to that repo and a logged-in Hugging Face token on this machine.\n"
            "Try `huggingface-cli login` after requesting access, or use an open model such as "
            "`google/gemma-2-2b-it`, `Qwen/Qwen2.5-1.5B-Instruct`, or another ungated instruct model."
        )

    if "repository not found" in lowered or "404 client error" in lowered:
        return (
            f"Cannot load model '{model}' because Hugging Face could not find that repo.\n"
            "Check the model id spelling, or point `--model` to a local path that already contains the files."
        )

    if (
        "couldn't connect" in lowered
        or "cannot find the requested files in the disk cache" in lowered
    ):
        return (
            f"Cannot load model '{model}' from the local Hugging Face cache.\n"
            "In `--local` mode this CLI now prefers cached files to avoid slow or hanging network metadata lookups.\n"
            "If the model is not cached yet, rerun with `--allow_remote_model_downloads`, or download it first."
        )

    return None


def _load_tokenizer_for_cli(
    model: str,
    *,
    local: bool,
    allow_remote_model_downloads: bool,
) -> Any:
    """Load a tokenizer with the CLI's local-cache / gated-repo fallback policy."""

    from transformers import AutoTokenizer

    force_local = local and not allow_remote_model_downloads
    if force_local:
        print("Status  : loading tokenizer from local cache")
    try:
        return AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True,
            local_files_only=force_local,
        )
    except Exception as exc:
        # If the Hub fetch failed due to gating/auth and we haven't already
        # tried local-only, retry from the local cache (the model may already
        # be cached by vLLM or a prior download even though we have no token).
        if not force_local and _friendly_model_load_error(model, exc) is not None:
            try:
                print("Status  : loading tokenizer from local cache (Hub gated)")
                return AutoTokenizer.from_pretrained(
                    model,
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except Exception as local_exc:
                friendly = _friendly_model_load_error(model, local_exc)
                if friendly is not None:
                    raise RuntimeError(friendly) from local_exc
                raise RuntimeError(
                    f"Tokenizer for '{model}' is not in the local cache and the Hub is inaccessible "
                    "(model is gated). Use `huggingface-cli login`, pass a local model directory, "
                    "or switch to an ungated model."
                ) from local_exc

        friendly = _friendly_model_load_error(model, exc)
        if friendly is not None:
            raise RuntimeError(friendly) from exc
        raise


def _build_sampling_params(args: argparse.Namespace) -> dict[str, Any]:
    """Normalize generation-related CLI flags into one sampling-params dict."""

    return {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
    }


@dataclass(frozen=True)
class ModelRouteDecision:
    """Selected generation model for one CLI request."""

    model: str
    route: str
    reason: str
    metadata: dict[str, Any]


def _resolve_model_route(
    args: argparse.Namespace,
    intent_prediction: Any | None = None,
) -> ModelRouteDecision:
    """Choose a request-level generation model without touching agent loops.

    The selected model is still passed through the existing tokenizer and
    server-manager path.  This is deliberately request-level routing; per-turn
    model routing would require a multi-backend server manager.
    """

    metadata: dict[str, Any] = {
        "model_routing": args.model_routing,
        "base_model": args.model,
    }
    if args.model_routing == "off":
        return ModelRouteDecision(
            model=args.model,
            route="base",
            reason="model routing disabled",
            metadata=metadata,
        )

    if intent_prediction is None:
        metadata["model_routing_applied"] = False
        return ModelRouteDecision(
            model=args.model,
            route="base",
            reason="no intent prediction available",
            metadata=metadata,
        )

    metadata.update(
        {
            "predicted_intent": intent_prediction.intent,
            "intent_confidence": intent_prediction.confidence,
        }
    )
    if intent_prediction.confidence < args.model_routing_min_confidence:
        metadata["model_routing_applied"] = False
        return ModelRouteDecision(
            model=args.model,
            route="base",
            reason="intent confidence below routing threshold",
            metadata=metadata,
        )

    route_by_intent = {
        "qa": "fast",
        "navigate": "fast",
        "recommendation": "balanced",
        "purchase": "reasoning",
    }
    route = route_by_intent.get(intent_prediction.intent, "base")
    model_by_route = {
        "base": args.model,
        "fast": args.fast_model or args.model,
        "balanced": args.balanced_model or args.model,
        "reasoning": args.reasoning_model or args.balanced_model or args.model,
    }
    model = model_by_route[route]
    metadata.update(
        {
            "model_routing_applied": model != args.model,
            "selected_route": route,
            "selected_model": model,
        }
    )
    return ModelRouteDecision(
        model=model,
        route=route,
        reason=f"intent={intent_prediction.intent}",
        metadata=metadata,
    )


def _build_server_manager(args: argparse.Namespace, tokenizer: Any) -> Any:
    """Create the appropriate generation backend for the current CLI mode."""

    if args.local:
        return build_server_manager(
            tokenizer,
            model=args.model,
            device=args.device,
            torch_dtype=args.dtype,
            allow_unsafe_mps=args.allow_unsafe_mps,
            local_files_only=not args.allow_remote_model_downloads,
            generation_timeout_seconds=args.generation_timeout_seconds,
            generation_heartbeat_seconds=args.generation_heartbeat_seconds,
        )
    return build_server_manager(
        tokenizer,
        server_url=args.server_url,
        model=args.model,
    )


def _handle_local_cli_value_error(exc: ValueError) -> bool:
    """Print friendly local-runtime errors. Returns True when handled."""

    message = str(exc)
    if "Local generation mode requires a generative language model" in message:
        print(f"Error   : {message}")
        print(
            "Hint    : Use a generative instruct model for local agent runs, or keep encoder models for retrieval and indexing only."
        )
        return True
    if "Local generation on Apple MPS is disabled by default" in message:
        print(f"Error   : {message}")
        print("Hint    : Retry with `--device cpu` for the stable path.")
        return True
    if (
        "Local MPS generation on macOS is blocked for the older runtime stack"
        in message
    ):
        print(f"Error   : {message}")
        print(
            "Hint    : Use `--device cpu` for the stable path, or upgrade torch and transformers first."
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Flow runners
# ---------------------------------------------------------------------------


async def run_single_turn(
    tokenizer: Any,
    server_manager: Any,
    question: str,
    sampling_params: dict[str, Any],
    max_tokens: int,
    *,
    search_url: str = "http://localhost:8000/retrieve",
    topk: int = 5,
) -> None:
    from src import (
        PlainGenerationLoopConfig,
        get_registered_agent_loop,
        resolve_agent_name,
    )

    del search_url, topk

    loop_cls = get_registered_agent_loop(resolve_agent_name("single"))
    loop = loop_cls(
        tokenizer=tokenizer,
        server_manager=server_manager,
        config=PlainGenerationLoopConfig(response_length=max_tokens),
    )
    messages = [{"role": "user", "content": question}]
    output = await loop.run(messages=messages, sampling_params=sampling_params)
    answer = output.final_answer or tokenizer.decode(
        output.response_ids, skip_special_tokens=True
    )
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
    intent_prediction: Any | None = None,
    intent_min_confidence: float = 0.6,
) -> None:
    """Run the SearchAgentLoop and print results.

    Can be called directly as a library function or via the CLI.

    Example::

        from examples.run_agentic_search import run_search_agent, OpenAIServerManager
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        server_manager = OpenAIServerManager(tokenizer, "http://localhost:8080", "meta-llama/Llama-3.1-8B-Instruct")
        await run_search_agent(
            tokenizer, server_manager,
            question="Compare dense vs sparse retrieval.",
            sampling_params={"temperature": 0.7},
            search_url="http://localhost:8000/retrieve",
            topk=5,
            max_turns=8,
        )
    """
    from src import (
        SearchAgentLoopConfig,
        SearchEvaluationConfig,
        get_registered_agent_loop,
        resolve_agent_name,
    )

    sampling_params = sampling_params or {"temperature": 0.7, "max_tokens": 512}
    effective_search_limit = max_search_limit or max_turns
    if intent_pipeline is not None or intent_prediction is not None:
        if intent_prediction is None:
            intent_prediction = intent_pipeline.predict_text(question)
        from src.model.intent_classifier import resolve_search_settings

        (
            resolved_topk,
            effective_search_limit,
            require_evidence,
            allow_internal_knowledge,
            intent_metadata,
        ) = resolve_search_settings(
            intent_prediction,
            topk=topk,
            max_search_limit=effective_search_limit,
            require_evidence=require_evidence,
            allow_internal_knowledge=allow_internal_knowledge,
            min_confidence=intent_min_confidence,
        )
    else:
        resolved_topk = topk
        intent_metadata: dict[str, Any] = {"intent_routing_used": False}
    loop_cls = get_registered_agent_loop(resolve_agent_name("search"))
    loop = loop_cls(
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
        intent_metadata=intent_metadata
        if intent_metadata.get("intent_routing_used")
        else None,
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
    tool_format: str = "json",
) -> None:
    """Run the ToolAgentLoop with a built-in search tool and print results.

    Can be called directly as a library function or via the CLI.
    """
    from src import (
        FunctionTool,
        ToolAgentLoopConfig,
        get_registered_agent_loop,
        resolve_agent_name,
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

    loop_cls = get_registered_agent_loop(resolve_agent_name("tool"))
    loop = loop_cls(
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
    parser.add_argument(
        "--model", type=str, required=True, help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--model_routing",
        choices=["off", "intent"],
        default="off",
        help="Route the generation model before running the loop",
    )
    parser.add_argument(
        "--fast_model",
        type=str,
        default=None,
        help="Low-latency model for simple QA / navigation when --model_routing intent is enabled",
    )
    parser.add_argument(
        "--balanced_model",
        type=str,
        default=None,
        help="Medium model for synthesis / recommendation when --model_routing intent is enabled",
    )
    parser.add_argument(
        "--reasoning_model",
        type=str,
        default=None,
        help="Larger model for high-stakes or complex intents when --model_routing intent is enabled",
    )
    parser.add_argument(
        "--model_routing_min_confidence",
        type=float,
        default=0.7,
        help="Minimum intent confidence required before switching models",
    )
    parser.add_argument(
        "--local", action="store_true", help="Run model locally (no vLLM server)"
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://localhost:8080",
        help="OpenAI-compatible server URL (mlx-lm, vLLM, Ollama, …)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for local model: auto, cpu, cuda, or mps",
    )
    parser.add_argument(
        "--allow_unsafe_mps",
        action="store_true",
        help="Allow local MPS generation on macOS even though it may segfault",
    )
    parser.add_argument(
        "--allow_remote_model_downloads",
        action="store_true",
        help="Allow `--local` model loading to query/download from Hugging Face instead of cache-only loading",
    )
    parser.add_argument(
        "--generation_timeout_seconds",
        type=float,
        default=120.0,
        help="Best-effort local generation timeout in seconds; pass 0 to disable",
    )
    parser.add_argument(
        "--generation_heartbeat_seconds",
        type=float,
        default=10.0,
        help="How often local generation prints a still-running heartbeat",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Model dtype for local inference: float32, bfloat16, or float16. Default: auto (bfloat16 on Apple Silicon CPU, float16 on CUDA/MPS, float32 elsewhere)",
    )

    # Search
    parser.add_argument(
        "--search_url", type=str, default="http://localhost:8000/retrieve"
    )
    parser.add_argument("--topk", type=int, default=5)

    # Loop tuning
    parser.add_argument("--max_turns", type=int, default=8)
    parser.add_argument(
        "--max_search_limit", type=int, default=0, help="0 = same as max_turns"
    )
    parser.add_argument("--max_answer_rejections", type=int, default=3)
    parser.add_argument(
        "--no_evidence_gate",
        action="store_true",
        help="Allow answer without sufficient evidence",
    )
    parser.add_argument(
        "--require_search",
        action="store_true",
        help="Disable internal-knowledge direct answers",
    )
    parser.add_argument(
        "--intent_model",
        type=str,
        default=None,
        help="Path to a pre-trained intent classifier (.pt file from src.model.intent_training.train_intent_classifier). "
        "Preferred over --intent_examples — loads instantly with no retraining.",
    )
    parser.add_argument(
        "--intent_examples",
        type=str,
        default=None,
        help="JSON file of intent-labeled examples for on-the-fly training. "
        "Use --intent_model instead when the model has been pre-trained.",
    )
    parser.add_argument(
        "--intent_min_confidence",
        type=float,
        default=0.6,
        help="Minimum confidence for intent-based routing",
    )
    parser.add_argument(
        "--tool_format", choices=["hermes", "llama3", "json"], default="json"
    )

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

    intent_pipeline = None
    intent_prediction = None
    if args.intent_model:
        # Fast path: load a pre-trained model saved by src.model.intent_training.train_intent_classifier.
        from src.model.intent_classifier import IntentPipeline

        print(f"Status  : loading intent model from {args.intent_model}")
        intent_pipeline = IntentPipeline.load(args.intent_model)
        print(
            f"Status  : intent model ready (vocab size {len(intent_pipeline._vocab.token2idx)})"
        )
    elif args.intent_examples:
        # Slow path: train from scratch on the fly (use --intent_model for production)
        from src.model.intent_classifier import IntentPipeline, load_training_data

        print(f"Status  : training intent classifier from {args.intent_examples}")
        training_data = load_training_data(args.intent_examples)
        if training_data:
            intent_pipeline = IntentPipeline()
            intent_pipeline.train(training_data, epochs=10)
            print("Status  : intent classifier ready")

    if intent_pipeline is not None:
        intent_prediction = intent_pipeline.predict_text(args.question)

    model_route = _resolve_model_route(args, intent_prediction)
    args.model = model_route.model

    print(f"\nMode    : {args.mode}")
    print(f"Model   : {args.model}")
    print(f"Question: {args.question}\n")
    if args.model_routing != "off":
        print(
            "Model route: "
            f"{model_route.route} ({model_route.reason}; {model_route.metadata})"
        )

    # Load tokenizer
    logger.info("Loading tokenizer %s …", args.model)
    try:
        tokenizer = _load_tokenizer_for_cli(
            args.model,
            local=args.local,
            allow_remote_model_downloads=args.allow_remote_model_downloads,
        )
    except ModuleNotFoundError as exc:
        print(
            f"Error   : missing Python dependency {exc.name!r}. "
            "Install the project requirements first, for example: "
            "`python3 -m pip install -r requirements.txt`."
        )
        return
    except RuntimeError as exc:
        print(f"Error   : {exc}")
        return

    server_manager = _build_server_manager(args, tokenizer)

    sampling_params = _build_sampling_params(args)

    try:
        if args.mode == "single":
            await run_single_turn(
                tokenizer,
                server_manager,
                args.question,
                sampling_params,
                args.max_tokens,
                search_url=args.search_url,
                topk=args.topk,
            )
        elif args.mode == "search":
            await run_search_agent(
                tokenizer,
                server_manager,
                args.question,
                sampling_params,
                search_url=args.search_url,
                topk=args.topk,
                max_turns=args.max_turns,
                max_search_limit=args.max_search_limit,
                require_evidence=not args.no_evidence_gate,
                max_answer_rejections=args.max_answer_rejections,
                allow_internal_knowledge=not args.require_search,
                intent_pipeline=intent_pipeline,
                intent_prediction=intent_prediction,
                intent_min_confidence=args.intent_min_confidence,
            )
        elif args.mode == "tool":
            await run_tool_agent(
                tokenizer,
                server_manager,
                args.question,
                sampling_params,
                search_url=args.search_url,
                topk=args.topk,
                max_turns=args.max_turns,
                tool_format=args.tool_format,
            )
    except ValueError as exc:
        if args.local and _handle_local_cli_value_error(exc):
            return
        raise
    finally:
        if hasattr(server_manager, "aclose"):
            await server_manager.aclose()


if __name__ == "__main__":
    asyncio.run(main())
