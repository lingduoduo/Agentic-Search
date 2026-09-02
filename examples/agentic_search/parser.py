"""Argparse surface for ``examples.run_agentic_search``.

Split out because 135 lines of flag declarations sit between a reader and the
flow they came to read; nothing here decides behaviour.
"""

from __future__ import annotations

import argparse


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
        help="Low-latency model for search / lookup intents when --model_routing intent is enabled",
    )
    parser.add_argument(
        "--balanced_model",
        type=str,
        default=None,
        help="Medium model for chat / synthesis intents when --model_routing intent is enabled",
    )
    parser.add_argument(
        "--reasoning_model",
        type=str,
        default=None,
        help="Larger model for tool / action intents when --model_routing intent is enabled",
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
        "--intent_index",
        type=str,
        default=None,
        help="Directory holding a canonical-example intent index (index.npz), "
        "built with `python -m src.model.pre_training.intents.cli build`. Nothing is "
        "trained; routing compares the question against curated examples.",
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
