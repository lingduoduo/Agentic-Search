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
    --model Qwen/Qwen2.5-1.5B-Instruct \\
    --local --device cpu --max_tokens 256
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import platform
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_GENERATIVE_MODEL_TYPES = {
    "aria_text",
    "bamba",
    "bart",
    "bert_generation",
    "big_bird_pegasus",
    "blenderbot",
    "blenderbot_small",
    "bloom",
    "code_llama",
    "cohere",
    "cohere2",
    "falcon",
    "gemma",
    "gemma2",
    "gpt2",
    "gpt_bigcode",
    "gpt_neo",
    "gpt_neox",
    "gptj",
    "granite",
    "granitemoe",
    "jamba",
    "jetmoe",
    "llama",
    "mamba",
    "mistral",
    "mixtral",
    "mllama",
    "mpt",
    "olmo",
    "olmo2",
    "opt",
    "persimmon",
    "phi",
    "phi3",
    "qwen2",
    "qwen2_moe",
    "recurrent_gemma",
    "smolvlm",
    "stablelm",
    "starcoder2",
    "xglm",
}


def _validate_local_generation_config(model_path: str, config: Any) -> None:
    """Raise a clear error when a local model cannot be used for generation."""

    model_type = str(getattr(config, "model_type", "") or "").lower()
    is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
    if model_type in _GENERATIVE_MODEL_TYPES or is_encoder_decoder:
        return

    architectures = getattr(config, "architectures", None) or []
    architecture_text = ", ".join(architectures) if architectures else "unknown architecture"
    raise ValueError(
        "Local generation mode requires a generative language model. "
        f"'{model_path}' looks like a non-generative encoder model "
        f"(model_type='{model_type}', architectures={architecture_text}). "
        "Use a chat or instruct causal LM such as Llama, Qwen, Mistral, or Gemma for "
        "`--mode single`, `--mode search`, or `--mode tool`."
    )


def _friendly_model_load_error(model: str, exc: Exception) -> str | None:
    """Return a concise user-facing message for common HF model-load failures."""

    message = str(exc)
    lowered = message.lower()

    if "gated repo" in lowered or "cannot access gated repo" in lowered or "401 client error" in lowered:
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

    if "couldn't connect" in lowered or "cannot find the requested files in the disk cache" in lowered:
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


def _build_server_manager(args: argparse.Namespace, tokenizer: Any) -> Any:
    """Create the appropriate generation backend for the current CLI mode."""

    if args.local:
        return LocalServerManager(
            model_path=args.model,
            device=args.device,
            torch_dtype=args.dtype,
            allow_unsafe_mps=args.allow_unsafe_mps,
            local_files_only=not args.allow_remote_model_downloads,
            generation_timeout_seconds=args.generation_timeout_seconds,
            generation_heartbeat_seconds=args.generation_heartbeat_seconds,
        )
    return VLLMServerManager(
        tokenizer=tokenizer,
        base_url=args.vllm_url,
        model=args.model,
    )


def _handle_local_cli_value_error(exc: ValueError) -> bool:
    """Print friendly local-runtime errors. Returns True when handled."""

    message = str(exc)
    if "Local generation mode requires a generative language model" in message:
        print(f"Error   : {message}")
        print("Hint    : Use a generative instruct model for local agent runs, or keep encoder models for retrieval and indexing only.")
        return True
    if "Local generation on Apple MPS is disabled by default" in message:
        print(f"Error   : {message}")
        print("Hint    : Retry with `--device cpu` for the stable path.")
        return True
    if "Local MPS generation on macOS is blocked for the older runtime stack" in message:
        print(f"Error   : {message}")
        print("Hint    : Use `--device cpu` for the stable path, or upgrade torch and transformers first.")
        return True
    return False


def _resolve_local_device(requested_device: str) -> str:
    """Resolve a local inference device, preferring accelerators when asked."""

    if requested_device != "auto":
        return requested_device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if (
        platform.system() != "Darwin"
        and getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def _has_accelerate() -> bool:
    try:
        import accelerate  # noqa: F401
    except ImportError:
        return False
    return True


def _transformers_supports_dtype_kwarg() -> bool:
    """Return True when the installed transformers uses 'dtype' instead of 'torch_dtype'."""
    try:
        import transformers
        major, minor = _parse_major_minor(transformers.__version__)
        return (major, minor) >= (4, 42)
    except Exception:
        return False


def _auto_select_dtype(device: str) -> str:
    """Choose the best inference dtype for the given device.

    Apple Silicon CPU supports bfloat16 natively — using it gives ~2-3x
    throughput vs float32 due to halved memory bandwidth and native BF16 ALUs.
    CUDA defaults to float16; everything else stays at float32.
    """
    if device.startswith("cuda"):
        return "float16"
    if device == "mps":
        return "float16"
    # Apple Silicon (arm64): prefer bfloat16 on CPU
    if platform.machine().lower() in ("arm64", "aarch64"):
        return "bfloat16"
    return "float32"


def _parse_major_minor(version_text: str) -> tuple[int, int]:
    core = version_text.split("+", 1)[0]
    parts = core.split(".")
    major = int(parts[0]) if parts and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return major, minor


def _validate_local_runtime_device(device: str, *, allow_unsafe_mps: bool = False) -> None:
    """Block known-unstable local runtime choices before native crashes happen."""

    if device == "mps" and not allow_unsafe_mps:
        raise ValueError(
            "Local generation on Apple MPS is disabled by default in this CLI because it can segfault with "
            "some Hugging Face causal LMs on macOS. Use `--device cpu` for the stable path, or pass "
            "`--allow_unsafe_mps` if you want to try MPS anyway."
        )


def _validate_local_runtime_stack(
    device: str,
    *,
    platform_system: str | None = None,
    torch_version: str | None = None,
    transformers_version: str | None = None,
) -> None:
    """Warn on known-unstable combinations before native crashes occur.

    The MPS segfault only affects MPS execution, not CPU. CPU inference is
    safe even on older macOS + torch stacks, so no error is raised for it.
    """
    if device != "mps":
        return

    if platform_system is None or torch_version is None or transformers_version is None:
        import torch
        import transformers

        platform_system = platform.system()
        torch_version = torch.__version__
        transformers_version = transformers.__version__

    if platform_system != "Darwin":
        return

    torch_mm = _parse_major_minor(torch_version)
    transformers_mm = _parse_major_minor(transformers_version)
    if torch_mm <= (2, 2) and transformers_mm <= (4, 39):
        raise ValueError(
            "Local MPS generation on macOS is blocked for the older runtime stack "
            f"(torch {torch_version}, transformers {transformers_version}) because it can segfault. "
            "Use `--device cpu` for the stable path, or upgrade the stack first."
        )


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
        self._session: Any = None

    def _get_session(self) -> Any:
        import aiohttp

        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    async def generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
    ) -> list[int]:
        import aiohttp

        prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt_text,
            "max_tokens": sampling_params.get("max_tokens", 512),
            "temperature": sampling_params.get("temperature", 0.7),
            "top_p": sampling_params.get("top_p", 1.0),
        }
        stop = sampling_params.get("stop")
        if stop is not None:
            payload["stop"] = stop
        try:
            session = self._get_session()
            async with session.post(
                f"{self.base_url}/v1/completions", json=payload
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError):
            raise RuntimeError(
                f"Cannot connect to vLLM server at {self.base_url}. "
                f"Start it first with: vllm serve {self.model} --port 8080"
            )

        completion_text = data["choices"][0]["text"]
        return list(self.tokenizer.encode(completion_text))


class LocalServerManager:
    """Runs generation in-process using a loaded HuggingFace model.

    Intended for offline testing.  Loads the model lazily on the first call.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        torch_dtype: str | None = None,
        allow_unsafe_mps: bool = False,
        local_files_only: bool = True,
        generation_timeout_seconds: float | None = 120.0,
        generation_heartbeat_seconds: float = 10.0,
    ) -> None:
        self.model_path = model_path
        self.device = _resolve_local_device(device)
        # dtype=None → auto-select based on device/platform at load time
        self.torch_dtype = torch_dtype
        self.allow_unsafe_mps = allow_unsafe_mps
        self.local_files_only = local_files_only
        self.generation_timeout_seconds = generation_timeout_seconds
        self.generation_heartbeat_seconds = generation_heartbeat_seconds
        self._model: Any = None
        self._tokenizer: Any = None

    def _build_model_load_kwargs(self, torch_dtype: Any) -> dict[str, Any]:
        """Assemble kwargs for AutoModelForCausalLM.from_pretrained()."""

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": self.local_files_only,
        }
        dtype_kwarg = "dtype" if _transformers_supports_dtype_kwarg() else "torch_dtype"
        model_kwargs[dtype_kwarg] = torch_dtype
        if _has_accelerate():
            model_kwargs["low_cpu_mem_usage"] = True
        else:
            print("Status  : accelerate not installed; using standard model loading")
        return model_kwargs

    def _build_generate_kwargs(
        self,
        *,
        inputs: Any,
        max_new: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> dict[str, Any]:
        """Build generation kwargs for one local inference call."""

        import copy

        attention_mask = inputs.new_ones(inputs.shape, dtype=inputs.dtype)
        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self._tokenizer.eos_token_id
        generation_config = copy.deepcopy(self._model.generation_config)
        generation_config.pad_token_id = pad_token_id
        generation_config.do_sample = do_sample

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new,
            "generation_config": generation_config,
            "attention_mask": attention_mask,
        }
        if self.generation_timeout_seconds is not None and self.generation_timeout_seconds > 0:
            # Use a StoppingCriteria instead of max_time so the deadline is
            # wall-clock based and fires on the first check AFTER the timeout,
            # including right after prefill. max_time only checks between tokens
            # and does not interrupt a slow first-token computation.
            try:
                from transformers import StoppingCriteria, StoppingCriteriaList

                deadline = time.perf_counter() + float(self.generation_timeout_seconds)

                class _WallClockStop(StoppingCriteria):
                    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                        return time.perf_counter() >= deadline

                generate_kwargs["stopping_criteria"] = StoppingCriteriaList([_WallClockStop()])
            except ImportError:
                generate_kwargs["max_time"] = float(self.generation_timeout_seconds)
        if do_sample:
            generation_config.temperature = temperature
            generation_config.top_p = top_p
        else:
            # Reset sampling-only knobs to neutral defaults so newer
            # Transformers versions do not warn about ignored flags.
            generation_config.temperature = 1.0
            generation_config.top_p = 1.0
            generation_config.top_k = 50
        return generate_kwargs

    def _run_generate_with_heartbeat(self, inputs: Any, generate_kwargs: dict[str, Any]) -> Any:
        """Run model.generate() with a periodic heartbeat for slow local inference."""

        import torch

        start = time.perf_counter()
        stop_event = threading.Event()

        def _heartbeat() -> None:
            while not stop_event.wait(self.generation_heartbeat_seconds):
                elapsed = time.perf_counter() - start
                print(f"Status  : still generating on {self.device} ({elapsed:.1f}s elapsed)")

        heartbeat_thread: threading.Thread | None = None
        if self.generation_heartbeat_seconds > 0:
            heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
            heartbeat_thread.start()
        try:
            with torch.no_grad():
                return self._model.generate(inputs, **generate_kwargs)
        finally:
            stop_event.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=0.1)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        import torch

        _validate_local_runtime_device(self.device, allow_unsafe_mps=self.allow_unsafe_mps)
        _validate_local_runtime_stack(self.device)
        print(f"Status  : loading local model on {self.device}")
        logger.info("Loading model %s onto %s …", self.model_path, self.device)
        config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        _validate_local_generation_config(self.model_path, config)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        dtype_name = self.torch_dtype or _auto_select_dtype(self.device)
        torch_dtype = getattr(torch, dtype_name)
        print(f"Status  : using dtype {dtype_name}")
        model_kwargs = self._build_model_load_kwargs(torch_dtype)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **model_kwargs,
        )
        self._model.eval()
        if self.device != "cpu":
            self._model.to(self.device)
        print("Status  : local model ready")

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
        top_p = float(sampling_params.get("top_p", 1.0))
        do_sample = temp > 0
        print(f"Status  : generating up to {max_new} new tokens on {self.device}")
        inputs = torch.tensor([prompt_ids], dtype=torch.long).to(self.device)
        generate_kwargs = self._build_generate_kwargs(
            inputs=inputs,
            max_new=max_new,
            do_sample=do_sample,
            temperature=temp,
            top_p=top_p,
        )
        out = self._run_generate_with_heartbeat(inputs, generate_kwargs)
        print("Status  : generation complete")
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
    parser.add_argument("--device", type=str, default="auto", help="Device for local model: auto, cpu, cuda, or mps")
    parser.add_argument("--allow_unsafe_mps", action="store_true", help="Allow local MPS generation on macOS even though it may segfault")
    parser.add_argument("--allow_remote_model_downloads", action="store_true", help="Allow `--local` model loading to query/download from Hugging Face instead of cache-only loading")
    parser.add_argument("--generation_timeout_seconds", type=float, default=120.0, help="Best-effort local generation timeout in seconds")
    parser.add_argument("--generation_heartbeat_seconds", type=float, default=10.0, help="How often local generation prints a still-running heartbeat")
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Model dtype for local inference: float32, bfloat16, or float16. Default: auto (bfloat16 on Apple Silicon CPU, float16 on CUDA/MPS, float32 elsewhere)",
    )

    # Search
    parser.add_argument("--search_url", type=str, default="http://localhost:8000/retrieve")
    parser.add_argument("--topk", type=int, default=5)

    # Loop tuning
    parser.add_argument("--max_turns", type=int, default=8)
    parser.add_argument("--max_search_limit", type=int, default=0, help="0 = same as max_turns")
    parser.add_argument("--max_answer_rejections", type=int, default=3)
    parser.add_argument("--no_evidence_gate", action="store_true", help="Allow answer without sufficient evidence")
    parser.add_argument("--require_search", action="store_true", help="Disable internal-knowledge direct answers")
    parser.add_argument(
        "--intent_model",
        type=str,
        default=None,
        help="Path to a pre-trained intent classifier (.pt file from train_intent_classifier.py). "
             "Preferred over --intent_examples — loads instantly with no retraining.",
    )
    parser.add_argument(
        "--intent_examples",
        type=str,
        default=None,
        help="JSON file of intent-labeled examples for on-the-fly training. "
             "Use --intent_model instead when the model has been pre-trained.",
    )
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

    print(f"\nMode    : {args.mode}")
    print(f"Model   : {args.model}")
    print(f"Question: {args.question}\n")

    # Load tokenizer
    logger.info("Loading tokenizer %s …", args.model)
    try:
        tokenizer = _load_tokenizer_for_cli(
            args.model,
            local=args.local,
            allow_remote_model_downloads=args.allow_remote_model_downloads,
        )
    except RuntimeError as exc:
        print(f"Error   : {exc}")
        return

    server_manager = _build_server_manager(args, tokenizer)

    sampling_params = _build_sampling_params(args)

    intent_pipeline = None
    if args.intent_model:
        # Fast path: load a pre-trained model saved by train_intent_classifier.py
        from src.agent_loop.intent_classifier import IntentPipeline
        print(f"Status  : loading intent model from {args.intent_model}")
        intent_pipeline = IntentPipeline.load(args.intent_model)
        print(f"Status  : intent model ready (vocab size {len(intent_pipeline._vocab.token2idx)})")
    elif args.intent_examples:
        # Slow path: train from scratch on the fly (use --intent_model for production)
        from src.agent_loop.intent_classifier import IntentPipeline, load_training_data
        print(f"Status  : training intent classifier from {args.intent_examples}")
        training_data = load_training_data(args.intent_examples)
        if training_data:
            intent_pipeline = IntentPipeline()
            intent_pipeline.train(training_data, epochs=10)
            print("Status  : intent classifier ready")

    try:
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
    except ValueError as exc:
        if args.local and _handle_local_cli_value_error(exc):
            return
        raise
    finally:
        if hasattr(server_manager, "aclose"):
            await server_manager.aclose()


if __name__ == "__main__":
    asyncio.run(main())
