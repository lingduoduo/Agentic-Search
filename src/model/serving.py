"""Serving-side LLM backend: the ServerManager protocol, concrete managers, and
a factory selecting between them. The training-side LLMGenerationManager
(model/generation.py) is a separate concern and intentionally not here.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import threading
import time
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class ServerManager(Protocol):
    """The model boundary every agent loop calls: tokens in, tokens out."""

    async def generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
    ) -> list[int]: ...


# ---------------------------------------------------------------------------
# Helper functions (moved from examples/run_agentic_search.py)
# ---------------------------------------------------------------------------

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
    architecture_text = (
        ", ".join(architectures) if architectures else "unknown architecture"
    )
    raise ValueError(
        "Local generation mode requires a generative language model. "
        f"'{model_path}' looks like a non-generative encoder model "
        f"(model_type='{model_type}', architectures={architecture_text}). "
        "Use a chat or instruct causal LM such as Llama, Qwen, Mistral, or Gemma for "
        "`--mode single`, `--mode search`, or `--mode tool`."
    )


def _parse_major_minor(version_text: str) -> tuple[int, int]:
    core = version_text.split("+", 1)[0]
    parts = core.split(".")
    major = int(parts[0]) if parts and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return major, minor


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
    except (ImportError, ValueError, IndexError):
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


def _validate_local_runtime_device(
    device: str, *, allow_unsafe_mps: bool = False
) -> None:
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


class OpenAIServerManager:
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
                f"Cannot connect to inference server at {self.base_url}. "
                f"Start one first, e.g.: mlx_lm.server --model {self.model} --port 8080"
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
        if (
            self.generation_timeout_seconds is not None
            and self.generation_timeout_seconds > 0
        ):
            # Use a StoppingCriteria instead of max_time so the deadline is
            # wall-clock based and fires on the first check AFTER the timeout,
            # including right after prefill. max_time only checks between tokens
            # and does not interrupt a slow first-token computation.
            try:
                from transformers import StoppingCriteria, StoppingCriteriaList

                deadline = time.perf_counter() + float(self.generation_timeout_seconds)

                class _WallClockStop(StoppingCriteria):
                    def __call__(
                        self, input_ids: Any, scores: Any, **kwargs: Any
                    ) -> bool:
                        return time.perf_counter() >= deadline

                generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    [_WallClockStop()]
                )
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

    def _run_generate_with_heartbeat(
        self, inputs: Any, generate_kwargs: dict[str, Any]
    ) -> Any:
        """Run model.generate() with a periodic heartbeat for slow local inference."""

        import torch

        start = time.perf_counter()
        stop_event = threading.Event()

        def _heartbeat() -> None:
            while not stop_event.wait(self.generation_heartbeat_seconds):
                elapsed = time.perf_counter() - start
                print(
                    f"Status  : still generating on {self.device} ({elapsed:.1f}s elapsed)"
                )

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

        _validate_local_runtime_device(
            self.device, allow_unsafe_mps=self.allow_unsafe_mps
        )
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
        generation_start = time.perf_counter()
        out = self._run_generate_with_heartbeat(inputs, generate_kwargs)
        elapsed = time.perf_counter() - generation_start
        response_ids = out[0][len(prompt_ids) :].tolist()
        if (
            self.generation_timeout_seconds is not None
            and self.generation_timeout_seconds > 0
            and elapsed >= float(self.generation_timeout_seconds)
            and len(response_ids) < max_new
        ):
            print(
                "Warning : generation stopped by "
                f"--generation_timeout_seconds={self.generation_timeout_seconds} "
                f"after {len(response_ids)} token(s). Increase it or pass 0 to "
                "disable the timeout."
            )
        print("Status  : generation complete")
        return response_ids


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_server_manager(
    tokenizer: Any,
    *,
    server_url: str | None = None,
    model: str | None = None,
    device: str | None = None,
    **kwargs: Any,
) -> ServerManager:
    """Select the serving backend from resolved config.

    server_url set -> OpenAIServerManager (remote); else model set ->
    LocalServerManager (in-process); else ValueError.
    """
    if server_url:
        return OpenAIServerManager(
            tokenizer=tokenizer, base_url=server_url, model=model
        )
    if model:
        # Pass device through unchanged (do not coerce None -> "auto"): the prior
        # call sites passed device verbatim to LocalServerManager, and
        # _resolve_local_device treats None and "auto" differently. Keeping this a
        # transparent pass-through makes the factory strictly behavior-preserving.
        return LocalServerManager(model_path=model, device=device, **kwargs)
    raise ValueError("no model backend configured (set server_url or model)")
