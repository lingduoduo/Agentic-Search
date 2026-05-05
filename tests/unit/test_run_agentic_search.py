from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agent_loop.agent_loop import AgentLoopBase
from src.run_agentic_search import (
    _build_sampling_params,
    _has_accelerate,
    _friendly_model_load_error,
    _handle_local_cli_value_error,
    _parse_major_minor,
    _resolve_local_device,
    _validate_local_runtime_device,
    _validate_local_runtime_stack,
    _validate_local_generation_config,
)


class _DummyLoop(AgentLoopBase):
    async def run(self, messages, sampling_params):
        raise NotImplementedError


class _TokenizerWithoutChatTemplate:
    chat_template = None

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        raise AssertionError(
            "apply_chat_template should not be used when chat_template is missing"
        )

    def encode(self, text: str) -> list[int]:
        return [len(text)]


class _TokenizerWithIds:
    pad_token_id = None
    eos_token_id = 99


def test_build_prompt_ids_falls_back_to_encode_without_chat_template():
    loop = _DummyLoop(
        tokenizer=_TokenizerWithoutChatTemplate(),
        server_manager=object(),
    )
    prompt_ids = loop._build_prompt_ids_sync(
        [{"role": "user", "content": "What is FAISS?"}]
    )
    assert prompt_ids == [14]


def test_validate_local_generation_config_rejects_encoder_only_model():
    config = SimpleNamespace(
        model_type="bert",
        is_encoder_decoder=False,
        architectures=["BertModel"],
    )

    with pytest.raises(
        ValueError, match="Local generation mode requires a generative language model"
    ):
        _validate_local_generation_config("BAAI/bge-base-en-v1.5", config)


def test_validate_local_generation_config_allows_causal_lm():
    config = SimpleNamespace(
        model_type="llama",
        is_encoder_decoder=False,
        architectures=["LlamaForCausalLM"],
    )

    _validate_local_generation_config("meta-llama/Llama-3.1-8B-Instruct", config)


def test_friendly_model_load_error_formats_gated_repo_failure():
    exc = OSError(
        "You are trying to access a gated repo. 401 Client Error. Cannot access gated repo"
    )
    message = _friendly_model_load_error("meta-llama/Llama-3.1-8B-Instruct", exc)
    assert message is not None
    assert "gated on Hugging Face" in message
    assert "huggingface-cli login" in message


def test_friendly_model_load_error_formats_missing_repo_failure():
    exc = OSError("404 Client Error. Repository Not Found for url")
    message = _friendly_model_load_error("missing/model", exc)
    assert message is not None
    assert "could not find that repo" in message


def test_friendly_model_load_error_formats_cache_only_miss():
    exc = OSError(
        "Couldn't connect to the Hub and cannot find the requested files in the disk cache"
    )
    message = _friendly_model_load_error("Qwen/Qwen2.5-1.5B-Instruct", exc)
    assert message is not None
    assert "local Hugging Face cache" in message
    assert "--allow_remote_model_downloads" in message


def test_build_sampling_params_uses_cli_namespace():
    args = SimpleNamespace(temperature=0.2, max_tokens=64, top_p=0.9)
    assert _build_sampling_params(args) == {
        "temperature": 0.2,
        "max_tokens": 64,
        "top_p": 0.9,
    }


def test_resolve_local_device_returns_explicit_choice():
    assert _resolve_local_device("cpu") == "cpu"


def test_has_accelerate_returns_boolean():
    assert isinstance(_has_accelerate(), bool)


def test_parse_major_minor_handles_build_suffix():
    assert _parse_major_minor("2.2.2+cpu") == (2, 2)


def test_validate_local_runtime_device_rejects_mps_without_override():
    with pytest.raises(ValueError, match="Apple MPS is disabled by default"):
        _validate_local_runtime_device("mps", allow_unsafe_mps=False)


def test_validate_local_runtime_device_allows_mps_with_override():
    _validate_local_runtime_device("mps", allow_unsafe_mps=True)


def test_validate_local_runtime_stack_rejects_old_macos_mps_stack():
    with pytest.raises(ValueError, match="older runtime stack"):
        _validate_local_runtime_stack(
            "mps",
            platform_system="Darwin",
            torch_version="2.2.2",
            transformers_version="4.39.3",
        )


def test_validate_local_runtime_stack_allows_cpu_on_old_macos_stack():
    # CPU is safe even on old torch — only MPS can segfault.
    _validate_local_runtime_stack(
        "cpu",
        platform_system="Darwin",
        torch_version="2.2.2",
        transformers_version="4.39.3",
    )


def test_validate_local_runtime_stack_allows_newer_macos_mps_stack():
    _validate_local_runtime_stack(
        "mps",
        platform_system="Darwin",
        torch_version="2.5.1",
        transformers_version="4.46.0",
    )


def test_handle_local_cli_value_error_returns_true_for_known_error(capsys):
    handled = _handle_local_cli_value_error(
        ValueError(
            "Local generation on Apple MPS is disabled by default in this CLI because it can segfault"
        )
    )
    captured = capsys.readouterr()
    assert handled is True
    assert "Retry with `--device cpu`" in captured.out


def test_local_generate_sync_adds_attention_mask_for_greedy_decode():
    torch = pytest.importorskip("torch")
    from src.run_agentic_search import LocalServerManager

    captured: dict[str, object] = {}

    class _Model:
        generation_config = SimpleNamespace(
            pad_token_id=None,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=20,
        )

        def generate(self, inputs, **kwargs):
            captured["inputs"] = inputs
            captured["kwargs"] = kwargs
            return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    manager = LocalServerManager(
        model_path="dummy",
        device="cpu",
        generation_timeout_seconds=5.0,
        generation_heartbeat_seconds=999.0,
    )
    manager._tokenizer = _TokenizerWithIds()
    manager._model = _Model()

    result = manager._generate_sync([1, 2], {"max_tokens": 2, "temperature": 0})
    assert result == [3, 4]
    kwargs = captured["kwargs"]
    generation_config = kwargs["generation_config"]
    assert generation_config.do_sample is False
    assert generation_config.temperature == 1.0
    assert generation_config.top_p == 1.0
    assert generation_config.top_k == 50
    assert generation_config.pad_token_id == 99
    # max_time replaced by a wall-clock StoppingCriteria; verify it is present
    assert "stopping_criteria" in kwargs
    assert kwargs["attention_mask"].tolist() == [[1, 1]]
