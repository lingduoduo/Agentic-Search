from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agent_loop.agent_loop import AgentLoopBase
from src.run_agentic_search import _validate_local_generation_config


class _DummyLoop(AgentLoopBase):
    async def run(self, messages, sampling_params):
        raise NotImplementedError


class _TokenizerWithoutChatTemplate:
    chat_template = None

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        raise AssertionError("apply_chat_template should not be used when chat_template is missing")

    def encode(self, text: str) -> list[int]:
        return [len(text)]


def test_build_prompt_ids_falls_back_to_encode_without_chat_template():
    loop = _DummyLoop(
        tokenizer=_TokenizerWithoutChatTemplate(),
        server_manager=object(),
    )
    prompt_ids = loop._build_prompt_ids_sync([{"role": "user", "content": "What is FAISS?"}])
    assert prompt_ids == [14]


def test_validate_local_generation_config_rejects_encoder_only_model():
    config = SimpleNamespace(
        model_type="bert",
        is_encoder_decoder=False,
        architectures=["BertModel"],
    )

    with pytest.raises(ValueError, match="Local generation mode requires a generative language model"):
        _validate_local_generation_config("BAAI/bge-base-en-v1.5", config)


def test_validate_local_generation_config_allows_causal_lm():
    config = SimpleNamespace(
        model_type="llama",
        is_encoder_decoder=False,
        architectures=["LlamaForCausalLM"],
    )

    _validate_local_generation_config("meta-llama/Llama-3.1-8B-Instruct", config)
