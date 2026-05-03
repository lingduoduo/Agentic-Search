"""Unit tests for src.agent_loop."""

import asyncio

import pytest

from src.agent_loop import (
    AgentLoopBase,
    AgentLoopConfig,
    SingleTurnAgentLoop,
    get_registered_agent_loop,
    list_registered_agent_loops,
    register,
)


class DummyTokenizerWithTemplate:
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        assert add_generation_prompt is True
        assert tokenize is True
        return [11, 12, 13, 14, 15]


class DummyTokenizerWithEncode:
    def encode(self, text):
        return [ord(char) for char in text]


class DummyServerManager:
    def __init__(self, response_ids):
        self.response_ids = response_ids
        self.calls = []

    async def generate(self, request_id, prompt_ids, sampling_params):
        self.calls.append(
            {
                "request_id": request_id,
                "prompt_ids": prompt_ids,
                "sampling_params": sampling_params,
            }
        )
        return list(self.response_ids)


class DummySyncServerManager:
    def __init__(self, response_ids):
        self.response_ids = response_ids

    def generate(self, request_id, prompt_ids, sampling_params):
        del request_id, prompt_ids, sampling_params
        return list(self.response_ids)


class ConcreteAgentLoop(AgentLoopBase):
    async def run(self, messages, sampling_params):
        del messages, sampling_params
        raise NotImplementedError


def test_build_prompt_ids_uses_chat_template_when_available():
    loop = ConcreteAgentLoop(
        tokenizer=DummyTokenizerWithTemplate(),
        server_manager=DummyServerManager([]),
        config=AgentLoopConfig(prompt_length=3, response_length=5),
    )
    prompt_ids = asyncio.run(loop.build_prompt_ids([{"role": "user", "content": "hello"}]))
    assert prompt_ids == [13, 14, 15]


def test_build_prompt_ids_falls_back_to_encode():
    loop = ConcreteAgentLoop(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=DummyServerManager([]),
        config=AgentLoopConfig(prompt_length=4, response_length=5),
    )
    prompt_ids = asyncio.run(
        loop.build_prompt_ids([{"role": "user", "content": "abc"}, {"role": "assistant", "content": "de"}])
    )
    assert len(prompt_ids) == 4


def test_generate_response_ids_truncates_to_response_length():
    server_manager = DummyServerManager([1, 2, 3, 4, 5])
    loop = ConcreteAgentLoop(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=server_manager,
        config=AgentLoopConfig(prompt_length=10, response_length=3),
    )
    response_ids = asyncio.run(loop.generate_response_ids([9, 9], {"temperature": 0.1}, request_id="req-1"))
    assert response_ids == [1, 2, 3]
    assert server_manager.calls[0]["request_id"] == "req-1"


def test_single_turn_agent_loop_returns_expected_output():
    server_manager = DummyServerManager([21, 22, 23, 24])
    loop = SingleTurnAgentLoop(
        tokenizer=DummyTokenizerWithTemplate(),
        server_manager=server_manager,
        config=AgentLoopConfig(prompt_length=4, response_length=2),
    )
    output = asyncio.run(loop.run([{"role": "user", "content": "hello"}], {"temperature": 0.7}))
    assert output.prompt_ids == [12, 13, 14, 15]
    assert output.response_ids == [21, 22]
    assert output.response_mask == [1, 1]
    assert output.num_turns == 1
    assert "generate_sequences" in output.metrics
    assert output.request_id is not None


def test_register_stores_class_by_name():
    @register("test_agent_loop")
    class RegisteredLoop(ConcreteAgentLoop):
        pass

    assert RegisteredLoop.__name__ == "RegisteredLoop"


def test_get_registered_agent_loop_returns_single_turn_loop():
    registered = get_registered_agent_loop("single_turn_agent")
    assert registered is SingleTurnAgentLoop


def test_list_registered_agent_loops_includes_single_turn():
    assert "single_turn_agent" in list_registered_agent_loops()


def test_get_registered_agent_loop_raises_for_unknown_name():
    with pytest.raises(KeyError, match="Unknown agent loop"):
        get_registered_agent_loop("missing_loop")


def test_generate_response_ids_supports_sync_server_manager():
    loop = ConcreteAgentLoop(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=DummySyncServerManager([7, 8, 9, 10]),
        config=AgentLoopConfig(prompt_length=10, response_length=2),
    )
    response_ids = asyncio.run(loop.generate_response_ids([1, 2], {"temperature": 0.3}, request_id="req-sync"))
    assert response_ids == [7, 8]
