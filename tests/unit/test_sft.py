"""Tests for supervised-training helpers built from search trajectories."""

from __future__ import annotations

import asyncio

from src.agent_loop import (
    SearchAgentLoop,
    SearchAgentLoopConfig,
    SearchEvaluationConfig,
    build_search_sft_example,
)
from src.agent_loop.context import SearchResult


class DummyTokenizer:
    chat_template = None

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(i) for i in ids if 0 <= i < 0x110000)


class DummyServerManager:
    def __init__(self, response_ids):
        self.response_ids = response_ids
        self.index = 0

    async def generate(self, request_id, prompt_ids, sampling_params):
        del request_id, prompt_ids, sampling_params
        response = self.response_ids[self.index]
        self.index += 1
        return list(response)


class FakeSearchClient:
    def __init__(self, responses):
        self.responses = responses

    async def retrieve(self, queries, topk=None):
        del topk
        return self.responses[tuple(queries)]

    async def fetch_urls(self, urls):
        del urls
        return []

    async def aclose(self):
        return None


def test_search_agent_loop_returns_full_action_trace_for_sft():
    tokenizer = DummyTokenizer()
    responses = [
        tokenizer.encode(
            "<plan>Find the date.</plan>"
            "<search_decision>search</search_decision>"
            "<subquestions>\nT1: identify the year\n</subquestions>"
            "<search>[T1] opening year</search>"
        ),
        tokenizer.encode("<answer>1931 [R1Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=4,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("opening year",): [
                [SearchResult(contents='"Bridge"\nOpened in 1931')],
            ],
        }
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "When did it open?"}], {})
    )

    assert output.action_trace is not None
    assert "<plan>Find the date.</plan>" in output.action_trace
    assert "<search>[T1] opening year</search>" in output.action_trace
    assert "<answer>1931 [R1Q1D1]</answer>" in output.action_trace
    assert any(msg["role"] == "assistant" for msg in output.trajectory_messages)
    assert any(msg["role"] == "user" for msg in output.trajectory_messages[1:])


def test_build_search_sft_example_uses_full_trace_not_just_final_answer():
    tokenizer = DummyTokenizer()
    responses = [
        tokenizer.encode("<plan>Plan.</plan><search>alpha</search>"),
        tokenizer.encode("<answer>Done [R1Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=3,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1, min_content_length=1
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("alpha",): [
                [SearchResult(contents='"A"\nbody')],
            ],
        }
    )
    input_messages = [{"role": "user", "content": "research this"}]
    output = asyncio.run(loop.run(input_messages, {}))

    example = build_search_sft_example(input_messages, output)

    assert "<plan>Plan.</plan>" in example.completion
    assert "<search>alpha</search>" in example.completion
    assert "<answer>Done [R1Q1D1]</answer>" in example.completion
    assert example.prompt_messages == input_messages
    assert example.trajectory_messages == [
        {"role": "assistant", "content": example.completion}
    ]


def test_build_search_sft_example_can_include_environment_messages():
    output_messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "<plan>Plan.</plan>"},
        {"role": "user", "content": "<information>Evidence</information>"},
        {"role": "assistant", "content": "<answer>Done</answer>"},
    ]
    from src.agent_loop import AgentLoopOutput

    output = AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=2,
        trajectory_messages=output_messages,
        action_trace="<plan>Plan.</plan>\n<answer>Done</answer>",
        final_answer="Done",
    )

    example = build_search_sft_example(
        [{"role": "user", "content": "question"}],
        output,
        include_environment_messages=True,
    )

    assert example.trajectory_messages == output_messages
