"""Unit tests for shared runtime agent state models."""

from __future__ import annotations

import asyncio

from src import (
    AgentState,
    FunctionTool,
    PerformanceMetrics,
    RouteDecision,
    TaskStatus,
    ToolAgentLoop,
    ToolAgentLoopConfig,
    ToolCall,
    ToolExecutionResult,
    ToolResult,
    UserRequest,
)
from src.tools import ToolEffect


class _CharTokenizer:
    chat_template = ""

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(
        self,
        messages,
        tools=None,
        add_generation_prompt=True,
        tokenize=True,
    ):
        del tools, add_generation_prompt
        text = "\n".join(message.get("content", "") for message in messages)
        return self.encode(text) if tokenize else text


class _SequencedServer:
    def __init__(self, responses: list[list[int]]) -> None:
        self.responses = responses
        self.index = 0

    async def generate(self, request_id, prompt_ids, sampling_params):
        del request_id, prompt_ids, sampling_params
        response = self.responses[self.index]
        self.index += 1
        return response


def test_agent_state_keeps_runtime_fields_structured_and_slotted():
    request = UserRequest(user_id="u1", channel="web", message="Find docs")
    state = AgentState(
        request_id="req-1",
        user_request=request,
        route=RouteDecision(intent="qa", confidence=0.91),
    )

    state.add_tool_result(ToolResult("search", True, ["doc-1"]))
    state.record_trace("route_selected", target_agent="qa")

    assert not hasattr(state, "__dict__")
    assert state.tool_results[0].success is True
    assert state.question == "Find docs"
    assert state.previous_queries == []
    assert state.retrieved_docs == []
    assert state.evidence_score == 0.0
    assert state.search_rounds == 0
    assert state.citations == []
    assert state.to_dict()["route"]["confidence"] == 0.91
    assert state.to_dict()["question"] == "Find docs"
    assert state.trace == [{"event": "route_selected", "target_agent": "qa"}]


def test_tool_execution_result_converts_to_simple_tool_result():
    result = ToolExecutionResult(
        tool_name="search",
        status=TaskStatus.COMPLETED,
        result={"hits": 2},
        performance=PerformanceMetrics(execution_time=0.12),
    )

    simple = result.to_tool_result()

    assert simple == ToolResult("search", True, {"hits": 2})
    assert result.to_dict()["status"] is TaskStatus.COMPLETED


def test_tool_agent_loop_records_structured_tool_trace():
    tokenizer = _CharTokenizer()

    @FunctionTool.from_fn(name="echo", effect=ToolEffect.READ_ONLY)
    def echo(text: str) -> str:
        return text.upper()

    server = _SequencedServer(
        [
            tokenizer.encode('{"name":"echo","arguments":{"text":"hello"}}'),
            tokenizer.encode("done"),
        ]
    )
    loop = ToolAgentLoop(
        tokenizer=tokenizer,
        server_manager=server,
        tools=[echo],
        config=ToolAgentLoopConfig(response_length=128),
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "say hello"}], {"temperature": 0.0})
    )

    assert output.final_answer == "done"
    assert output.trajectory_messages[-2] == {"role": "tool", "content": "HELLO"}
    assert output.action_trace is not None
    assert "TaskStatus.COMPLETED" in output.action_trace
    assert ToolCall("echo", {"text": "hello"}).to_dict() == {
        "tool_name": "echo",
        "arguments": {"text": "hello"},
    }
