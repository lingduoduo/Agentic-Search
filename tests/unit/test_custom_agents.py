"""Unit tests for custom agent configuration."""

from __future__ import annotations

import asyncio

import pytest

from src import (
    AgentActionConfig,
    AgentKnowledgeSource,
    CustomAgentConfig,
    FunctionTool,
    MCPIntegrationConfig,
    ToolAgentLoop,
    ToolAgentLoopConfig,
    WorkflowStepConfig,
)


class _CapturingTokenizer:
    chat_template = ""

    def __init__(self) -> None:
        self.tool_batches: list[list[dict] | None] = []
        self.message_batches: list[list[dict]] = []

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
        del add_generation_prompt
        self.tool_batches.append(tools)
        self.message_batches.append(list(messages))
        text = "\n".join(message.get("content", "") for message in messages)
        return self.encode(text) if tokenize else text


class _SequencedServer:
    def __init__(self, tokenizer: _CapturingTokenizer, responses: list[str]) -> None:
        self.tokenizer = tokenizer
        self.responses = responses
        self.index = 0

    async def generate(self, request_id, prompt_ids, sampling_params):
        del request_id, prompt_ids, sampling_params
        response = self.responses[self.index]
        self.index += 1
        return self.tokenizer.encode(response)


def test_custom_agent_config_builds_prompt_from_agent_capabilities():
    config = CustomAgentConfig(
        name="Research Agent",
        instructions="Answer with cited enterprise knowledge.",
        knowledge=(
            AgentKnowledgeSource(
                source_id="policies",
                title="Policy docs",
                description="Authoritative HR and legal material",
                tags=("internal", "approved"),
            ),
        ),
        actions=(
            AgentActionConfig(
                name="lookup_policy",
                description="Retrieve approved policies",
                tool_name="search",
            ),
        ),
        mcp_integrations=(
            MCPIntegrationConfig(
                name="linear",
                endpoint="https://mcp.example.test",
                tool_names=("create_ticket",),
            ),
        ),
        workflow=(
            WorkflowStepConfig(
                step_id="search",
                instruction="Retrieve relevant policy snippets",
                action_name="lookup_policy",
            ),
        ),
    )

    prompt = config.system_prompt()

    assert "Research Agent" in prompt
    assert "Policy docs" in prompt
    assert "lookup_policy (search)" in prompt
    assert "linear" in prompt
    assert "Retrieve relevant policy snippets" in prompt
    assert config.enabled_tool_names() == {"search", "create_ticket"}


def test_custom_agent_config_rejects_unknown_workflow_action():
    config = CustomAgentConfig(
        name="Broken Agent",
        workflow=(
            WorkflowStepConfig(
                step_id="missing",
                instruction="Call an action",
                action_name="missing_action",
            ),
        ),
    )

    with pytest.raises(ValueError, match="unknown action"):
        config.system_prompt()


def test_tool_agent_loop_uses_custom_agent_prompt_and_filters_actions():
    tokenizer = _CapturingTokenizer()

    @FunctionTool.from_fn(name="search")
    def search(query: str) -> str:
        return f"found {query}"

    @FunctionTool.from_fn(name="delete_record")
    def delete_record(record_id: str) -> str:
        return f"deleted {record_id}"

    agent_config = CustomAgentConfig(
        name="Safe Research Agent",
        instructions="Only use approved read actions.",
        actions=(AgentActionConfig(name="search"),),
        workflow=(
            WorkflowStepConfig(
                step_id="retrieve",
                instruction="Search before answering when context is missing",
                action_name="search",
            ),
        ),
    )
    server = _SequencedServer(
        tokenizer,
        [
            '{"name":"search","arguments":{"query":"policy"}}',
            "done",
        ],
    )
    loop = ToolAgentLoop(
        tokenizer=tokenizer,
        server_manager=server,
        tools=[search, delete_record],
        config=ToolAgentLoopConfig(response_length=256),
        agent_config=agent_config,
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "check policy"}], {"temperature": 0.0})
    )

    exposed_tool_names = {
        tool["function"]["name"] for tool in tokenizer.tool_batches[1] or []
    }
    first_message = tokenizer.message_batches[1][0]
    assert exposed_tool_names == {"search"}
    assert first_message["role"] == "system"
    assert "Safe Research Agent" in first_message["content"]
    assert "Search before answering" in first_message["content"]
    assert output.final_answer == "done"
