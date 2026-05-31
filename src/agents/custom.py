"""Custom agent configuration helpers.

This module keeps persona instructions, knowledge bindings, actions, MCP
integrations, and workflows in one normalized shape that agent loops can
consume without bespoke prompt assembly at each call site.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AgentKnowledgeSource:
    source_id: str
    title: str | None = None
    description: str = ""
    tags: tuple[str, ...] = ()
    enabled: bool = True

    def prompt_line(self) -> str:
        label = self.title or self.source_id
        parts = [label]
        if self.description:
            parts.append(self.description)
        if self.tags:
            parts.append(f"tags: {', '.join(self.tags)}")
        return " - ".join(parts)


@dataclass(frozen=True, slots=True)
class AgentActionConfig:
    name: str
    description: str = ""
    tool_name: str | None = None
    mcp_integration: str | None = None
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_tool_name(self) -> str:
        return self.tool_name or self.name

    def prompt_line(self) -> str:
        label = self.name
        if self.tool_name and self.tool_name != self.name:
            label = f"{label} ({self.tool_name})"
        if self.mcp_integration:
            label = f"{label} via {self.mcp_integration}"
        return f"{label}: {self.description}" if self.description else label


@dataclass(frozen=True, slots=True)
class MCPIntegrationConfig:
    name: str
    endpoint: str
    tool_names: tuple[str, ...] = ()
    auth: str = "inherit"
    enabled: bool = True

    def prompt_line(self) -> str:
        tools = f" tools: {', '.join(self.tool_names)}" if self.tool_names else ""
        return f"{self.name} ({self.auth}) at {self.endpoint}{tools}"


@dataclass(frozen=True, slots=True)
class WorkflowStepConfig:
    step_id: str
    instruction: str
    action_name: str | None = None
    required: bool = True

    def prompt_line(self, index: int) -> str:
        action = f" Use action: {self.action_name}." if self.action_name else ""
        optional = "" if self.required else " Optional."
        return f"{index}. {self.instruction}{action}{optional}"


@dataclass(frozen=True, slots=True)
class CustomAgentConfig:
    name: str
    instructions: str = ""
    knowledge: tuple[AgentKnowledgeSource, ...] = ()
    actions: tuple[AgentActionConfig, ...] = ()
    mcp_integrations: tuple[MCPIntegrationConfig, ...] = ()
    workflow: tuple[WorkflowStepConfig, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def enabled_tool_names(self) -> set[str]:
        names = {action.resolved_tool_name for action in self.actions if action.enabled}
        for integration in self.mcp_integrations:
            if integration.enabled:
                names.update(integration.tool_names)
        return names

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("agent name is required.")
        action_names = {action.name for action in self.actions}
        tool_names = self.enabled_tool_names()
        for step in self.workflow:
            if step.action_name and step.action_name not in action_names | tool_names:
                raise ValueError(
                    f"workflow step {step.step_id!r} references unknown action "
                    f"{step.action_name!r}."
                )

    def system_prompt(self) -> str:
        self.validate()
        sections = [f"You are {self.name}."]
        if self.instructions.strip():
            sections.append(self.instructions.strip())

        knowledge = [
            source.prompt_line() for source in self.knowledge if source.enabled
        ]
        if knowledge:
            sections.append(
                "Knowledge sources:\n" + "\n".join(f"- {k}" for k in knowledge)
            )

        actions = [action.prompt_line() for action in self.actions if action.enabled]
        if actions:
            sections.append(
                "Available actions:\n" + "\n".join(f"- {a}" for a in actions)
            )

        integrations = [
            integration.prompt_line()
            for integration in self.mcp_integrations
            if integration.enabled
        ]
        if integrations:
            sections.append(
                "MCP integrations:\n" + "\n".join(f"- {mcp}" for mcp in integrations)
            )

        if self.workflow:
            steps = [
                step.prompt_line(index)
                for index, step in enumerate(self.workflow, start=1)
            ]
            sections.append("Workflow:\n" + "\n".join(steps))
        return "\n\n".join(sections)


def build_custom_agent_messages(
    messages: Sequence[dict[str, Any]],
    config: CustomAgentConfig | None,
) -> list[dict[str, Any]]:
    if config is None:
        return list(messages)

    system_prompt = config.system_prompt()
    if messages and messages[0].get("role") == "system":
        merged = dict(messages[0])
        existing = str(merged.get("content", "")).strip()
        merged["content"] = (
            f"{system_prompt}\n\n{existing}" if existing else system_prompt
        )
        return [merged, *messages[1:]]
    return [{"role": "system", "content": system_prompt}, *messages]


def filter_tools_for_agent(
    tools: Iterable[Any], config: CustomAgentConfig | None
) -> list[Any]:
    tool_list = list(tools)
    if config is None:
        return tool_list
    enabled = config.enabled_tool_names()
    if not enabled:
        return tool_list
    return [tool for tool in tool_list if getattr(tool, "name", None) in enabled]
