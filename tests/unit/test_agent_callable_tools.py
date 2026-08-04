"""Whether the agent may call a tool is a registration property, not a name match.

The tool agent used to filter a hardcoded set of tool names. That silently
no-ops the moment a tool is renamed — and one of those names arrives from a
remote MCP server, which can rename it without touching this repo.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from src.internal.tools import mcp_client
from src.internal.tools.base import FunctionTool
from src.internal.tools.mcp_client import McpServerSpec, register_mcp_tools
from src.internal.tools.registry import ToolRegistry


async def _noop() -> str:
    return ""


def _tool(name: str) -> FunctionTool:
    return FunctionTool(
        _noop, name=name, description=name, parameters={"type": "object"}
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_tools_are_agent_callable_by_default():
    registry = ToolRegistry()
    registry.register(_tool("weather"))
    assert registry.list()[0].agent_callable is True
    assert [t.name for t in registry.agent_tools()] == ["weather"]


def test_registering_a_tool_as_not_agent_callable_hides_it_from_the_agent():
    registry = ToolRegistry()
    registry.register(_tool("weather"))
    registry.register(_tool("runs_an_agent"), agent_callable=False)

    assert [t.name for t in registry.agent_tools()] == ["weather"]
    # Still registered and still invocable directly — only the agent is denied.
    assert registry.get("runs_an_agent") is not None
    assert {e.tool.name for e in registry.list()} == {"weather", "runs_an_agent"}


def test_summaries_expose_agent_callable():
    registry = ToolRegistry()
    registry.register(_tool("runs_an_agent"), agent_callable=False)
    assert registry.tool_summary("runs_an_agent")["agent_callable"] is False


# ---------------------------------------------------------------------------
# Seeded local tools
# ---------------------------------------------------------------------------


def test_rag_routing_tool_is_seeded_as_not_agent_callable():
    # It generates a whole answer instead of returning evidence, so offering it
    # to the agent lets the model delegate its own job.
    from src.internal.tools.knowledge_base import seed_tools, tool_knowledge_base
    from unittest.mock import MagicMock

    registry = ToolRegistry()
    seed_tools(registry, tools=tool_knowledge_base(llm=MagicMock()))

    assert registry.get("rag_routing_tool") is not None
    assert "rag_routing_tool" not in [t.name for t in registry.agent_tools()]
    assert "search" in [t.name for t in registry.agent_tools()]


# ---------------------------------------------------------------------------
# MCP-sourced tools
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_mcp(monkeypatch):
    session = SimpleNamespace(
        initialize=_ok,
        list_tools=_listing,
        call_tool=_ok,
    )

    @asynccontextmanager
    async def _connect(spec):
        yield session

    monkeypatch.setattr(mcp_client, "_connect", _connect)


async def _ok(*a, **k):
    return SimpleNamespace(isError=False, content=[])


async def _listing(*a, **k):
    return SimpleNamespace(
        tools=[
            SimpleNamespace(name="save_memory", description="", inputSchema={}),
            SimpleNamespace(name="ask_agentic_search", description="", inputSchema={}),
        ]
    )


@pytest.mark.asyncio
async def test_recursive_mcp_tool_is_registered_but_not_agent_callable(fake_mcp):
    registry = ToolRegistry()
    await register_mcp_tools(
        registry,
        [
            McpServerSpec(
                name="agentic",
                url="http://x/",
                agent_exclude=frozenset({"ask_agentic_search"}),
            )
        ],
    )

    names = [t.name for t in registry.agent_tools()]
    assert "save_memory" in names
    assert "ask_agentic_search" not in names
    assert registry.get("ask_agentic_search") is not None  # still usable directly


@pytest.mark.asyncio
async def test_exclusion_is_per_server_configuration(fake_mcp):
    # A server that exposes no agent-invoking tool excludes nothing.
    registry = ToolRegistry()
    await register_mcp_tools(registry, [McpServerSpec(name="agentic", url="http://x/")])
    assert "ask_agentic_search" in [t.name for t in registry.agent_tools()]


def test_parse_mcp_servers_applies_the_default_exclusion():
    from src.internal.tools.mcp_client import parse_mcp_servers

    spec = parse_mcp_servers("agentic=http://x/")[0]
    assert "ask_agentic_search" in spec.agent_exclude


def test_parse_mcp_servers_accepts_an_explicit_exclusion():
    from src.internal.tools.mcp_client import parse_mcp_servers

    spec = parse_mcp_servers("a=http://x/", agent_exclude="foo, bar")[0]
    assert spec.agent_exclude == frozenset({"foo", "bar"})


# ---------------------------------------------------------------------------
# User-scoped tools. Memory tools write per user; offering them with no user
# pools everyone's memories into one shared bucket.
# ---------------------------------------------------------------------------


def test_tools_are_not_user_scoped_by_default():
    registry = ToolRegistry()
    registry.register(_tool("weather"))
    assert registry.list()[0].user_scoped is False


def test_user_scoped_tools_are_withheld_when_there_is_no_user():
    registry = ToolRegistry()
    registry.register(_tool("weather"))
    registry.register(_tool("save_memory"), user_scoped=True)

    assert [t.name for t in registry.agent_tools(user_present=False)] == ["weather"]
    assert {t.name for t in registry.agent_tools(user_present=True)} == {
        "weather",
        "save_memory",
    }


def test_user_scoped_tools_stay_registered_and_invocable():
    registry = ToolRegistry()
    registry.register(_tool("save_memory"), user_scoped=True)
    # Withheld from the agent, still reachable through /admin/tools.
    assert registry.get("save_memory") is not None
    assert registry.tool_summary("save_memory")["user_scoped"] is True


def test_agent_callable_still_wins_over_user_presence():
    registry = ToolRegistry()
    registry.register(_tool("runs_an_agent"), agent_callable=False, user_scoped=True)
    assert registry.agent_tools(user_present=True) == []


@pytest.mark.asyncio
async def test_mcp_memory_tools_register_as_user_scoped(fake_mcp):
    from src.internal.tools.mcp_client import DEFAULT_USER_SCOPED

    registry = ToolRegistry()
    await register_mcp_tools(
        registry,
        [
            McpServerSpec(
                name="agentic", url="http://x/", user_scoped=DEFAULT_USER_SCOPED
            )
        ],
    )

    assert registry._entries["save_memory"].user_scoped is True
    assert [t.name for t in registry.agent_tools(user_present=False)] == [
        "ask_agentic_search"
    ]


def test_parse_mcp_servers_applies_the_default_user_scope():
    from src.internal.tools.mcp_client import parse_mcp_servers

    spec = parse_mcp_servers("agentic=http://x/")[0]
    assert "save_memory" in spec.user_scoped
    assert "search_memories" in spec.user_scoped


def test_parse_mcp_servers_accepts_an_explicit_user_scope():
    from src.internal.tools.mcp_client import parse_mcp_servers

    spec = parse_mcp_servers("a=http://x/", user_scoped="foo, bar")[0]
    assert spec.user_scoped == frozenset({"foo", "bar"})
