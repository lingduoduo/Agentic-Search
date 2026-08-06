"""The nine public data tools are collected and seeded as agent-callable."""

from __future__ import annotations

from src.internal.tools.public_data import public_data_tools
from src.internal.tools.knowledge_base import (
    NOT_AGENT_CALLABLE,
    seed_tools,
    tool_knowledge_base,
)
from src.internal.tools.registry import ToolRegistry
from src.internal.tools.validation import validate_arguments

EXPECTED = {
    "search_wikipedia",
    "search_arxiv",
    "search_wayback",
    "get_weather",
    "get_stock_quote",
    "get_crypto_price",
    "convert_currency",
    "search_location",
    "search_nearby_places",
}


def test_public_data_tools_returns_the_nine():
    names = [t.name for t in public_data_tools()]
    assert len(names) == 9
    assert set(names) == EXPECTED
    assert len(set(names)) == len(names)


def test_every_tool_has_a_usable_schema():
    for tool in public_data_tools():
        params = tool.schema.parameters
        assert params["type"] == "object"
        assert params["properties"]
        assert tool.schema.description
        # Required names must exist in properties, or validation can never pass.
        for name in params.get("required", []):
            assert name in params["properties"]


def test_schema_required_fields_are_enforced():
    tool = next(t for t in public_data_tools() if t.name == "get_weather")
    assert validate_arguments(tool.schema.parameters, {}) != []
    assert validate_arguments(tool.schema.parameters, {"location": "Berlin"}) == []


def test_knowledge_base_includes_the_public_data_tools():
    names = {t.name for t in tool_knowledge_base()}
    assert EXPECTED <= names


def test_seeded_public_data_tools_are_agent_callable():
    registry = ToolRegistry()
    seed_tools(registry, tools=tool_knowledge_base())
    callable_names = {t.name for t in registry.agent_tools()}
    assert EXPECTED <= callable_names
    assert EXPECTED.isdisjoint(NOT_AGENT_CALLABLE)


def test_only_the_knowledge_tools_are_citeable():
    citeable = {t.name for t in public_data_tools() if t.citeable}
    assert citeable == {"search_wikipedia", "search_arxiv", "search_wayback"}
