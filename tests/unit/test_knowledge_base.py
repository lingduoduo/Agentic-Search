from __future__ import annotations

from src.internal.tools.base import Tool
from src.internal.tools.knowledge_base import seed_tools, tool_knowledge_base
from src.internal.tools.registry import ToolRegistry
from src.internal.tools.semantic_router import catalog_from_registry

BUILTIN_NAMES = {"web_search", "search", "search_routing_tool"}


def test_knowledge_base_default_has_three_builtin_tools():
    tools = tool_knowledge_base()
    assert all(isinstance(t, Tool) for t in tools)
    assert {t.name for t in tools} == BUILTIN_NAMES


def test_knowledge_base_adds_rag_tool_when_llm_present():
    tools = tool_knowledge_base(llm=object())
    assert {t.name for t in tools} == BUILTIN_NAMES | {"rag_routing_tool"}


def test_seed_tools_registers_into_fresh_registry():
    reg = ToolRegistry()
    count = seed_tools(reg)
    assert count == 3
    assert reg.get("web_search") is not None
    # Built-ins register under source="function".
    assert all(e.source == "function" for e in reg.list())


def test_seeded_registry_surfaces_in_catalog_from_registry():
    reg = ToolRegistry()
    seed_tools(reg)
    catalog = catalog_from_registry(reg)
    # Function tools group into a single "local" server.
    by_name = {s.name: s for s in catalog}
    assert set(by_name) == {"local"}
    assert {t.name for t in by_name["local"].tools} == BUILTIN_NAMES


def test_seed_tools_accepts_explicit_tools():
    reg = ToolRegistry()
    tools = tool_knowledge_base(llm=object())
    assert seed_tools(reg, tools=tools) == 4
    assert reg.get("rag_routing_tool") is not None
