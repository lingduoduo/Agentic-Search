from __future__ import annotations

from types import SimpleNamespace

from src.tools.semantic_router import (
    catalog_from_registry,
    default_tool_catalog,
)


def test_default_catalog_has_three_named_servers_with_expected_tools():
    catalog = default_tool_catalog()
    by_name = {s.name: s for s in catalog}
    assert set(by_name) == {"web_search", "knowledge_base", "answer"}
    assert {t.name for t in by_name["web_search"].tools} == {
        "search_web",
        "open_urls",
        "browser_search",
    }
    assert {t.name for t in by_name["knowledge_base"].tools} == {
        "search_indexed_documents",
        "retrieve_documents",
        "expand_query",
    }
    assert {t.name for t in by_name["answer"].tools} == {
        "ask_agentic_search",
        "rag_routing_tool",
    }
    # Every tool records its owning server.
    for server in catalog:
        for tool in server.tools:
            assert tool.server == server.name
            assert tool.description


def _entry(name, desc, source, provider_id=None):
    tool = SimpleNamespace(name=name, schema=SimpleNamespace(description=desc))
    return SimpleNamespace(tool=tool, source=source, provider_id=provider_id)


def _fake_registry(entries):
    return SimpleNamespace(list=lambda: entries)


def test_catalog_from_registry_groups_by_provider_and_source():
    reg = _fake_registry(
        [
            _entry("get_forecast", "weather forecast", "openapi", "weather"),
            _entry("get_alerts", "weather alerts", "openapi", "weather"),
            _entry("greet", "say hi", "function", None),
        ]
    )
    by_name = {s.name: s for s in catalog_from_registry(reg)}
    assert set(by_name) == {"weather", "local"}
    assert [t.name for t in by_name["weather"].tools] == ["get_forecast", "get_alerts"]
    assert [t.name for t in by_name["local"].tools] == ["greet"]
    assert by_name["weather"].tools[0].source == "openapi"


def test_catalog_from_registry_empty_is_empty():
    assert catalog_from_registry(_fake_registry([])) == []


def _documented_mcp_tools() -> set[str]:
    """Tool names from the 'Tools available to the LLM client' table in docs/mcp.md.

    Consolidation guard: the catalog's MCP-sourced tools must match the doc.
    """
    from pathlib import Path
    import re

    doc = (Path(__file__).resolve().parents[2] / "docs" / "mcp.md").read_text()
    start = doc.index("## Tools available to the LLM client")
    section = doc[start : doc.index("\n## ", start + 1)]
    names: set[str] = set()
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        first_col = line.split("|")[1].strip()
        m = re.fullmatch(r"`([a-z_]+)`", first_col)
        if m:
            names.add(m.group(1))
    return names


def test_catalog_mcp_tools_match_docs_mcp_table():
    catalog = default_tool_catalog()
    catalog_mcp = {
        t.name for server in catalog for t in server.tools if t.source == "mcp"
    }
    assert catalog_mcp == _documented_mcp_tools()
