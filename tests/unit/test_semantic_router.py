from __future__ import annotations

from types import SimpleNamespace

from src.tools.semantic_router import (
    RoutingConfig,
    SemanticRouter,
    ServerDefinition,
    StructuredRequestParser,
    ToolDefinition,
    catalog_from_registry,
    default_tool_catalog,
    discover_tools,
    get_all_tools,
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


def test_get_all_tools_flattens_every_server():
    catalog = default_tool_catalog()
    flat = get_all_tools(catalog)
    # One flat entry per tool across all servers, order preserved.
    assert flat == [tool for server in catalog for tool in server.tools]
    assert len(flat) == sum(len(server.tools) for server in catalog)
    assert {t.name for t in flat} == {
        "search_web",
        "open_urls",
        "browser_search",
        "search_indexed_documents",
        "retrieve_documents",
        "expand_query",
        "ask_agentic_search",
        "rag_routing_tool",
    }


def test_get_all_tools_empty_catalog_is_empty():
    assert get_all_tools([]) == []


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


def test_router_ranks_web_tools_for_internet_request():
    router = SemanticRouter(default_tool_catalog())
    tools = router.route_request("search the public internet for recent news")
    assert tools, "expected at least one routed tool"
    assert tools[0].server == "web_search"


def test_router_ranks_knowledge_base_for_internal_docs_request():
    router = SemanticRouter(default_tool_catalog())
    tools = router.route_request("find internal indexed documents about FAISS")
    assert tools[0].server == "knowledge_base"


def _divergence_catalog():
    return [
        ServerDefinition(
            "weather",
            "weather forecast temperature climate",
            [
                ToolDefinition(
                    "lookup_weather", "get the current temperature", "", "weather"
                )
            ],
        ),
        ServerDefinition(
            "finance",
            "stock market finance trading",
            [
                ToolDefinition(
                    "lookup_stock", "get the current temperature", "", "finance"
                )
            ],
        ),
    ]


def test_server_hint_changes_stage1_winner():
    router = SemanticRouter(_divergence_catalog())
    # Without a hint the request text drives stage 1: "temperature" matches weather.
    no_hint = router.route_request("get the current temperature")
    assert no_hint[0].name == "lookup_weather"
    # With a hint the server-stage text picks finance instead.
    hinted = router.route_request(
        "get the current temperature", server_hint="stock market finance"
    )
    assert hinted[0].name == "lookup_stock"


def test_empty_catalog_routes_to_nothing():
    assert SemanticRouter([]).route_request("anything") == []


def test_threshold_filters_zero_similarity_requests():
    router = SemanticRouter(
        default_tool_catalog(), RoutingConfig(similarity_threshold=0.5)
    )
    # No shared vocabulary with any server/tool description.
    assert router.route_request("qwerty zxcvbn asdfgh") == []


def test_top_k_larger_than_catalog_is_clamped_and_deduped():
    router = SemanticRouter(
        default_tool_catalog(), RoutingConfig(top_k_servers=10, top_k_tools=10)
    )
    tools = router.route_request("search retrieve documents and answer questions")
    names = [t.name for t in tools]
    assert len(names) == len(set(names))  # no duplicates
    assert len(names) <= 8  # total tools in the default catalog


def test_routing_details_shape():
    router = SemanticRouter(default_tool_catalog())
    details = router.get_routing_details("search the public internet for news")
    assert details["request"] == "search the public internet for news"
    assert isinstance(details["stage1_servers"], list)
    assert details["stage1_servers"][0]["name"] == "web_search"
    assert "web_search" in details["stage2_tools"]
    assert details["final_tools"][0]["server"] == "web_search"


def test_parser_round_trips():
    text = StructuredRequestParser.format_request("web platform", "fetch a page")
    parsed = StructuredRequestParser.parse_request(text)
    assert parsed == {"server": "web platform", "tool": "fetch a page"}


def test_parser_without_tags_returns_none():
    assert StructuredRequestParser.parse_request("just some text") is None


def test_parser_missing_tool_line_returns_none():
    text = "<tool_request>\nserver: web\n</tool_request>"
    assert StructuredRequestParser.parse_request(text) is None


def test_discover_tools_unstructured_web_request():
    tools = discover_tools("search the public internet for recent news")
    assert tools[0].server == "web_search"


def test_discover_tools_uses_structured_server_hint():
    request = StructuredRequestParser.format_request(
        "public web internet search", "fetch the full text of a web page url"
    )
    tools = discover_tools(request)
    assert tools[0].server == "web_search"


def test_discover_tools_empty_catalog():
    assert discover_tools("anything", catalog=[]) == []
