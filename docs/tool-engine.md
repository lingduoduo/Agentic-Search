# Tool engine

[← Back to README](../README.md)

This guide covers the tool agent: multi-turn function calling with structured
tool dispatch. For the full routing contract, see
[API request routing](request-routing.md); for exposing tools to external MCP
clients, see [MCP server](mcp.md).

## Capabilities

- **Structured tool dispatch** — `ToolAgentLoop` (`src/agents/tool/tool_calling.py`)
  runs a generic multi-turn function-calling loop over registered tools, threading
  tool results back into the conversation.
- **Routing and memory** — tool requests are auto-routed and carry bounded session
  history, so follow-ups keep context across turns.
- **Registered tool catalog** — a `ToolRegistry` seeded at startup, extensible at
  runtime with OpenAPI-backed tools.

## Routing into the tool engine

With `mode` omitted, `/api/agent` classifies each request as `chat`, `search`, or
`tool`. Action commands such as “send”, “deploy”, or “create a ticket” route to
`tool`, which runs `ToolAgentLoop`; when no tool or policy model is available it
falls back to grounded chat. The explicit `tool_agent` mode selects the loop
directly. See [API request routing](request-routing.md) for the full decision
order and response metadata (tool calls are surfaced as `tool_calls`).

## Tool registry and discovery

The **`ToolRegistry` is the single source of truth** for the web/agent process's
runnable tools. `src/tools/knowledge_base.py` provides the built-in seed set, and
`seed_tools(tool_registry)` loads it at web startup; OpenAPI tools are added at
runtime via `register_from_openapi`.

- `discover_tools(request)` returns the tools most relevant to a natural-language
  request, using a two-stage TF-IDF match (rank servers, then tools).
- `default_tool_catalog()` reads the registry at call time, so it reflects
  whatever is registered. Built-in tools group into a `local` server; each OpenAPI
  provider gets its own server.

Built-in seed tools: `web_search`, `search`, `search_routing_tool` (and
`rag_routing_tool` when an LLM is configured). Discovery is a ranking aid, not a
dispatcher.

## Relationship to MCP

The [MCP server](mcp.md) exposes a set of tools to external MCP clients (Claude
Desktop, Cursor, etc.). That selection is client-driven and independent of this
auto-router: an MCP client invokes an exposed tool by its own policy and does not
pass through the web backend's routing. The `dynamic.py` bridge mirrors registry
tools **into** MCP via `sync_tool_to_mcp(name)`; the registry feeds MCP, not the
reverse.
