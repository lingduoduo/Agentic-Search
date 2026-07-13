# MCP server

[← Back to README](../README.md)

This guide explains how to install, run, configure, and connect to the Agentic Search MCP server.

## Overview

The MCP server exposes Agentic Search capabilities as [Model Context Protocol](https://modelcontextprotocol.io/) tools, letting any MCP-compatible client (Claude Desktop, Cursor, etc.) query your knowledge base directly. It delegates data access to the main API server, which manages access controls.

## Install and run

Install the `mcp` extra and launch the HTTP server on port 8090:

```bash
pip install -e ".[mcp]"
uvicorn src.internal.mcp_server.api:mcp_app --port 8090
```

The server also launches with the `Run All Services` task from the default `launch.json`, and can be launched independently through the VS Code debugger.

## Authentication and transport

Provide a Personal Access Token or API key in the `Authorization` header as a Bearer token. The MCP server quickly validates and passes through the token on every request.

- **Transport:** HTTP POST (MCP over HTTP)
- **Port:** 8090 (shares the API server's domain)
- **Framework:** FastMCP with a FastAPI wrapper
- **Database:** none; all work delegates to the API server

OAuth and stdio transport may be supported in the future.

## Configure MCP clients

### Claude Desktop

Add this to `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS:

```json
{
  "mcpServers": {
    "agentic-search": {
      "type": "http",
      "url": "http://localhost:8090/",
      "transport": "http",
      "headers": { "Authorization": "Bearer YOUR_TOKEN_HERE" }
    }
  }
}
```

For a remote deployment, replace the URL with `https://[YOUR_DOMAIN]:8090/`. Other MCP clients generally support HTTP transport with custom headers; refer to the client's configuration documentation.

## Tools available to the LLM client

| Tool | What it does |
|------|-------------|
| `search_indexed_documents` | Search the private knowledge base with optional source filter |
| `search_web` | Web search via Google Custom Search or SerpAPI |
| `open_urls` | Fetch full page text from a list of URLs |
| `ask_agentic_search` | Full `SearchAgentLoop` answer with citations |
| `retrieve_documents` | Raw retrieval — returns full document content and relevance scores |
| `expand_query` | Query decomposition and HyDE expansion |

MCP tool selection is independent of the web UI's `/api/agent` auto-router. An MCP client explicitly invokes an exposed tool according to its own model and client policy; it does not pass through the web backend's internal → SerpAPI → browser fallback sequence. For the web API contract, see [API request routing](request-routing.md).

Dynamic tools registered via `FunctionTool` / `ApiToolRegistry` can be mirrored to MCP by calling `sync_tool_to_mcp(name)` after registration (`src/internal/mcp_server/tools/dynamic.py`).

## Resources

| Resource | What it exposes |
|----------|----------------|
| `indexed_sources` | Available retrieval source types based on configured API keys |
| `document_sets` | Document sets scoped for search |

## Debug with MCP Inspector

```bash
npx @modelcontextprotocol/inspector http://localhost:8090/
```

In Inspector, ignore the OAuth menus, open **Authentication**, select **Bearer Token**, paste the token, and connect. You can then browse tools, test calls, inspect payloads, and debug authentication.

Check server health with:

```bash
curl http://localhost:8090/health
```

Expected response:

```json
{
  "status": "healthy",
  "service": "mcp_server"
}
```

## Environment variables

| Var | Default | Description |
|-----|---------|-------------|
| `MCP_SERVER_ENABLED` | `false` | Enable the MCP server |
| `MCP_SERVER_PORT` | `8090` | MCP server port |
| `MCP_SERVER_CORS_ORIGINS` | — | Comma-separated allowed origins for CORS |
| `API_SERVER_HOST` | `127.0.0.1` | Host of the web backend |
| `API_SERVER_PROTOCOL` | `http` | Protocol for the web backend URL |
| `API_SERVER_URL_OVERRIDE_FOR_HTTP_REQUESTS` | — | Override the full web backend URL; takes precedence over protocol and host |
