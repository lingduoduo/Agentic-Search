# Agentic Search MCP Server

## Overview

The Agentic Search MCP server allows LLMs to connect to your instance and access its knowledge base and search capabilities through the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/).

With the Agentic Search MCP Server, you can search your knowledgebase,
give your LLMs web search, and upload and manage documents.

All access controls are managed within the main application.

### Authentication

Provide a Personal Access Token or API Key in the `Authorization` header as a Bearer token.
FastMCP verifies the credential. Indexed-document tools then forward the request token only to the authenticated web search endpoint, where the backend derives user and group ACLs. Optional document-set filters can narrow those ACLs but cannot expand them.

```text
MCP bearer token → FastMCP verification → authenticated web search endpoint
→ server-derived user/group ACL + optional document-set narrowing
→ internal retrieval → MCP result or grounded synthesis
```

Authentication and backend failures fail closed: MCP tools return an error or empty result and never fall back to unfiltered raw retrieval.

Depending on usage, the MCP Server may support OAuth and stdio in the future.

### Default Configuration
- **Transport**: HTTP POST (MCP over HTTP)
- **Port**: 8090 (shares domain with API server)
- **Framework**: FastMCP with FastAPI wrapper
- **Database**: None (all work delegates to the API server)

### Architecture

The MCP server is built on [FastMCP](https://github.com/jlowin/fastmcp) and runs alongside the main API server:

```
┌─────────────────┐
│  LLM Client     │
│  (Claude, etc)  │
└────────┬────────┘
         │ MCP over HTTP
         │ (POST with bearer)
         ▼
┌─────────────────┐
│  MCP Server     │
│  Port 8090      │
│  ├─ Auth        │
│  ├─ Tools       │
│  └─ Resources   │
└────────┬────────┘
         │ Authenticated web search request
         │ (caller's bearer token)
         ▼
┌─────────────────┐
│  API Server     │
│  Port 8080      │
│  ├─ Auth checks │
│  ├─ Derive ACLs │
│  └─ Retrieval   │
└─────────────────┘
```

## Configuring MCP Clients

### Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "agentic-search": {
      "url": "https://[YOUR_DOMAIN]:8090/",
      "transport": "http",
      "headers": {
        "Authorization": "Bearer YOUR_TOKEN_HERE"
      }
    }
  }
}
```

### Other MCP Clients

Most MCP clients support HTTP transport with custom headers. Refer to your client's documentation for configuration details.

## Capabilities

### Tools

The server provides tools for searching, retrieving, and synthesizing information. Indexed-document tools all use the authenticated web search boundary described above.

1. `search_indexed_documents`
Search the user's private knowledge base. Returns ranked documents with content snippets, scores, and metadata.

2. `search_web`
Search the public internet for current events and general knowledge. Returns web search results with titles, URLs, and snippets.

3. `open_urls`
Retrieve the complete text content from specific web URLs. Useful for fetching full page content after finding relevant URLs via `search_web`.

4. `retrieve_documents`
Return authenticated document content and relevance scores without answer synthesis.

5. `ask_agentic_search`
Synthesize an answer only from authenticated, non-blank retrieved evidence. Grounding verification validates citation labels and answer/evidence overlap; it reduces unsupported output but is not a hard no-hallucination guarantee.

6. `expand_query`
Generate keyword variants for improved search recall when an LLM is configured.

7. `extract_document`
Extract bounded content from a document supplied directly in the tool request. Supported extensions are `.pdf`, `.docx`, `.pptx`, `.csv`, and `.txt`. Install the optional parsers with:

```bash
pip install "agentic-search[mcp-documents]"
```

The tool accepts only a simple file name and base64-encoded document bytes; paths and URLs are intentionally unsupported. Its request schema is:

```json
{
  "file_name": "report.pdf",
  "content_base64": "<base64 document bytes>",
  "page_range": "1-3"
}
```

`page_range` is optional and applies only to PDF files. CSV requests may set `max_rows` (default 1,000; maximum 10,000 rows). Decoded input is limited to 20 MiB and returned text or structured document content is limited to 50,000 characters; responses report when output is truncated.

### Resources

1. `indexed_sources`
Lists all document sources currently indexed in the tenant (e.g., `"confluence"`, `"github"`). Use these values to filter results when calling `search_indexed_documents`.

## Local Development

### Running the MCP Server

The MCP Server automatically launches with the `Run All Services` task from the default launch.json.

You can also independently launch the Server via the vscode debugger.

### Testing with MCP Inspector

The [MCP Inspector](https://github.com/modelcontextprotocol/inspector) is a debugging tool for MCP servers:

```bash
npx @modelcontextprotocol/inspector http://localhost:8090/
```

**Setup in Inspector:**

1. Ignore the OAuth configuration menus
2. Open the **Authentication** tab
3. Select **Bearer Token** authentication
4. Paste your bearer token
5. Click **Connect**

Once connected, you can:
- Browse available tools
- Test tool calls with different parameters
- View request/response payloads
- Debug authentication issues

### Health Check

Verify the server is running:

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

### Environment Variables

**MCP Server Configuration:**
- `MCP_SERVER_ENABLED`: Enable MCP server (set to "true" to enable, default: disabled)
- `MCP_SERVER_PORT`: Port for MCP server (default: 8090)
- `MCP_SERVER_CORS_ORIGINS`: Comma-separated CORS origins (optional)

**API Server Connection:**
- `API_SERVER_PROTOCOL`: Protocol for API server connection (default: "http")
- `API_SERVER_HOST`: Hostname for API server connection (default: "127.0.0.1")
- `API_SERVER_URL_OVERRIDE_FOR_HTTP_REQUESTS`: Optional override URL. If set, takes precedence over the protocol/host variables.
