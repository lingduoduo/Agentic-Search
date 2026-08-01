# MCP as an entrance

**Date:** 2026-08-01
**Status:** Approved

## Problem

Everything the tool agent can call is a retrieval wrapper. `tool_knowledge_base()`
seeds exactly four tools — `web_search`, `search`, `search_routing_tool`,
`rag_routing_tool` — and that is the entire tool inventory of the web process.
Narrowing that set (spec `2026-08-01-tool-agent-tool-selection-design.md`) made
the agent call tools reliably, but did not change what there is to call.

Meanwhile the MCP server exposes 13 genuinely different tools — memory,
documents, research — and **none of them are reachable from the web process.**
The bridge is one-directional: `mcp_server/tools/dynamic.py::_sync_all()` mirrors
`tool_registry` *outward* so external hosts (Claude Desktop, Cursor) can call our
tools. Nothing pulls the other way. There is no MCP client anywhere in `src/`;
every `ClientSession` in the tree is `aiohttp`, not MCP.

So MCP is an exit, not an entrance.

## Goals

- Tools from configured MCP servers become ordinary `tool_registry` tools:
  callable by the tool agent, listed by `/admin/tools`, visible in the Dev Console.
- Any MCP server works, not just ours — a third-party server's tools become agent
  tools by configuration alone.
- Off by default. Unset config changes nothing.
- An unreachable MCP server never stops the web process from starting.

## Non-goals

- No UI for managing MCP servers; configuration is environment only.
- No stdio transport. The server we ship is HTTP, so this is HTTP-only.
- No change to the outbound bridge's purpose.

## Design

### Configuration

`AGENTIC_SEARCH_MCP_SERVERS` holds `name=url` pairs, comma separated;
`AGENTIC_SEARCH_MCP_TOKEN` supplies the bearer token. Empty or unset means the
feature is off. A malformed entry is logged and skipped rather than raising —
one typo in an env var should not take the web process down.

### Registration

At startup each server is contacted, `list_tools()` is called, and every remote
tool is wrapped as a `FunctionTool` whose body opens a session and calls
`call_tool`. Entries register with `source="mcp"` and `provider_id=<server name>`.

`effect` is left `UNSPECIFIED` on purpose. Only `READ_ONLY` tools bypass the tool
agent's approval gate, so a remote tool that may write requires approval.

### Connection lifetime

Per call, not long-lived: discovery opens one session and each invocation opens
its own. That costs an HTTP round trip per call and buys a web process that
neither babysits a socket nor dies when an MCP server restarts.

### Three hazards, and what handles each

**Export loop.** `_sync_all()` mirrors the registry out to MCP. Re-exporting
tools we pulled *in* would offer a server its own tools back and duplicate the
catalog on every restart. `_exportable_entries()` now filters `source == "mcp"`.

**Recursion.** `ask_agentic_search` runs an agent. Handing it to the agent lets
the agent call itself, so it joins `_SHADOWED_TOOL_NAMES` in the tool-agent
runner. It stays fully usable through `/admin/tools`; only the agent is denied it.

**Startup deadlock.** Found while testing, not by inspection. Our MCP server
authenticates by calling back into this process's `/me`. A web process that
awaits discovery inside its own lifespan is not yet accepting connections, so
that callback is refused, the server returns 401, and discovery silently
registers nothing — failing against exactly the server we ship. Discovery is
therefore an `asyncio.create_task` scheduled during lifespan and running once
startup completes.

## Verification

Live MCP server on 8090, live web backend on 7860, feature configured:

```
registered: [web_search, search, search_routing_tool, rag_routing_tool,
             ask_agentic_search, extract_document, retrieve_documents,
             expand_query, search_indexed_documents, search_web, open_urls,
             save_memory, update_memory_from_conversation,
             generate_user_profile, get_user_profile, search_memories,
             consolidate_memories]

server 'local'   -> web_search, search, search_routing_tool, rag_routing_tool
server 'agentic' -> the 13 MCP tools
```

Round trips through the protocol, via `/admin/tools/{name}/invoke`:

- `expand_query{"query": "faiss"}` → `{"original": "faiss", "expanded": [], "all": ["faiss"]}`
- `save_memory{"text": "..."}` → `{"status": "ok", "memory_id": "mem_9c836f34..."}`

## Risks

- **Name collisions.** A remote tool named like a local one replaces it —
  `ToolRegistry.register` is last-write-wins. Servers are trusted-by-configuration,
  so this is accepted rather than namespaced; worth revisiting if untrusted
  servers are ever configured.
- **Latency.** A session per call adds a round trip. Acceptable next to model
  inference; revisit with pooling if it ever matters.
- **Trust.** A configured MCP server can put arbitrary tool descriptions in front
  of the model. Only configure servers you trust.
