import type { CatalogServer, ToolView } from "./types";

/**
 * Group a flat /admin/tools listing into servers, the way the dev-console
 * catalog is already grouped server-side.
 *
 * Mirrors `catalog_from_registry` in src/internal/tools/semantic_router.py:
 * openapi/mcp tools group by provider_id (the provider or MCP server they came
 * from), everything else lands under "local", input order is preserved within a
 * server, and empty servers are omitted. Kept in one place because the same
 * rule now exists in two languages and would otherwise drift.
 */
export function groupToolsByServer(tools: ToolView[]): CatalogServer[] {
  const servers = new Map<string, CatalogServer>();

  for (const tool of tools) {
    const grouped = tool.source === "openapi" || tool.source === "mcp";
    // A provider-less openapi/mcp entry should not create an "undefined"
    // server; fall back to local rather than inventing a group.
    const serverName = grouped && tool.provider_id ? tool.provider_id : "local";

    let server = servers.get(serverName);
    if (!server) {
      server = { name: serverName, description: serverName, tools: [] };
      servers.set(serverName, server);
    }
    server.tools.push({
      name: tool.name,
      description: tool.description,
      source: tool.source,
      server: serverName,
      agent_callable: tool.agent_callable,
      user_scoped: tool.user_scoped,
    });
  }

  return [...servers.values()].filter((s) => s.tools.length > 0);
}
