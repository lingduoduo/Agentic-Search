import { describe, it, expect } from "vitest";
import { groupToolsByServer } from "../toolCatalog";
import type { ToolView } from "../types";

function tool(over: Partial<ToolView>): ToolView {
  return {
    name: "t",
    description: "d",
    parameters: {},
    source: "function",
    provider_id: null,
    agent_callable: true,
    user_scoped: false,
    ...over,
  };
}

describe("groupToolsByServer", () => {
  it("puts function tools under 'local'", () => {
    const out = groupToolsByServer([tool({ name: "a" }), tool({ name: "b" })]);
    expect(out).toHaveLength(1);
    expect(out[0].name).toBe("local");
    expect(out[0].tools.map((t) => t.name)).toEqual(["a", "b"]);
  });

  it("groups openapi and mcp tools by provider_id", () => {
    const out = groupToolsByServer([
      tool({ name: "echo", source: "openapi", provider_id: "prov-1" }),
      tool({ name: "remote", source: "mcp", provider_id: "my-mcp" }),
      tool({ name: "other", source: "openapi", provider_id: "prov-1" }),
    ]);
    expect(out.map((s) => s.name)).toEqual(["prov-1", "my-mcp"]);
    expect(out[0].tools.map((t) => t.name)).toEqual(["echo", "other"]);
  });

  it("preserves input order within a server", () => {
    const out = groupToolsByServer([
      tool({ name: "z" }),
      tool({ name: "a" }),
      tool({ name: "m" }),
    ]);
    expect(out[0].tools.map((t) => t.name)).toEqual(["z", "a", "m"]);
  });

  it("falls back to local when an openapi tool has no provider_id", () => {
    // Rather than inventing an "undefined" server group.
    const out = groupToolsByServer([
      tool({ name: "orphan", source: "openapi", provider_id: null }),
    ]);
    expect(out.map((s) => s.name)).toEqual(["local"]);
  });

  it("carries the agent_callable and user_scoped flags through", () => {
    const out = groupToolsByServer([
      tool({ name: "hidden", agent_callable: false }),
      tool({ name: "scoped", user_scoped: true }),
    ]);
    const byName = Object.fromEntries(out[0].tools.map((t) => [t.name, t]));
    expect(byName.hidden.agent_callable).toBe(false);
    expect(byName.scoped.user_scoped).toBe(true);
  });

  it("returns no servers for no tools", () => {
    expect(groupToolsByServer([])).toEqual([]);
  });
});
