import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import { ToolCatalog } from "../ToolCatalog";
import type { CatalogServer } from "../../types";

const servers: CatalogServer[] = [
  {
    name: "local",
    description: "local",
    tools: [
      {
        name: "web_search",
        description: "Search the web",
        source: "function",
        server: "local",
        agent_callable: true,
        user_scoped: false,
      },
      {
        name: "search",
        description: "Search the corpus",
        source: "function",
        server: "local",
        agent_callable: false,
        user_scoped: false,
      },
      {
        name: "add_memory",
        description: "Remember something",
        source: "function",
        server: "local",
        agent_callable: true,
        user_scoped: true,
      },
    ],
  },
];

function noop() {}

describe("ToolCatalog", () => {
  it("shows the registered count and each server's tools", () => {
    render(
      <ToolCatalog
        servers={servers}
        registeredCount={3}
        discovery={null}
        onDiscover={noop}
      />,
    );
    expect(screen.getByText(/3 registered across 1 server/)).toBeInTheDocument();
    expect(screen.getByText("local")).toBeInTheDocument();
    expect(screen.getByText("Search the web")).toBeInTheDocument();
  });

  it("badges only the tools the flags apply to", () => {
    render(
      <ToolCatalog
        servers={servers}
        registeredCount={3}
        discovery={null}
        onDiscover={noop}
      />,
    );
    // One tool has agent_callable: false, one has user_scoped: true.
    expect(screen.getAllByText("not offered to agents")).toHaveLength(1);
    expect(screen.getAllByText("needs sign-in")).toHaveLength(1);
  });

  it("does not badge tools whose flags are unknown", () => {
    // The /api/debug/tools catalog carries no flags, so nothing should render.
    render(
      <ToolCatalog
        servers={[
          {
            name: "local",
            description: "local",
            tools: [
              { name: "x", description: "d", source: "function", server: "local" },
            ],
          },
        ]}
        registeredCount={1}
        discovery={null}
        onDiscover={noop}
      />,
    );
    expect(screen.queryByText("not offered to agents")).not.toBeInTheDocument();
    expect(screen.queryByText("needs sign-in")).not.toBeInTheDocument();
  });

  it("fires onDiscover with the trimmed query", async () => {
    const onDiscover = vi.fn();
    render(
      <ToolCatalog
        servers={servers}
        registeredCount={3}
        discovery={null}
        onDiscover={onDiscover}
      />,
    );
    await userEvent.type(screen.getByLabelText("Discovery query"), "  weather  ");
    await userEvent.click(screen.getByRole("button", { name: "Discover" }));
    expect(onDiscover).toHaveBeenCalledWith("weather");
  });

  it("does not fire onDiscover for a blank query", async () => {
    const onDiscover = vi.fn();
    render(
      <ToolCatalog
        servers={servers}
        registeredCount={3}
        discovery={null}
        onDiscover={onDiscover}
      />,
    );
    await userEvent.click(screen.getByRole("button", { name: "Discover" }));
    expect(onDiscover).not.toHaveBeenCalled();
  });

  it("renders ranked discovery results", () => {
    render(
      <ToolCatalog
        servers={servers}
        registeredCount={3}
        discovery={{
          request: "weather",
          stage1_servers: [],
          stage2_tools: {},
          final_tools: [{ name: "web_search", server: "local", score: 0.874 }],
        }}
        onDiscover={noop}
      />,
    );
    expect(screen.getByText(/\(local, 0\.874\)/)).toBeInTheDocument();
  });

  it("says so when discovery matched nothing", () => {
    render(
      <ToolCatalog
        servers={servers}
        registeredCount={3}
        discovery={{
          request: "zzz",
          stage1_servers: [],
          stage2_tools: {},
          final_tools: [],
        }}
        onDiscover={noop}
      />,
    );
    expect(screen.getByText("No tools matched.")).toBeInTheDocument();
  });

  it("shows a note instead of the catalog when one is given", () => {
    render(
      <ToolCatalog
        servers={[]}
        registeredCount={0}
        discovery={null}
        onDiscover={noop}
        note="Tool inventory needs an admin session."
      />,
    );
    expect(
      screen.getByText("Tool inventory needs an admin session."),
    ).toBeInTheDocument();
    expect(screen.queryByText(/registered across/)).not.toBeInTheDocument();
  });

  it("shows a loading state while servers are null", () => {
    render(
      <ToolCatalog
        servers={null}
        registeredCount={0}
        discovery={null}
        onDiscover={noop}
      />,
    );
    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });

  it("warns that ranking is not how the agent picks", () => {
    // The whole reason the panel exists: not mistaking this for agent behavior.
    render(
      <ToolCatalog
        servers={servers}
        registeredCount={3}
        discovery={null}
        onDiscover={noop}
      />,
    );
    expect(screen.getByText(/no model involved/)).toBeInTheDocument();
  });
});
