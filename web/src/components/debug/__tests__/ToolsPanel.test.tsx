import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import { ToolsPanel } from "../ToolsPanel";
import * as api from "../../../api";

describe("ToolsPanel", () => {
  it("renders the registered count and catalog grouped by server", async () => {
    vi.spyOn(api, "getDebugTools").mockResolvedValue({
      registered: [
        { name: "search", description: "d", parameters: {}, source: "function", provider_id: null },
      ],
      catalog: [
        {
          name: "local",
          description: "local",
          tools: [{ name: "search", description: "Search the corpus", source: "function", server: "local" }],
        },
      ],
    });
    render(<ToolsPanel />);
    await waitFor(() => expect(screen.getByText("1 registered")).toBeInTheDocument());
    expect(screen.getByText("local")).toBeInTheDocument();
    expect(screen.getByText("Search the corpus")).toBeInTheDocument();
  });

  it("renders ranked results after a discovery query", async () => {
    vi.spyOn(api, "getDebugTools").mockResolvedValue({ registered: [], catalog: [] });
    const discover = vi.spyOn(api, "discoverTools").mockResolvedValue({
      request: "find docs",
      stage1_servers: [{ name: "local", score: 0.9 }],
      stage2_tools: {},
      final_tools: [{ name: "search", server: "local", score: 0.87 }],
    });
    render(<ToolsPanel />);
    await screen.findByText("0 registered");
    await userEvent.type(screen.getByLabelText("Discovery query"), "find docs");
    await userEvent.click(screen.getByText("Discover"));
    await waitFor(() => expect(discover).toHaveBeenCalledWith("find docs"));
    expect(screen.getByText(/search/)).toBeInTheDocument();
  });
});
