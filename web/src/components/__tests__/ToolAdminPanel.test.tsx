import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ToolAdminPanel } from "../ToolAdminPanel";
import * as api from "../../api";
import type { ToolView } from "../../types";

const WEB_SEARCH: ToolView = {
  name: "web_search",
  description: "Search the web for information. Pass multiple queries to search in parallel.",
  parameters: {
    type: "object",
    properties: {
      queries: {
        type: "array",
        items: { type: "string" },
        description: "One or more search queries to run in parallel.",
      },
    },
    required: ["queries"],
  },
  source: "function",
  provider_id: null,
  agent_callable: true,
  user_scoped: false,
};

const NESTED: ToolView = {
  ...WEB_SEARCH,
  name: "complex_tool",
  parameters: {
    type: "object",
    properties: { filters: { type: "object", properties: { a: { type: "string" } } } },
  },
};

async function openInvoke(tool: ToolView) {
  vi.spyOn(api, "listTools").mockResolvedValue([tool]);
  render(<ToolAdminPanel />);
  fireEvent.click(await screen.findByRole("button", { name: /test/i }));
}

describe("ToolAdminPanel invoke dialog", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("renders a labelled input from the schema instead of a JSON box", async () => {
    await openInvoke(WEB_SEARCH);

    // The field the schema declares, as a real labelled control.
    expect(await screen.findByLabelText(/queries/i)).toBeTruthy();
    // Its description is shown, not hidden in a placeholder that never renders.
    expect(screen.getByText(/one or more search queries/i)).toBeTruthy();
    // No raw JSON editor by default.
    expect(screen.queryByLabelText(/arguments \(json\)/i)).toBeNull();
  });

  it("sends what the user typed, shaped by the schema", async () => {
    const invokeSpy = vi
      .spyOn(api, "invokeTool")
      .mockResolvedValue({ response: "ok", raw: null, errors: [] });
    await openInvoke(WEB_SEARCH);

    fireEvent.change(await screen.findByLabelText(/queries/i), {
      target: { value: "faiss" },
    });
    fireEvent.click(screen.getByRole("button", { name: /▶ Invoke/i }));

    await waitFor(() =>
      // A string typed into an array-of-strings field is sent as a list.
      expect(invokeSpy).toHaveBeenCalledWith("web_search", {
        arguments: { queries: ["faiss"] },
      }),
    );
  });

  it("refuses to invoke while a required field is empty", async () => {
    const invokeSpy = vi.spyOn(api, "invokeTool");
    await openInvoke(WEB_SEARCH);

    fireEvent.click(await screen.findByRole("button", { name: /▶ Invoke/i }));

    expect(await screen.findByText(/queries is required/i)).toBeTruthy();
    expect(invokeSpy).not.toHaveBeenCalled();
  });

  it("falls back to the JSON editor for a schema a form cannot express", async () => {
    await openInvoke(NESTED);

    expect(await screen.findByLabelText(/arguments \(json\)/i)).toBeTruthy();
    expect(screen.getByText(/nested arguments/i)).toBeTruthy();
  });

  it("offers a JSON escape hatch for supported schemas too", async () => {
    const invokeSpy = vi
      .spyOn(api, "invokeTool")
      .mockResolvedValue({ response: "ok", raw: null, errors: [] });
    await openInvoke(WEB_SEARCH);

    fireEvent.click(await screen.findByRole("button", { name: /edit as json/i }));
    fireEvent.change(screen.getByLabelText(/arguments \(json\)/i), {
      target: { value: '{"queries": ["hand written"]}' },
    });
    fireEvent.click(screen.getByRole("button", { name: /▶ Invoke/i }));

    await waitFor(() =>
      expect(invokeSpy).toHaveBeenCalledWith("web_search", {
        arguments: { queries: ["hand written"] },
      }),
    );
  });
});
