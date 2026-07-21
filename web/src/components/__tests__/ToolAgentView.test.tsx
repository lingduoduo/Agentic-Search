import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ToolAgentView } from "../ToolAgentView";
import * as api from "../../api";

describe("ToolAgentView", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("renders streamed tool calls and the answer", async () => {
    async function* fake() {
      yield { type: "progress", turn: 1, text: "search · 3 docs" } as const;
      yield {
        type: "tool_call",
        tool_name: "search",
        status: "completed",
        arguments: {},
        result_summary: "3 items",
        latency_ms: 10,
        error: null,
      } as const;
      yield { type: "answer", text: "done answer" } as const;
      yield { type: "done", session_id: "s1", tool_calls: [], num_turns: 1 } as const;
    }
    vi.spyOn(api, "sendToolMessage").mockImplementation(fake as never);

    render(<ToolAgentView />);
    fireEvent.change(screen.getByLabelText("Tool agent message"), {
      target: { value: "find X" },
    });
    fireEvent.click(screen.getByText("Send"));

    await waitFor(() => expect(screen.getByText("done answer")).toBeInTheDocument());
    expect(screen.getByText(/search · 3 docs/)).toBeInTheDocument();
  });

  it("shows the no-model banner on NO_LOCAL_MODEL", async () => {
    vi.spyOn(api, "sendToolMessage").mockImplementation((() => {
      async function* g() {
        throw new Error("NO_LOCAL_MODEL");
        // eslint-disable-next-line no-unreachable
        yield undefined as never;
      }
      return g();
    }) as never);

    render(<ToolAgentView />);
    fireEvent.change(screen.getByLabelText("Tool agent message"), {
      target: { value: "hi" },
    });
    fireEvent.click(screen.getByText("Send"));

    await waitFor(() =>
      expect(screen.getByText(/needs a local model/)).toBeInTheDocument(),
    );
  });
});
