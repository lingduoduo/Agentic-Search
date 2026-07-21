import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Transcript } from "../Transcript";
import type { ConversationTurn } from "../../types";

describe("Transcript", () => {
  it("renders user and assistant turns in order", () => {
    const turns: ConversationTurn[] = [
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
    ];
    render(<Transcript turns={turns} />);
    expect(screen.getByText("hi")).toBeInTheDocument();
    expect(screen.getByText("hello")).toBeInTheDocument();
  });

  it("renders an assistant turn's tool calls", () => {
    const turns: ConversationTurn[] = [
      {
        role: "assistant",
        content: "done",
        toolCalls: [
          {
            tool_name: "search",
            status: "completed",
            arguments: {},
            result_summary: "3 items",
            latency_ms: 10,
            error: null,
          },
        ],
      },
    ];
    render(<Transcript turns={turns} />);
    expect(screen.getByText(/search/)).toBeInTheDocument();
  });

  it("renders nothing for an empty transcript", () => {
    const { container } = render(<Transcript turns={[]} />);
    expect(container.querySelector(".transcript")?.children.length ?? 0).toBe(0);
  });
});
