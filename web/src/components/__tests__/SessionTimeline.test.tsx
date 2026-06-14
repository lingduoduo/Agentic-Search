import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SessionTimeline } from "../SessionTimeline";
import type { ChatMessageView } from "../../types";

const userMsg: ChatMessageView = { role: "user", content: "Hello" };
const assistantMsg: ChatMessageView = { role: "assistant", content: "Hi there" };
const assistantWithRounds: ChatMessageView = {
  role: "assistant",
  content: "Answer",
  metadata: { rounds_used: 3 },
};
const assistantWithTurns: ChatMessageView = {
  role: "assistant",
  content: "Tool answer",
  metadata: { num_turns: 2 },
};

describe("SessionTimeline", () => {
  it("renders empty state when no messages", () => {
    render(<SessionTimeline messages={[]} />);
    expect(screen.getByText(/start a query/i)).toBeInTheDocument();
  });

  it("renders user and assistant messages", () => {
    render(<SessionTimeline messages={[userMsg, assistantMsg]} />);
    expect(screen.getByText("Hello")).toBeInTheDocument();
    expect(screen.getByText("Hi there")).toBeInTheDocument();
  });

  it("shows rounds_used badge when present", () => {
    render(<SessionTimeline messages={[assistantWithRounds]} />);
    expect(screen.getByText(/3 rounds/i)).toBeInTheDocument();
  });

  it("shows num_turns badge when present", () => {
    render(<SessionTimeline messages={[assistantWithTurns]} />);
    expect(screen.getByText(/2 turns/i)).toBeInTheDocument();
  });

  it("does not show badges when metadata is absent", () => {
    render(<SessionTimeline messages={[assistantMsg]} />);
    expect(screen.queryByText(/round/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/turn/i)).not.toBeInTheDocument();
  });

  it("shows rounds_used=1 as '1 rounds' badge", () => {
    const msg: ChatMessageView = {
      role: "assistant",
      content: "x",
      metadata: { rounds_used: 1 },
    };
    render(<SessionTimeline messages={[msg]} />);
    expect(screen.getByText("1 rounds")).toBeInTheDocument();
  });
});
