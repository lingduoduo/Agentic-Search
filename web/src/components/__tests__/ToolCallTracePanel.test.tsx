import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ToolCallTracePanel } from "../ToolCallTracePanel";
import type { ToolCallTraceView } from "../../types";

const completedCall: ToolCallTraceView = {
  tool_name: "search_tool",
  status: "completed",
  arguments: { query: "test" },
  result_summary: "some result",
  latency_ms: 123,
  error: null,
};

const failedCall: ToolCallTraceView = {
  tool_name: "broken_tool",
  status: "failed",
  arguments: { query: "fail" },
  result_summary: "",
  latency_ms: 45,
  error: "connection refused",
};

describe("ToolCallTracePanel", () => {
  it("renders null when calls is empty", () => {
    const { container } = render(<ToolCallTracePanel calls={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders two cards for two completed calls", () => {
    const secondCall: ToolCallTraceView = { ...completedCall, tool_name: "other_tool" };
    render(<ToolCallTracePanel calls={[completedCall, secondCall]} />);
    const cards = document.querySelectorAll(".tool-trace-card");
    expect(cards).toHaveLength(2);
  });

  it("completed card has tool-trace-status--ok containing checkmark", () => {
    render(<ToolCallTracePanel calls={[completedCall]} />);
    const okStatus = document.querySelector(".tool-trace-status--ok");
    expect(okStatus).not.toBeNull();
    expect(okStatus?.textContent).toBe("✓");
  });

  it("failed card has tool-trace-card--failed class", () => {
    render(<ToolCallTracePanel calls={[failedCall]} />);
    const failedCard = document.querySelector(".tool-trace-card--failed");
    expect(failedCard).not.toBeNull();
  });

  it("failed card has tool-trace-status--failed containing cross mark", () => {
    render(<ToolCallTracePanel calls={[failedCall]} />);
    const failedStatus = document.querySelector(".tool-trace-status--failed");
    expect(failedStatus).not.toBeNull();
    expect(failedStatus?.textContent).toBe("✗");
  });

  it("shows latency as '123 ms'", () => {
    render(<ToolCallTracePanel calls={[completedCall]} />);
    expect(screen.getByText("123 ms")).toBeInTheDocument();
  });

  it("shows arguments as formatted JSON in a code element", () => {
    render(<ToolCallTracePanel calls={[completedCall]} />);
    const codeElements = document.querySelectorAll("code");
    const jsonText = JSON.stringify({ query: "test" }, null, 2);
    const found = Array.from(codeElements).some((el) => el.textContent === jsonText);
    expect(found).toBe(true);
  });

  it("shows result_summary for completed call", () => {
    render(<ToolCallTracePanel calls={[completedCall]} />);
    expect(screen.getByText("some result")).toBeInTheDocument();
  });

  it("shows error message for failed call", () => {
    render(<ToolCallTracePanel calls={[failedCall]} />);
    expect(screen.getByText("connection refused")).toBeInTheDocument();
  });
});
