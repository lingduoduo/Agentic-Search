import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { RequestTracePanel } from "../debug/RequestTracePanel";
import type { ControlFlowEventView } from "../../types";

function ev(
  sequence: number,
  component: string,
  duration_ms: number | null,
  details: Record<string, string | number | boolean | null> = {},
): ControlFlowEventView {
  return {
    sequence,
    timestamp: `2026-06-29T12:00:00.00${sequence}Z`,
    turn: 1,
    component,
    action: "did_thing",
    status: "completed",
    duration_ms,
    details,
  };
}

describe("RequestTracePanel", () => {
  it("shows an empty state with no events", () => {
    render(<RequestTracePanel events={[]} />);
    expect(screen.getByText(/no trace yet/i)).toBeInTheDocument();
  });

  it("renders one bar per event with width proportional to duration_ms", () => {
    render(
      <RequestTracePanel
        events={[ev(1, "planner", 100), ev(2, "search_tool", 50)]}
      />,
    );
    const bar1 = screen.getByTestId("trace-bar-1");
    const bar2 = screen.getByTestId("trace-bar-2");
    // bar1 is the longest (100ms) → 100%; bar2 is half → 50%
    expect(bar1.style.width).toBe("100%");
    expect(bar2.style.width).toBe("50%");
    expect(screen.getByText(/planner/)).toBeInTheDocument();
    expect(screen.getByText(/search_tool/)).toBeInTheDocument();
  });

  it("reveals details when a span is clicked", () => {
    render(
      <RequestTracePanel
        events={[ev(1, "route", 12, { retriever: "hybrid", confidence: 0.9 })]}
      />,
    );
    expect(screen.queryByText(/retriever/i)).not.toBeInTheDocument();
    fireEvent.click(screen.getByTestId("trace-bar-1"));
    expect(screen.getByText(/retriever/i)).toBeInTheDocument();
    expect(screen.getByText(/hybrid/i)).toBeInTheDocument();
  });
});
