import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { ControlFlowTracePanel } from "../ControlFlowTracePanel";
import type { ControlFlowEventView } from "../../types";

const event1: ControlFlowEventView = {
  sequence: 1,
  timestamp: "2026-06-27T12:00:00.000Z",
  turn: 1,
  component: "planner",
  action: "search_planned",
  status: "decided",
  duration_ms: null,
  details: { decision: "search", safe_message: "secret" },
};

const event2: ControlFlowEventView = {
  sequence: 2,
  timestamp: "2026-06-27T12:00:00.010Z",
  turn: 1,
  component: "evidence_judge",
  action: "evidence_evaluated",
  status: "completed",
  duration_ms: 3,
  details: { evidence_score: 0.72, sufficient: true, document_count: 5 },
};

describe("ControlFlowTracePanel", () => {
  it("renders live events in sequence order and hides non-summary details", () => {
    render(<ControlFlowTracePanel events={[event2, event1]} live />);

    const items = screen.getAllByRole("listitem");
    expect(items[0]).toHaveTextContent("Planner");
    expect(items[1]).toHaveTextContent("Evidence judge");
    expect(items[1]).toHaveTextContent("0.72");
    expect(screen.queryByText("secret")).not.toBeInTheDocument();
  });

  it("collapses a completed trace and expands on request", async () => {
    const user = userEvent.setup();
    render(<ControlFlowTracePanel events={[event1]} live={false} />);

    expect(screen.queryByRole("list")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /show control flow/i }));
    expect(screen.getByRole("list")).toBeVisible();
  });

  it("announces failed and active statuses", () => {
    render(
      <ControlFlowTracePanel
        live
        events={[
          { ...event1, status: "failed" },
          { ...event2, status: "started" },
        ]}
      />,
    );

    expect(screen.getByText("Failed")).toBeVisible();
    expect(screen.getByText("In progress")).toBeVisible();
  });

  it("renders nothing for an empty trace", () => {
    const { container } = render(<ControlFlowTracePanel events={[]} live />);
    expect(container).toBeEmptyDOMElement();
  });
});
