import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ToolApprovalCard } from "../ToolApprovalCard";
import type { ToolApprovalView } from "../../types";

const approval: ToolApprovalView = {
  id: "a1",
  tool_name: "send_email",
  arguments: {
    recipient: "reader@example.com",
    options: { urgent: true },
    subject: "<img src=x onerror=alert(1)>",
  },
  expires_at: "2026-06-27T12:00:30.000Z",
};

afterEach(() => {
  vi.useRealTimers();
});

describe("ToolApprovalCard", () => {
  it("shows the tool, safe argument text, and expiration countdown", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-06-27T12:00:00.000Z"));

    const { container } = render(
      <ToolApprovalCard approval={approval} onDecision={vi.fn()} />,
    );

    expect(
      screen.getByRole("region", { name: "Approval required for send_email" }),
    ).toBeInTheDocument();
    expect(screen.getByText("reader@example.com")).toBeInTheDocument();
    expect(screen.getByText('{"urgent":true}')).toBeInTheDocument();
    expect(screen.getByText("<img src=x onerror=alert(1)>")).toBeInTheDocument();
    expect(container.querySelector("img")).not.toBeInTheDocument();
    expect(screen.getByText("Expires in 30 seconds")).toBeInTheDocument();

    act(() => vi.advanceTimersByTime(1_000));
    expect(screen.getByText("Expires in 29 seconds")).toBeInTheDocument();
  });

  it.each([
    ["Approve", "approve"],
    ["Deny", "deny"],
  ] as const)("sends the %s decision", async (buttonName, decision) => {
    const onDecision = vi.fn().mockResolvedValue(undefined);
    render(<ToolApprovalCard approval={approval} onDecision={onDecision} />);

    await userEvent.click(screen.getByRole("button", { name: buttonName }));

    expect(onDecision).toHaveBeenCalledWith(decision);
  });

  it("locks both actions while submitting and after success", async () => {
    let resolveDecision!: () => void;
    const onDecision = vi.fn(
      () => new Promise<void>((resolve) => { resolveDecision = resolve; }),
    );
    const user = userEvent.setup();
    render(<ToolApprovalCard approval={approval} onDecision={onDecision} />);

    await user.click(screen.getByRole("button", { name: "Approve" }));
    expect(screen.getByRole("button", { name: "Approve" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Deny" })).toBeDisabled();

    await act(async () => resolveDecision());
    expect(screen.getByRole("button", { name: "Approve" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Deny" })).toBeDisabled();
    expect(screen.getByText("Decision submitted")).toBeInTheDocument();
  });

  it("shows an endpoint error and allows a retry", async () => {
    const onDecision = vi.fn().mockRejectedValue(new Error("Approval expired"));
    render(<ToolApprovalCard approval={approval} onDecision={onDecision} />);

    await userEvent.click(screen.getByRole("button", { name: "Deny" }));

    expect(screen.getByRole("alert")).toHaveTextContent("Approval expired");
    expect(screen.getByRole("button", { name: "Approve" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Deny" })).toBeEnabled();
  });
});
