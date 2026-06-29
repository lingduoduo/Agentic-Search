import { render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { ServerHealthGrid } from "../debug/ServerHealthGrid";

vi.mock("../../api", () => ({ getServerHealth: vi.fn() }));
import { getServerHealth } from "../../api";
const mockHealth = vi.mocked(getServerHealth);

beforeEach(() => mockHealth.mockReset());

describe("ServerHealthGrid — health grid", () => {
  it("renders up/down per server", async () => {
    mockHealth.mockResolvedValue({
      servers: [
        { name: "web", url: "self", status: "up" },
        { name: "retrieval", url: "http://r:8001", status: "down" },
      ],
    });
    render(<ServerHealthGrid answer="" citations={[]} />);
    await waitFor(() => expect(screen.getByText("retrieval")).toBeInTheDocument());
    expect(screen.getByTestId("health-web")).toHaveTextContent(/up/i);
    expect(screen.getByTestId("health-retrieval")).toHaveTextContent(/down/i);
  });

  it("renders gracefully with an empty server list", async () => {
    // /api/debug/health always returns 200 up/down; empty list must not crash.
    mockHealth.mockResolvedValue({ servers: [] });
    render(<ServerHealthGrid answer="" citations={[]} />);
    await waitFor(() =>
      expect(screen.getByText(/server health/i)).toBeInTheDocument(),
    );
  });
});

describe("ServerHealthGrid — grounding debug", () => {
  beforeEach(() =>
    mockHealth.mockResolvedValue({ servers: [] }),
  );

  it("labels 'grounded, no answer' when citations exist but answer is empty", async () => {
    render(<ServerHealthGrid answer="" citations={["[D1]"]} />);
    await waitFor(() =>
      expect(screen.getByText(/grounded, no answer/i)).toBeInTheDocument(),
    );
  });

  it("labels 'answer, ungrounded' when answer exists but no citations", async () => {
    render(<ServerHealthGrid answer="Hello." citations={[]} />);
    await waitFor(() =>
      expect(screen.getByText(/answer, ungrounded/i)).toBeInTheDocument(),
    );
  });

  it("labels 'grounded answer' when both present", async () => {
    render(<ServerHealthGrid answer="Hello." citations={["[D1]"]} />);
    await waitFor(() =>
      expect(screen.getByText(/grounded answer/i)).toBeInTheDocument(),
    );
  });
});
