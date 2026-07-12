import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { RequestInspector } from "../debug/RequestInspector";
import type { RequestSnapshot } from "../../types";

vi.mock("../../api", () => ({
  listDebugRequests: vi.fn(),
  getDebugRequest: vi.fn(),
}));

import * as api from "../../api";

const mockList = api.listDebugRequests as ReturnType<typeof vi.fn>;
const mockGet = api.getDebugRequest as ReturnType<typeof vi.fn>;

function snap(id: string): RequestSnapshot {
  return {
    request_id: id,
    query: id,
    created_at: 0,
    route: null,
    route_degraded: null,
    total_ms: null,
    stages: [
      { stage: "final", label: `label-${id}`, timestamp: 0, duration_ms: null, payload: {} },
    ],
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  mockList.mockResolvedValue({
    requests: [
      { request_id: "req-A", query: "alpha", created_at: 1, route: "search", stage_count: 1 },
      { request_id: "req-B", query: "beta", created_at: 2, route: "chat", stage_count: 1 },
    ],
  });
  mockGet.mockImplementation(async (id: string) => snap(id));
});

describe("RequestInspector run selection", () => {
  it("auto-follows the latest request until the user clicks a different run", async () => {
    render(<RequestInspector selectedRequestId="req-A" />);

    // Defaults to following the streamed latest request.
    await waitFor(() => expect(screen.getByText(/final · label-req-A/)).toBeInTheDocument());

    // A manual click must win over the streamed latest id.
    await userEvent.click(screen.getByRole("button", { name: /beta/i }));

    await waitFor(() => expect(mockGet).toHaveBeenLastCalledWith("req-B"));
    expect(await screen.findByText(/final · label-req-B/)).toBeInTheDocument();
  });
});
