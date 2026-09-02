import { render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { LatencyPanel } from "../LatencyPanel";
import * as api from "../../../api";

const row = {
  method: "POST",
  route: "/api/agent",
  count: 12,
  errors: 0,
  p50_ms: 812.4,
  p95_ms: 2140.9,
  max_ms: 2210,
};

describe("LatencyPanel", () => {
  it("renders one row per route with its percentiles", async () => {
    vi.spyOn(api, "getRouteLatency").mockResolvedValue({ routes: [row] });

    render(<LatencyPanel />);

    await waitFor(() =>
      expect(screen.getByText("POST /api/agent")).toBeInTheDocument(),
    );
    expect(screen.getByText("812.4")).toBeInTheDocument();
    expect(screen.getByText("2140.9")).toBeInTheDocument();
  });

  it("shows an empty state before any request is recorded", async () => {
    vi.spyOn(api, "getRouteLatency").mockResolvedValue({ routes: [] });

    render(<LatencyPanel />);

    await waitFor(() =>
      expect(screen.getByText(/no requests recorded yet/i)).toBeInTheDocument(),
    );
  });

  it("renders a dash instead of crashing on a missing percentile", async () => {
    vi.spyOn(api, "getRouteLatency").mockResolvedValue({
      routes: [{ ...row, p95_ms: null as unknown as number }],
    });

    render(<LatencyPanel />);

    await waitFor(() =>
      expect(screen.getByText("POST /api/agent")).toBeInTheDocument(),
    );
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("falls back to the empty state when the endpoint is unreachable", async () => {
    vi.spyOn(api, "getRouteLatency").mockRejectedValue(new Error("404"));

    render(<LatencyPanel />);

    await waitFor(() =>
      expect(screen.getByText(/no requests recorded yet/i)).toBeInTheDocument(),
    );
  });
});
