import { render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { WorkerMonitor } from "../debug/WorkerMonitor";

vi.mock("../../api", () => ({ getWorkerMetrics: vi.fn() }));
import { getWorkerMetrics } from "../../api";
const mockMetrics = vi.mocked(getWorkerMetrics);

beforeEach(() => mockMetrics.mockReset());

describe("WorkerMonitor", () => {
  it("renders the indexing snapshot metrics", async () => {
    mockMetrics.mockResolvedValue({
      metrics: {
        process_memory_mb: 128.5,
        pending_index_attempts: 3,
        in_progress_index_attempts: 1,
        active_connectors: 2,
        total_documents: 42,
        timestamp: "2026-06-29T12:00:00Z",
      },
    });
    render(<WorkerMonitor />);
    await waitFor(() =>
      expect(screen.getByTestId("metric-pending")).toHaveTextContent("3"),
    );
    expect(screen.getByTestId("metric-in_progress")).toHaveTextContent("1");
    expect(screen.getByTestId("metric-documents")).toHaveTextContent("42");
    expect(screen.getByTestId("metric-connectors")).toHaveTextContent("2");
  });

  it("shows 'no data yet' when metrics are null", async () => {
    mockMetrics.mockResolvedValue({ metrics: null });
    render(<WorkerMonitor />);
    await waitFor(() =>
      expect(screen.getByText(/no data yet/i)).toBeInTheDocument(),
    );
  });
});
