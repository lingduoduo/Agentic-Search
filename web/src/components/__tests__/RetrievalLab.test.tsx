import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { RetrievalLab } from "../debug/RetrievalLab";
import type { DebugRetrievalOutcome, RetrievalMode } from "../../types";

vi.mock("../../api", () => ({ runDebugRetrieval: vi.fn() }));
import { runDebugRetrieval } from "../../api";
const mockRun = vi.mocked(runDebugRetrieval);

function ok(mode: string, docId: string): DebugRetrievalOutcome {
  return {
    status: 200,
    ok: true,
    detail: null,
    data: {
      results: [{ doc_id: docId, title: `${mode} doc`, score: 0.5 }],
      retrieval_mode: mode,
      executed_queries: ["q"],
      latency_ms: 3.0,
    },
  };
}

beforeEach(() => mockRun.mockReset());

describe("RetrievalLab", () => {
  it("runs all four modes and renders their result rows", async () => {
    mockRun.mockImplementation(async (mode: RetrievalMode) => ok(mode, `${mode}-1`));

    render(<RetrievalLab />);
    fireEvent.change(screen.getByLabelText(/query/i), {
      target: { value: "vector database" },
    });
    fireEvent.click(screen.getByRole("button", { name: /run/i }));

    await waitFor(() => {
      expect(screen.getByText("sparse-1")).toBeInTheDocument();
    });
    for (const mode of ["sparse", "dense", "hybrid", "graph"]) {
      expect(mockRun).toHaveBeenCalledWith(
        mode,
        expect.objectContaining({ query: "vector database" }),
      );
      expect(screen.getByText(`${mode}-1`)).toBeInTheDocument();
    }
  });

  it("shows an endpoint-not-available message on 404", async () => {
    mockRun.mockResolvedValue({
      status: 404,
      ok: false,
      data: null,
      detail: "Not Found",
    });

    render(<RetrievalLab />);
    fireEvent.change(screen.getByLabelText(/query/i), { target: { value: "q" } });
    fireEvent.click(screen.getByRole("button", { name: /run/i }));

    await waitFor(() => {
      expect(screen.getAllByText(/endpoint not available/i).length).toBeGreaterThan(0);
    });
  });

  it("flags dense unavailable on 503", async () => {
    mockRun.mockImplementation(async (mode: RetrievalMode) =>
      mode === "dense"
        ? { status: 503, ok: false, data: null, detail: "Dense search not configured" }
        : ok(mode, `${mode}-1`),
    );

    render(<RetrievalLab />);
    fireEvent.change(screen.getByLabelText(/query/i), { target: { value: "q" } });
    fireEvent.click(screen.getByRole("button", { name: /run/i }));

    await waitFor(() => {
      expect(screen.getByText(/dense leg unavailable/i)).toBeInTheDocument();
    });
  });
});
