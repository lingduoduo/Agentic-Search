import { render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { EvalResultsPanel } from "../EvalResultsPanel";
import * as api from "../../../api";

describe("EvalResultsPanel", () => {
  it("renders a card with metric rows per result file", async () => {
    vi.spyOn(api, "getEvalResults").mockResolvedValue({
      results: [
        { name: "beir.json", modified: 1_700_000_000, metrics: { "recall@10": 0.5 } },
      ],
    });
    render(<EvalResultsPanel />);
    await waitFor(() => expect(screen.getByText("beir.json")).toBeInTheDocument());
    expect(screen.getByText("recall@10")).toBeInTheDocument();
    expect(screen.getByText("0.5000")).toBeInTheDocument();
  });

  it("shows an empty state when there are no results", async () => {
    vi.spyOn(api, "getEvalResults").mockResolvedValue({ results: [] });
    render(<EvalResultsPanel />);
    await waitFor(() =>
      expect(screen.getByText(/no eval results yet/i)).toBeInTheDocument(),
    );
  });
});
