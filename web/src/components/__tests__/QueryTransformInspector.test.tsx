import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryTransformInspector } from "../debug/QueryTransformInspector";
import type { QueryTransformResult } from "../../types";

vi.mock("../../api", () => ({ runQueryTransform: vi.fn() }));
import { runQueryTransform } from "../../api";
const mockRun = vi.mocked(runQueryTransform);

beforeEach(() => mockRun.mockReset());

const active: QueryTransformResult = {
  original: "vector db",
  variants: ["what is a vector database", "how do embeddings index", "vector db"],
  merged_filters: { year: 2024 },
  active: true,
  legs: { sub_queries: ["what is a vector database", "how do embeddings index"] },
};

describe("QueryTransformInspector", () => {
  it("runs the transform and renders variants", async () => {
    mockRun.mockResolvedValue(active);
    render(<QueryTransformInspector />);
    fireEvent.change(screen.getByLabelText(/query/i), {
      target: { value: "vector db" },
    });
    fireEvent.click(screen.getByRole("button", { name: /transform/i }));

    await waitFor(() => {
      expect(mockRun).toHaveBeenCalledWith("vector db");
    });
    expect(
      screen.getByText("what is a vector database"),
    ).toBeInTheDocument();
    expect(screen.getByText(/how do embeddings index/i)).toBeInTheDocument();
  });

  it("shows 'no transform active' when pipeline is off", async () => {
    mockRun.mockResolvedValue({
      original: "vector db",
      variants: ["vector db"],
      merged_filters: {},
      active: false,
      legs: {},
    });
    render(<QueryTransformInspector />);
    fireEvent.change(screen.getByLabelText(/query/i), {
      target: { value: "vector db" },
    });
    fireEvent.click(screen.getByRole("button", { name: /transform/i }));

    await waitFor(() => {
      expect(screen.getByText(/no transform active/i)).toBeInTheDocument();
    });
  });
});
