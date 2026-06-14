import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SourceGrid } from "../SourceGrid";
import type { SourceDocumentView } from "../../types";

const doc: SourceDocumentView = {
  id: "D1",
  citation: "[D1]",
  title: "FAISS paper",
  content: "Dense retrieval with FAISS.",
  url: "https://example.test/faiss",
  score: 0.95,
  metadata: { source: "retrieval" },
};

const docNoUrl: SourceDocumentView = {
  ...doc,
  id: "D2",
  citation: "[D2]",
  title: "Local doc",
  url: null,
};

describe("SourceGrid", () => {
  it("renders empty state when documents list is empty", () => {
    render(<SourceGrid documents={[]} />);
    expect(screen.getByText(/no sources yet/i)).toBeInTheDocument();
  });

  it("renders document title", () => {
    render(<SourceGrid documents={[doc]} />);
    expect(screen.getByText("FAISS paper")).toBeInTheDocument();
  });

  it("renders a link when url is present", () => {
    render(<SourceGrid documents={[doc]} />);
    const link = screen.getByRole("link", { name: /faiss paper/i });
    expect(link).toHaveAttribute("href", "https://example.test/faiss");
    expect(link).toHaveAttribute("target", "_blank");
  });

  it("renders heading instead of link when url is absent", () => {
    render(<SourceGrid documents={[docNoUrl]} />);
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Local doc" })).toBeInTheDocument();
  });

  it("renders the citation label", () => {
    render(<SourceGrid documents={[doc]} />);
    expect(screen.getByText("[D1]")).toBeInTheDocument();
  });

  it("renders multiple documents", () => {
    render(<SourceGrid documents={[doc, docNoUrl]} />);
    expect(screen.getByText("FAISS paper")).toBeInTheDocument();
    expect(screen.getByText("Local doc")).toBeInTheDocument();
  });

  it("shows mmr_rank badge from metadata", () => {
    const ranked = { ...doc, metadata: { source: "Local Retrieval", mmr_rank: 3 } };
    render(<SourceGrid documents={[ranked]} />);
    expect(screen.getByText("#3")).toBeInTheDocument();
  });

  it("applies green color to high score badge", () => {
    const { container } = render(<SourceGrid documents={[doc]} />); // doc.score = 0.95
    const badge = container.querySelector(".score-badge") as HTMLElement;
    expect(badge).not.toBeNull();
    expect(badge.style.color).toBe("rgb(34, 197, 94)");
  });
});
