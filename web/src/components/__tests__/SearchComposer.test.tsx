import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { SearchComposer } from "../SearchComposer";

const defaultProps = {
  query: "",
  searchUrl: "http://localhost:8001",
  topK: 5,
  mode: "chat_once" as const,
  sourceProvider: "retrieval" as const,
  isLoading: false,
  onQueryChange: vi.fn(),
  onSearchUrlChange: vi.fn(),
  onTopKChange: vi.fn(),
  onModeChange: vi.fn(),
  onSourceProviderChange: vi.fn(),
  onSubmit: vi.fn(),
};

describe("SearchComposer", () => {
  it("renders a textarea and submit button", () => {
    render(<SearchComposer {...defaultProps} />);
    expect(screen.getByRole("textbox", { name: /question/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /search/i })).toBeInTheDocument();
  });

  it("disables submit when query is empty", () => {
    render(<SearchComposer {...defaultProps} query="" />);
    expect(screen.getByRole("button", { name: /search/i })).toBeDisabled();
  });

  it("enables submit when query has content", () => {
    render(<SearchComposer {...defaultProps} query="What is FAISS?" />);
    expect(screen.getByRole("button", { name: /search/i })).not.toBeDisabled();
  });

  it("disables submit while loading", () => {
    render(<SearchComposer {...defaultProps} query="hello" isLoading={true} />);
    expect(screen.getByRole("button")).toBeDisabled();
  });

  it("calls onSubmit when form is submitted", async () => {
    const onSubmit = vi.fn();
    render(<SearchComposer {...defaultProps} query="test" onSubmit={onSubmit} />);
    await userEvent.click(screen.getByRole("button"));
    expect(onSubmit).toHaveBeenCalledOnce();
  });

  it("calls onQueryChange when user types", async () => {
    const onQueryChange = vi.fn();
    render(<SearchComposer {...defaultProps} onQueryChange={onQueryChange} />);
    await userEvent.type(screen.getByRole("textbox", { name: /question/i }), "hello");
    expect(onQueryChange).toHaveBeenCalled();
  });

  it("shows source selector only in search modes", () => {
    const { rerender } = render(
      <SearchComposer {...defaultProps} mode="search_tool" />,
    );
    expect(screen.getByLabelText(/source/i)).toBeInTheDocument();

    rerender(<SearchComposer {...defaultProps} mode="chat_once" />);
    expect(screen.queryByLabelText(/source/i)).not.toBeInTheDocument();
  });

  it("shows all six mode options", () => {
    render(<SearchComposer {...defaultProps} />);
    const select = screen.getByLabelText(/entry point/i);
    expect(select).toBeInTheDocument();
    const options = select.querySelectorAll("option");
    expect(options).toHaveLength(6);
  });
});
