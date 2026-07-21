import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SearchView } from "../SearchView";
import * as api from "../../api";

describe("SearchView", () => {
  it("renders returned docs", async () => {
    vi.spyOn(api, "sendSearchMessage").mockResolvedValue({
      all_executed_queries: ["q"],
      search_docs: [
        { title: "FAISS overview", url: null, content: "…", score: 1, metadata: {} },
      ],
    } as never);
    render(<SearchView />);
    fireEvent.change(screen.getByLabelText("Search query"), { target: { value: "faiss" } });
    fireEvent.click(screen.getByText("Search"));
    await waitFor(() =>
      expect(screen.getByRole("heading", { name: /FAISS overview/ })).toBeInTheDocument(),
    );
  });
});
