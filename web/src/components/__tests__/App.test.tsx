import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { App } from "../../App";

vi.mock("../../api", () => ({
  createSession: vi.fn().mockResolvedValue({ id: "s1", messages: [], title: null, user_id: null }),
  runAgent: vi.fn(),
  streamAgent: vi.fn(),
  getAdminSummary: vi.fn().mockRejectedValue(new Error("no admin")),
  getAnalyticsByLLM: vi.fn().mockRejectedValue(new Error()),
  getAnalyticsByPersona: vi.fn().mockRejectedValue(new Error()),
  getAnalyticsByFlow: vi.fn().mockRejectedValue(new Error()),
}));

import * as api from "../../api";

const mockRunAgent = api.runAgent as ReturnType<typeof vi.fn>;

const baseResponse = {
  session_id: "s1", answer: "The answer", citations: ["[D1]"],
  documents: [{ id: "D1", citation: "[D1]", title: "Doc", content: "c", url: null, score: 0.9, metadata: {} }],
  messages: [{ role: "user", content: "q", metadata: {} }, { role: "assistant", content: "The answer", metadata: {} }],
};

beforeEach(() => { vi.clearAllMocks(); });

async function submitQuery(query = "explain FAISS") {
  const textarea = screen.getByRole("textbox", { name: /question/i });
  await userEvent.clear(textarea);
  await userEvent.type(textarea, query);
  await userEvent.click(screen.getByRole("button", { name: /search/i }));
}

describe("App adaptive layout", () => {
  it("adds intent-search class to results layout on search response", async () => {
    mockRunAgent.mockResolvedValue({ ...baseResponse, intent: "search" });
    render(<App />);
    await submitQuery("find the onboarding doc");
    await waitFor(() => {
      const layout = document.querySelector(".results-layout");
      expect(layout?.classList).toContain("intent-search");
    });
  });

  it("adds intent-chat class to results layout on chat response", async () => {
    mockRunAgent.mockResolvedValue({ ...baseResponse, intent: "chat" });
    render(<App />);
    await submitQuery("explain FAISS");
    await waitFor(() => {
      const layout = document.querySelector(".results-layout");
      expect(layout?.classList).toContain("intent-chat");
    });
  });

  it("adds intent-tool class to results layout on tool response", async () => {
    mockRunAgent.mockResolvedValue({ ...baseResponse, intent: "tool", documents: [] });
    render(<App />);
    await submitQuery("run API tool");
    await waitFor(() => {
      const layout = document.querySelector(".results-layout");
      expect(layout?.classList).toContain("intent-tool");
    });
  });

  it("shows error banner on API failure", async () => {
    mockRunAgent.mockRejectedValue(new Error("No LLM configured"));
    render(<App />);
    await submitQuery("explain FAISS");
    await waitFor(() => {
      expect(screen.getByText(/no llm configured/i)).toBeInTheDocument();
    });
  });

  it("clears error when new session is created", async () => {
    mockRunAgent.mockRejectedValue(new Error("failed"));
    render(<App />);
    await submitQuery("q");
    await waitFor(() => expect(screen.getByText(/failed/i)).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: /new/i }));
    await waitFor(() => expect(screen.queryByText(/failed/i)).not.toBeInTheDocument());
  });
});
