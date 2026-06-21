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

const mockStreamAgent = api.streamAgent as ReturnType<typeof vi.fn>;

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

function fakeStream(intent: string, documents = baseResponse.documents) {
  async function* gen() {
    yield { type: "answer" as const, text: baseResponse.answer };
    yield { type: "done" as const, session_id: baseResponse.session_id, citations: baseResponse.citations, documents, intent };
  }
  return gen();
}

function errorStream(message: string) {
  async function* gen() {
    yield { type: "error" as const, detail: message };
  }
  return gen();
}

describe("App adaptive layout", () => {
  it("has no intent class on initial render", () => {
    render(<App />);
    const layout = document.querySelector(".results-layout");
    expect(layout?.classList).not.toContain("intent-search");
    expect(layout?.classList).not.toContain("intent-chat");
    expect(layout?.classList).not.toContain("intent-tool");
  });


  it("adds intent-search class to results layout on search response", async () => {
    mockStreamAgent.mockReturnValue(fakeStream("search"));
    render(<App />);
    await submitQuery("find the onboarding doc");
    await waitFor(() => {
      const layout = document.querySelector(".results-layout");
      expect(layout?.classList).toContain("intent-search");
    });
  });

  it("adds intent-chat class to results layout on chat response", async () => {
    mockStreamAgent.mockReturnValue(fakeStream("chat"));
    render(<App />);
    await submitQuery("explain FAISS");
    await waitFor(() => {
      const layout = document.querySelector(".results-layout");
      expect(layout?.classList).toContain("intent-chat");
    });
  });

  it("adds intent-tool class to results layout on tool response", async () => {
    mockStreamAgent.mockReturnValue(fakeStream("tool", []));
    render(<App />);
    await submitQuery("run API tool");
    await waitFor(() => {
      const layout = document.querySelector(".results-layout");
      expect(layout?.classList).toContain("intent-tool");
    });
  });

  it("shows error banner on API failure", async () => {
    mockStreamAgent.mockReturnValue(errorStream("No LLM configured"));
    render(<App />);
    await submitQuery("explain FAISS");
    await waitFor(() => {
      expect(screen.getByText(/no llm configured/i)).toBeInTheDocument();
    });
  });

  it("clears error when new session is created", async () => {
    mockStreamAgent.mockReturnValue(errorStream("failed"));
    render(<App />);
    await submitQuery("q");
    await waitFor(() => expect(screen.getByText(/failed/i)).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: /new/i }));
    await waitFor(() => expect(screen.queryByText(/failed/i)).not.toBeInTheDocument());
  });

  it("has no intent-badge or intent class when response omits intent", async () => {
    async function* noIntentStream() {
      yield { type: "answer" as const, text: baseResponse.answer };
      yield {
        type: "done" as const,
        session_id: baseResponse.session_id,
        citations: baseResponse.citations,
        documents: baseResponse.documents,
        intent: undefined as unknown as string,
      };
    }
    mockStreamAgent.mockReturnValue(noIntentStream());
    render(<App />);
    await submitQuery("explain FAISS");
    await waitFor(() => expect(screen.getByText(baseResponse.answer)).toBeInTheDocument());
    const layout = document.querySelector(".results-layout");
    expect(layout?.className).not.toMatch(/intent-(search|chat|tool)/);
    expect(document.querySelector(".intent-badge")).toBeNull();
  });
});

describe("App example chips", () => {
  it("runs the agent with the chip's query in one click and applies the intent layout", async () => {
    mockStreamAgent.mockReturnValue(fakeStream("search"));
    render(<App />);
    await userEvent.click(
      screen.getByRole("button", { name: /find the onboarding checklist/i }),
    );
    await waitFor(() => {
      expect(mockStreamAgent).toHaveBeenCalledTimes(1);
      const layout = document.querySelector(".results-layout");
      expect(layout?.classList).toContain("intent-search");
    });
    const sentRequest = mockStreamAgent.mock.calls[0][0] as { query: string };
    expect(sentRequest.query).toBe("find the onboarding checklist");
  });
});

describe("App retrieval URL handling", () => {
  it("does not send a client search_url in normal (non-dev) mode", async () => {
    mockStreamAgent.mockReturnValue(fakeStream("search"));
    render(<App />);
    await submitQuery("find docs");
    await waitFor(() => expect(mockStreamAgent).toHaveBeenCalledTimes(1));
    const sentRequest = mockStreamAgent.mock.calls[0][0] as {
      search_url?: string;
      source_provider: string;
    };
    expect(sentRequest.search_url).toBeUndefined();
    expect(sentRequest.source_provider).toBe("retrieval");
  });
});
