import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { App } from "../../App";

vi.mock("../../api", () => ({
  createSession: vi.fn().mockResolvedValue({ id: "s1", messages: [], title: null, user_id: null }),
  runAgent: vi.fn(),
  streamAgent: vi.fn(),
  submitToolApproval: vi.fn(),
  getAdminSummary: vi.fn().mockRejectedValue(new Error("no admin")),
  getAnalyticsByLLM: vi.fn().mockRejectedValue(new Error()),
  getAnalyticsByPersona: vi.fn().mockRejectedValue(new Error()),
  getAnalyticsByFlow: vi.fn().mockRejectedValue(new Error()),
}));

import * as api from "../../api";

const mockStreamAgent = api.streamAgent as ReturnType<typeof vi.fn>;
const mockSubmitToolApproval = api.submitToolApproval as ReturnType<typeof vi.fn>;

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

  it("shows the authoritative control-flow trace after streaming completes", async () => {
    const trace = [{
      sequence: 1,
      timestamp: "2026-06-27T12:00:00.000Z",
      turn: 1,
      component: "planner",
      action: "search_planned",
      status: "decided",
      duration_ms: null,
      details: { decision: "search" },
    }];
    async function* tracedStream() {
      yield { type: "trace" as const, event: trace[0] };
      yield { type: "answer" as const, text: baseResponse.answer };
      yield {
        type: "done" as const,
        session_id: baseResponse.session_id,
        citations: baseResponse.citations,
        documents: baseResponse.documents,
        intent: "search" as const,
        control_flow_trace: trace,
      };
    }
    mockStreamAgent.mockReturnValue(tracedStream());

    render(<App />);
    await submitQuery("trace this search");

    const summary = await screen.findByRole("button", { name: /show control flow/i });
    expect(summary).toHaveTextContent("1 control-flow events");
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

describe("App grounding status", () => {
  it("shows 'Grounded' when the answer has citations", async () => {
    mockStreamAgent.mockReturnValue(fakeStream("chat"));
    render(<App />);
    await submitQuery("explain FAISS");
    await waitFor(() =>
      expect(screen.getByText(baseResponse.answer)).toBeInTheDocument(),
    );
    expect(document.querySelector(".status-pill")?.textContent).toBe("Grounded");
  });

  it("shows 'Answered' (not 'Grounded') when the answer has no citations", async () => {
    async function* noCitationsStream() {
      yield { type: "answer" as const, text: baseResponse.answer };
      yield {
        type: "done" as const,
        session_id: baseResponse.session_id,
        citations: [] as string[],
        documents: [],
        intent: "chat",
      };
    }
    mockStreamAgent.mockReturnValue(noCitationsStream());
    render(<App />);
    await submitQuery("FAISS");
    await waitFor(() =>
      expect(screen.getByText(baseResponse.answer)).toBeInTheDocument(),
    );
    expect(document.querySelector(".status-pill")?.textContent).toBe("Answered");
  });

  it("shows the chosen route chip, with a degraded marker", async () => {
    async function* routedStream() {
      yield { type: "answer" as const, text: baseResponse.answer };
      yield {
        type: "done" as const,
        session_id: baseResponse.session_id,
        citations: [] as string[],
        documents: [],
        intent: "search",
        route: "search_agent",
        route_degraded: "no_local_model",
      };
    }
    mockStreamAgent.mockReturnValue(routedStream());
    render(<App />);
    await submitQuery("FAISS");
    await waitFor(() =>
      expect(screen.getByText(baseResponse.answer)).toBeInTheDocument(),
    );
    const pill = document.querySelector(".route-pill");
    expect(pill?.textContent).toContain("search_agent");
    expect(pill?.getAttribute("title")).toContain("degraded: no_local_model");
    expect(pill?.classList.contains("route-pill--degraded")).toBe(true);
  });

  it("renders no route chip when the response omits route", async () => {
    mockStreamAgent.mockReturnValue(fakeStream("chat"));
    render(<App />);
    await submitQuery("explain FAISS");
    await waitFor(() =>
      expect(screen.getByText(baseResponse.answer)).toBeInTheDocument(),
    );
    expect(document.querySelector(".route-pill")).toBeNull();
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
      source_provider?: string;
    };
    expect(sentRequest.search_url).toBeUndefined();
    expect(sentRequest.source_provider).toBeUndefined();
  });
});

describe("App tool approvals", () => {
  const approval = {
    id: "approval-1",
    tool_name: "send_email",
    arguments: { recipient: "ops@example.com" },
    expires_at: "2099-01-01T00:00:00.000Z",
  };

  it.each(["approve", "deny"] as const)(
    "submits an %s decision without restarting the stream",
    async (decision) => {
      let resumeStream!: () => void;
      const decisionSubmitted = new Promise<void>((resolve) => { resumeStream = resolve; });
      async function* approvalStream() {
        yield { type: "approval_required" as const, approval };
        await decisionSubmitted;
        yield { type: "answer" as const, text: baseResponse.answer };
        yield {
          type: "done" as const,
          session_id: baseResponse.session_id,
          citations: baseResponse.citations,
          documents: baseResponse.documents,
          intent: "tool" as const,
        };
      }
      mockStreamAgent.mockReturnValue(approvalStream());
      mockSubmitToolApproval.mockImplementation(async () => { resumeStream(); });

      render(<App />);
      await submitQuery("send an email");
      await userEvent.click(await screen.findByRole("button", {
        name: decision === "approve" ? /approve/i : /deny/i,
      }));

      await waitFor(() => expect(screen.getByText(baseResponse.answer)).toBeInTheDocument());
      expect(mockSubmitToolApproval).toHaveBeenCalledTimes(1);
      expect(mockSubmitToolApproval).toHaveBeenCalledWith(
        approval.id,
        decision,
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      );
      expect(mockStreamAgent).toHaveBeenCalledTimes(1);
      expect(screen.queryByLabelText(/approval required for send_email/i)).not.toBeInTheDocument();
    },
  );

  it("keeps the approval available and shows an error when submission fails", async () => {
    async function* approvalStream() {
      yield { type: "approval_required" as const, approval };
      await new Promise(() => undefined);
    }
    mockStreamAgent.mockReturnValue(approvalStream());
    mockSubmitToolApproval.mockRejectedValue(new Error("Approval service unavailable"));

    render(<App />);
    await submitQuery("send an email");
    await userEvent.click(await screen.findByRole("button", { name: /approve/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Approval service unavailable");
    expect(screen.getByLabelText(/approval required for send_email/i)).toBeInTheDocument();
    expect(mockStreamAgent).toHaveBeenCalledTimes(1);
  });
});
