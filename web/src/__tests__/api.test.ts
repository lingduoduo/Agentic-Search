import { describe, it, expect, vi, beforeEach } from "vitest";
import { runAgent, streamAgent, submitToolApproval } from "../api";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

beforeEach(() => { mockFetch.mockReset(); });

describe("runAgent", () => {
  it("passes intent field through from response", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        session_id: "s1", answer: "hello", citations: [],
        documents: [], messages: [], intent: "search",
      }),
    });
    const result = await runAgent({ query: "find docs" });
    expect(result.intent).toBe("search");
  });
});

describe("submitToolApproval", () => {
  it("posts the approval decision to the pending approval", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: "accepted" }),
    });

    await submitToolApproval("a1", "approve");

    expect(mockFetch).toHaveBeenCalledWith("/api/agent/approvals/a1", {
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      method: "POST",
      body: JSON.stringify({ decision: "approve" }),
      signal: undefined,
    });
  });
});

describe("streamAgent", () => {
  it("yields error event when response is not ok", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 502,
      body: null,
    });
    const gen = streamAgent({ query: "q" });
    await expect(gen.next()).rejects.toThrow("502");
  });
});
