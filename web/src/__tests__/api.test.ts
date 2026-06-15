import { describe, it, expect, vi, beforeEach } from "vitest";
import { runAgent, streamAgent } from "../api";

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
