import { describe, it, expect, vi, beforeEach } from "vitest";
import { runDebugRetrieval } from "../api";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

beforeEach(() => {
  mockFetch.mockReset();
});

describe("runDebugRetrieval", () => {
  it("returns parsed results on success", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        results: [{ doc_id: "d1", title: "T", score: 0.9 }],
        retrieval_mode: "sparse",
        executed_queries: ["q"],
        latency_ms: 1.2,
      }),
    });

    const outcome = await runDebugRetrieval("sparse", { query: "q", top_k: 5 });

    expect(outcome.ok).toBe(true);
    expect(outcome.status).toBe(200);
    expect(outcome.data?.retrieval_mode).toBe("sparse");
    expect(outcome.data?.results[0].doc_id).toBe("d1");
    expect(mockFetch).toHaveBeenCalledWith(
      "/api/debug/retrieval/sparse",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("does not throw on 503 and surfaces the detail", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 503,
      json: async () => ({ detail: "Dense search not configured" }),
    });

    const outcome = await runDebugRetrieval("dense", { query: "q", top_k: 5 });

    expect(outcome.ok).toBe(false);
    expect(outcome.status).toBe(503);
    expect(outcome.data).toBeNull();
    expect(outcome.detail).toMatch(/dense/i);
  });

  it("reports 404 when the endpoint is not mounted", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 404,
      json: async () => ({ detail: "Not Found" }),
    });

    const outcome = await runDebugRetrieval("hybrid", { query: "q", top_k: 5 });

    expect(outcome.ok).toBe(false);
    expect(outcome.status).toBe(404);
  });
});
