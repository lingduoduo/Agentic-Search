import { describe, expect, it, vi } from "vitest";
import { sendChatMessage, sendSearchMessage } from "../../api";

function sseStream(chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(c) {
      for (const s of chunks) c.enqueue(enc.encode(s));
      c.close();
    },
  });
}

describe("sendChatMessage", () => {
  it("parses answer then done", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true, status: 200,
      body: sseStream([
        'data: {"type":"answer","text":"hi"}\n\n',
        'data: {"type":"done","session_id":"s1"}\n\n',
      ]),
    }));
    const types: string[] = [];
    for await (const e of sendChatMessage({ message: "q" })) types.push(e.type);
    expect(types).toEqual(["answer", "done"]);
  });

  it("throws NO_LOCAL_MODEL on 400", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 400 }));
    await expect(async () => {
      for await (const _ of sendChatMessage({ message: "q" })) void _;
    }).rejects.toThrow("NO_LOCAL_MODEL");
  });
});

describe("sendSearchMessage", () => {
  it("posts and returns docs", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true, status: 200,
      json: async () => ({ all_executed_queries: ["q"], search_docs: [] }),
    }));
    const r = await sendSearchMessage({ search_query: "q" });
    expect(r.all_executed_queries).toEqual(["q"]);
  });
});
