import { describe, expect, it, vi } from "vitest";
import { sendToolMessage } from "../../api";

function sseStream(chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(enc.encode(c));
      controller.close();
    },
  });
}

describe("sendToolMessage", () => {
  it("parses SSE events in order", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        body: sseStream([
          'data: {"type":"progress","turn":1,"text":"search · 3 docs"}\n\n',
          'data: {"type":"answer","text":"hi"}\n\n',
          'data: {"type":"done","session_id":"s1","tool_calls":[],"num_turns":1}\n\n',
        ]),
      }),
    );
    const types: string[] = [];
    for await (const e of sendToolMessage({ message: "q" })) types.push(e.type);
    expect(types).toEqual(["progress", "answer", "done"]);
  });

  it("throws NO_LOCAL_MODEL on 400", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 400 }));
    await expect(async () => {
      for await (const _ of sendToolMessage({ message: "q" })) void _;
    }).rejects.toThrow("NO_LOCAL_MODEL");
  });
});
