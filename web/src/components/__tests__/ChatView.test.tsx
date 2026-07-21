import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ChatView } from "../ChatView";
import * as api from "../../api";

describe("ChatView", () => {
  it("renders the streamed answer", async () => {
    async function* fake() {
      yield { type: "answer", text: "hello there" } as const;
      yield { type: "done", session_id: "s1" } as const;
    }
    vi.spyOn(api, "sendChatMessage").mockImplementation(fake as never);
    render(<ChatView />);
    fireEvent.change(screen.getByLabelText("Chat message"), { target: { value: "hi" } });
    fireEvent.click(screen.getByText("Send"));
    await waitFor(() => expect(screen.getByText("hello there")).toBeInTheDocument());
  });

  it("keeps prior turns visible across two submits", async () => {
    const answers = ["first answer", "second answer"];
    let call = 0;
    vi.spyOn(api, "sendChatMessage").mockImplementation((() => {
      const text = answers[call++];
      async function* g() {
        yield { type: "answer", text } as const;
        yield { type: "done", session_id: "s1" } as const;
      }
      return g();
    }) as never);

    render(<ChatView />);
    const input = screen.getByLabelText("Chat message");
    fireEvent.change(input, { target: { value: "q1" } });
    fireEvent.click(screen.getByText("Send"));
    await waitFor(() => expect(screen.getByText("first answer")).toBeInTheDocument());
    fireEvent.change(input, { target: { value: "q2" } });
    fireEvent.click(screen.getByText("Send"));
    await waitFor(() => expect(screen.getByText("second answer")).toBeInTheDocument());
    expect(screen.getByText("first answer")).toBeInTheDocument();
    expect(screen.getByText("q1")).toBeInTheDocument();
  });

  it("shows the no-model banner on NO_LOCAL_MODEL", async () => {
    vi.spyOn(api, "sendChatMessage").mockImplementation((() => {
      async function* g() { throw new Error("NO_LOCAL_MODEL"); yield undefined as never; }
      return g();
    }) as never);
    render(<ChatView />);
    fireEvent.change(screen.getByLabelText("Chat message"), { target: { value: "hi" } });
    fireEvent.click(screen.getByText("Send"));
    await waitFor(() => expect(screen.getByText(/needs a local model/)).toBeInTheDocument());
  });
});
