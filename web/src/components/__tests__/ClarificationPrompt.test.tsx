import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ClarificationPrompt } from "../ClarificationPrompt";
import type { ClarificationView } from "../../types";

describe("ClarificationPrompt", () => {
  const clarification: ClarificationView = {
    question: "I can take this a few different ways — which would you like?",
    options: [
      { route: "chat", label: "Explain or summarize it" },
      { route: "search", label: "Find the document or facts" },
      { route: "tool", label: "Take an action on it" },
    ],
  };

  it("asks the question and reports the chosen route", async () => {
    const onSelect = vi.fn();
    render(
      <ClarificationPrompt clarification={clarification} onSelect={onSelect} />,
    );

    expect(screen.getByText(clarification.question)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Find the document or facts" }));

    expect(onSelect).toHaveBeenCalledWith("search");
  });
});
