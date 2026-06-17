import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { AnswerPanel } from "../AnswerPanel";

describe("AnswerPanel", () => {
  it("renders empty state when answer is empty", () => {
    render(<AnswerPanel answer="" citations={[]} />);
    expect(screen.getByText(/results will appear here/i)).toBeInTheDocument();
  });

  it("renders the answer text", () => {
    render(<AnswerPanel answer="FAISS is a vector library." citations={[]} />);
    expect(screen.getByText(/FAISS is a vector library/)).toBeInTheDocument();
  });

  it("does not render citation row when citations are empty", () => {
    render(<AnswerPanel answer="Some answer." citations={[]} />);
    expect(screen.queryByLabelText(/citations/i)).not.toBeInTheDocument();
  });

  it("renders 'Searched · 5 sources' badge when intent is search", () => {
    render(<AnswerPanel answer="results" citations={[]} intent="search" documentCount={5} />);
    expect(screen.getByText(/searched · 5 sources/i)).toBeInTheDocument();
  });

  it("renders 'Answered · 2 citations' badge when intent is chat", () => {
    render(<AnswerPanel answer="answer" citations={["[D1]", "[D2]"]} intent="chat" />);
    expect(screen.getByText(/answered · 2 citations/i)).toBeInTheDocument();
  });

  it("renders 'Used tools' badge when intent is tool", () => {
    render(<AnswerPanel answer="tool output" citations={[]} intent="tool" />);
    expect(screen.getByText(/used tools/i)).toBeInTheDocument();
  });

  it("renders no badge when intent is undefined", () => {
    render(<AnswerPanel answer="answer" citations={[]} />);
    expect(document.querySelector(".intent-badge")).not.toBeInTheDocument();
  });

  it("renders no badge when answer is empty even if intent is set", () => {
    render(<AnswerPanel answer="" citations={[]} intent="chat" />);
    expect(document.querySelector(".intent-badge")).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// ProgressLog — live phase
// ---------------------------------------------------------------------------

it("renders live progress log when progressSteps is non-empty", () => {
  render(
    <AnswerPanel
      answer=""
      citations={[]}
      progressSteps={[{ turn: 1, text: "search_routing_tool · 5 docs" }]}
      completedSteps={[]}
    />
  );
  expect(screen.getByText(/agent reasoning/i)).toBeInTheDocument();
  expect(screen.getByText(/search_routing_tool · 5 docs/i)).toBeInTheDocument();
});

it("renders collapsed summary when progressSteps is empty and completedSteps is non-empty", () => {
  render(
    <AnswerPanel
      answer="Done."
      citations={[]}
      progressSteps={[]}
      completedSteps={[
        { turn: 1, text: "search_routing_tool · 5 docs" },
        { turn: 2, text: "writing answer..." },
      ]}
    />
  );
  expect(screen.getByText(/2 turns/i)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /show reasoning/i })).toBeInTheDocument();
});

it("expands the log when 'show reasoning' is clicked", async () => {
  render(
    <AnswerPanel
      answer="Done."
      citations={[]}
      progressSteps={[]}
      completedSteps={[{ turn: 1, text: "search_routing_tool · 5 docs" }]}
    />
  );
  const button = screen.getByRole("button", { name: /show reasoning/i });
  await userEvent.click(button);
  expect(screen.getByText(/search_routing_tool · 5 docs/i)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /hide/i })).toBeInTheDocument();
});

it("renders nothing for progress when both arrays are empty", () => {
  render(
    <AnswerPanel
      answer="answer"
      citations={[]}
      progressSteps={[]}
      completedSteps={[]}
    />
  );
  expect(document.querySelector(".progress-log")).not.toBeInTheDocument();
  expect(document.querySelector(".progress-summary")).not.toBeInTheDocument();
});

// ---------------------------------------------------------------------------
// Markdown rendering
// ---------------------------------------------------------------------------

it("renders bold text from markdown", () => {
  render(
    <AnswerPanel
      answer="See **bold** text here."
      citations={[]}
    />
  );
  expect(screen.getByText("bold")).toBeInTheDocument();
  const strong = document.querySelector("strong");
  expect(strong).not.toBeNull();
  expect(strong?.textContent).toBe("bold");
});

it("renders citation [1] as an anchor link", () => {
  render(
    <AnswerPanel
      answer="See [1] for details."
      citations={[]}
    />
  );
  const link = screen.getByRole("link", { name: "[1]" });
  expect(link).toHaveAttribute("href", "#source-[1]");
  expect(link).toHaveClass("citation-link");
});

it("renders no anchor when answer has no citation pattern", () => {
  render(
    <AnswerPanel
      answer="No citations here."
      citations={[]}
    />
  );
  expect(screen.queryByRole("link")).not.toBeInTheDocument();
});

it("does not render .citation-row div", () => {
  render(
    <AnswerPanel
      answer="answer"
      citations={["[1]", "[2]"]}
    />
  );
  expect(document.querySelector(".citation-row")).not.toBeInTheDocument();
});
