import { render, screen } from "@testing-library/react";
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

  it("renders citation chips when present", () => {
    render(<AnswerPanel answer="See [D1] for details." citations={["[D1]"]} />);
    expect(screen.getByText("[D1]")).toBeInTheDocument();
  });

  it("does not render citation row when citations are empty", () => {
    render(<AnswerPanel answer="Some answer." citations={[]} />);
    expect(screen.queryByLabelText(/citations/i)).not.toBeInTheDocument();
  });

  it("renders multiple citations", () => {
    render(
      <AnswerPanel answer="See [D1] and [D2]." citations={["[D1]", "[D2]"]} />,
    );
    expect(screen.getByText("[D1]")).toBeInTheDocument();
    expect(screen.getByText("[D2]")).toBeInTheDocument();
  });
});
