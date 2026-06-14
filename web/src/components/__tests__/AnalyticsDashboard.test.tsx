import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AnalyticsDashboard } from "../AnalyticsDashboard";
import type { BreakdownAnalytics } from "../../types";

const llmData: BreakdownAnalytics = {
  dimension: "llm",
  items: [
    { label: "gpt-4o", session_count: 42 },
    { label: "unknown", session_count: 8 },
  ],
  total_sessions: 50,
};

const emptyData: BreakdownAnalytics = {
  dimension: "persona",
  items: [],
  total_sessions: 0,
};

describe("AnalyticsDashboard", () => {
  it("always renders the usage breakdown heading", () => {
    render(<AnalyticsDashboard byLLM={null} byPersona={null} byFlow={null} />);
    expect(screen.getByText(/usage breakdown/i)).toBeInTheDocument();
  });

  it("renders LLM breakdown item labels and counts", () => {
    render(<AnalyticsDashboard byLLM={llmData} byPersona={null} byFlow={null} />);
    expect(screen.getByText("gpt-4o")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
  });

  it("shows total_sessions when data is provided", () => {
    render(<AnalyticsDashboard byLLM={llmData} byPersona={null} byFlow={null} />);
    expect(screen.getByText("50 total")).toBeInTheDocument();
  });

  it("shows loading state when data is null", () => {
    render(<AnalyticsDashboard byLLM={null} byPersona={null} byFlow={null} />);
    const loadingMessages = screen.getAllByText(/loading/i);
    expect(loadingMessages.length).toBeGreaterThan(0);
  });

  it("shows empty sessions message when items list is empty", () => {
    render(<AnalyticsDashboard byLLM={emptyData} byPersona={null} byFlow={null} />);
    expect(screen.getByText(/no sessions in this window/i)).toBeInTheDocument();
  });

  it("renders all three panel headings", () => {
    render(<AnalyticsDashboard byLLM={null} byPersona={null} byFlow={null} />);
    expect(screen.getByText("By LLM")).toBeInTheDocument();
    expect(screen.getByText(/by agent/i)).toBeInTheDocument();
    expect(screen.getByText(/by flow/i)).toBeInTheDocument();
  });
});
