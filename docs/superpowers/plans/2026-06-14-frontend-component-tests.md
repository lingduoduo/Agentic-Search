# Frontend Component Tests — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Vitest + React Testing Library to `web/` and write unit tests for the five core components: `SearchComposer`, `SessionTimeline`, `AnswerPanel`, `SourceGrid`, and `AnalyticsDashboard`.

**Architecture:** Vitest is Vite-native and shares the same config file — zero duplication with the existing `vite.config.ts`. React Testing Library renders components into a `jsdom` environment. Tests live alongside source in `web/src/components/__tests__/`. The existing `npm run typecheck` keeps running; `npm test` is extended to run Vitest after the type check.

**Tech Stack:** Vitest 3, `@testing-library/react` 16, `@testing-library/user-event` 14, `@testing-library/jest-dom` 6, `jsdom`.

---

## File Map

| File | Change |
|------|--------|
| `web/package.json` | Add Vitest + RTL to `devDependencies`; add `test:unit` and update `test` scripts |
| `web/vitest.config.ts` | Create — Vitest config pointing at jsdom environment |
| `web/src/test-setup.ts` | Create — imports `@testing-library/jest-dom` matchers |
| `web/src/components/__tests__/SearchComposer.test.tsx` | Create |
| `web/src/components/__tests__/SessionTimeline.test.tsx` | Create |
| `web/src/components/__tests__/AnswerPanel.test.tsx` | Create |
| `web/src/components/__tests__/SourceGrid.test.tsx` | Create |
| `web/src/components/__tests__/AnalyticsDashboard.test.tsx` | Create |

---

## Task 1: Install Vitest and configure test environment

**Files:**
- Modify: `web/package.json`
- Create: `web/vitest.config.ts`
- Create: `web/src/test-setup.ts`

- [ ] **Step 1: Add dependencies to `web/package.json`**

In `web/package.json`, replace the `devDependencies` block with:

```json
"devDependencies": {
  "@playwright/test": "^1.60.0",
  "@testing-library/jest-dom": "^6.6.3",
  "@testing-library/react": "^16.3.0",
  "@testing-library/user-event": "^14.5.2",
  "@types/react": "^19.0.0",
  "@types/react-dom": "^19.0.0",
  "jsdom": "^26.0.0",
  "vitest": "^3.2.0"
}
```

Also replace the `scripts` block with:

```json
"scripts": {
  "dev": "vite --host 127.0.0.1 --port 5173",
  "build": "tsc -b && vite build",
  "preview": "vite preview --host 127.0.0.1 --port 4173",
  "typecheck": "tsc -b --pretty false",
  "test:unit": "vitest run",
  "test:unit:watch": "vitest",
  "test": "npm run typecheck && npm run test:unit",
  "test:ci": "npm run typecheck && npm run build && npm run test:unit"
}
```

- [ ] **Step 2: Create `web/vitest.config.ts`**

```ts
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test-setup.ts"],
    globals: true,
  },
});
```

- [ ] **Step 3: Create `web/src/test-setup.ts`**

```ts
import "@testing-library/jest-dom";
```

- [ ] **Step 4: Install and verify the test runner starts**

```bash
cd web && npm install && npm run test:unit -- --reporter=verbose 2>&1 | head -20
```

Expected: `No test files found` (no tests yet, but vitest exits 0 with that message).

---

## Task 2: Tests for `SearchComposer`

**Files:**
- Create: `web/src/components/__tests__/SearchComposer.test.tsx`

- [ ] **Step 1: Write the tests**

```tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { SearchComposer } from "../SearchComposer";

const defaultProps = {
  query: "",
  searchUrl: "http://localhost:8001",
  topK: 5,
  mode: "chat_once" as const,
  sourceProvider: "retrieval" as const,
  isLoading: false,
  onQueryChange: vi.fn(),
  onSearchUrlChange: vi.fn(),
  onTopKChange: vi.fn(),
  onModeChange: vi.fn(),
  onSourceProviderChange: vi.fn(),
  onSubmit: vi.fn(),
};

describe("SearchComposer", () => {
  it("renders a textarea and submit button", () => {
    render(<SearchComposer {...defaultProps} />);
    expect(screen.getByRole("textbox", { name: /question/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /search/i })).toBeInTheDocument();
  });

  it("disables submit when query is empty", () => {
    render(<SearchComposer {...defaultProps} query="" />);
    expect(screen.getByRole("button", { name: /search/i })).toBeDisabled();
  });

  it("enables submit when query has content", () => {
    render(<SearchComposer {...defaultProps} query="What is FAISS?" />);
    expect(screen.getByRole("button", { name: /search/i })).not.toBeDisabled();
  });

  it("disables submit while loading", () => {
    render(<SearchComposer {...defaultProps} query="hello" isLoading={true} />);
    expect(screen.getByRole("button")).toBeDisabled();
  });

  it("calls onSubmit when form is submitted", async () => {
    const onSubmit = vi.fn();
    render(<SearchComposer {...defaultProps} query="test" onSubmit={onSubmit} />);
    await userEvent.click(screen.getByRole("button"));
    expect(onSubmit).toHaveBeenCalledOnce();
  });

  it("calls onQueryChange when user types", async () => {
    const onQueryChange = vi.fn();
    render(<SearchComposer {...defaultProps} onQueryChange={onQueryChange} />);
    await userEvent.type(screen.getByRole("textbox"), "hello");
    expect(onQueryChange).toHaveBeenCalled();
  });

  it("shows source selector only in search modes", () => {
    const { rerender } = render(
      <SearchComposer {...defaultProps} mode="search_tool" />
    );
    expect(screen.getByLabelText(/source/i)).toBeInTheDocument();

    rerender(<SearchComposer {...defaultProps} mode="chat_once" />);
    expect(screen.queryByLabelText(/source/i)).not.toBeInTheDocument();
  });

  it("shows all six mode options", () => {
    render(<SearchComposer {...defaultProps} />);
    const select = screen.getByLabelText(/entry point/i);
    expect(select).toBeInTheDocument();
    const options = select.querySelectorAll("option");
    expect(options).toHaveLength(6);
  });
});
```

- [ ] **Step 2: Run**

```bash
cd web && npm run test:unit -- --reporter=verbose 2>&1 | tail -20
```

Expected: 8 tests PASS.

- [ ] **Step 3: Commit**

```bash
cd .. && git add web/package.json web/vitest.config.ts web/src/test-setup.ts \
  web/src/components/__tests__/SearchComposer.test.tsx
git commit -m "test(frontend): add Vitest config and SearchComposer tests"
```

---

## Task 3: Tests for `SessionTimeline`

**Files:**
- Create: `web/src/components/__tests__/SessionTimeline.test.tsx`

- [ ] **Step 1: Write the tests**

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SessionTimeline } from "../SessionTimeline";
import type { ChatMessageView } from "../../types";

const userMsg: ChatMessageView = { role: "user", content: "Hello" };
const assistantMsg: ChatMessageView = { role: "assistant", content: "Hi there" };
const assistantWithRounds: ChatMessageView = {
  role: "assistant",
  content: "Answer",
  metadata: { rounds_used: 3 },
};
const assistantWithTurns: ChatMessageView = {
  role: "assistant",
  content: "Tool answer",
  metadata: { num_turns: 2 },
};

describe("SessionTimeline", () => {
  it("renders empty state when no messages", () => {
    render(<SessionTimeline messages={[]} />);
    expect(screen.getByText(/start a query/i)).toBeInTheDocument();
  });

  it("renders user and assistant messages", () => {
    render(<SessionTimeline messages={[userMsg, assistantMsg]} />);
    expect(screen.getByText("Hello")).toBeInTheDocument();
    expect(screen.getByText("Hi there")).toBeInTheDocument();
  });

  it("shows rounds_used badge when present", () => {
    render(<SessionTimeline messages={[assistantWithRounds]} />);
    expect(screen.getByText(/3 rounds/i)).toBeInTheDocument();
  });

  it("shows num_turns badge when present", () => {
    render(<SessionTimeline messages={[assistantWithTurns]} />);
    expect(screen.getByText(/2 turns/i)).toBeInTheDocument();
  });

  it("does not show badges when metadata is absent", () => {
    render(<SessionTimeline messages={[assistantMsg]} />);
    expect(screen.queryByText(/round/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/turn/i)).not.toBeInTheDocument();
  });

  it("uses singular 'round' for rounds_used=1", () => {
    const msg: ChatMessageView = {
      role: "assistant",
      content: "x",
      metadata: { rounds_used: 1 },
    };
    render(<SessionTimeline messages={[msg]} />);
    expect(screen.getByText("1 round")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run**

```bash
cd web && npm run test:unit -- --reporter=verbose 2>&1 | tail -20
```

Expected: 14 total PASS.

- [ ] **Step 3: Commit**

```bash
cd .. && git add web/src/components/__tests__/SessionTimeline.test.tsx
git commit -m "test(frontend): add SessionTimeline tests including badge rendering"
```

---

## Task 4: Tests for `AnswerPanel` and `SourceGrid`

**Files:**
- Create: `web/src/components/__tests__/AnswerPanel.test.tsx`
- Create: `web/src/components/__tests__/SourceGrid.test.tsx`

- [ ] **Step 1: Read the component files to understand their props**

```bash
cat web/src/components/AnswerPanel.tsx web/src/components/SourceGrid.tsx
```

- [ ] **Step 2: Write `AnswerPanel.test.tsx`**

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AnswerPanel } from "../AnswerPanel";

describe("AnswerPanel", () => {
  it("renders nothing when answer is empty", () => {
    const { container } = render(<AnswerPanel answer="" citations={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders the answer text", () => {
    render(<AnswerPanel answer="FAISS is a vector library." citations={[]} />);
    expect(screen.getByText(/FAISS is a vector library/)).toBeInTheDocument();
  });

  it("renders citation chips", () => {
    render(
      <AnswerPanel answer="See [D1]." citations={["[D1]"]} />
    );
    expect(screen.getByText("[D1]")).toBeInTheDocument();
  });
});
```

- [ ] **Step 3: Write `SourceGrid.test.tsx`**

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SourceGrid } from "../SourceGrid";
import type { SourceDocumentView } from "../../types";

const doc: SourceDocumentView = {
  id: "D1",
  title: "FAISS paper",
  content: "Dense retrieval with FAISS.",
  url: "https://example.test/faiss",
  score: 0.95,
  citation: "[D1]",
  source_provider: "retrieval",
};

describe("SourceGrid", () => {
  it("renders nothing when documents list is empty", () => {
    const { container } = render(<SourceGrid documents={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders document titles", () => {
    render(<SourceGrid documents={[doc]} />);
    expect(screen.getByText("FAISS paper")).toBeInTheDocument();
  });

  it("renders a link when url is present", () => {
    render(<SourceGrid documents={[doc]} />);
    const link = screen.getByRole("link");
    expect(link).toHaveAttribute("href", "https://example.test/faiss");
  });
});
```

- [ ] **Step 4: Run all tests so far**

```bash
cd web && npm run test:unit -- --reporter=verbose 2>&1 | tail -25
```

Expected: ~20 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd .. && git add web/src/components/__tests__/AnswerPanel.test.tsx \
  web/src/components/__tests__/SourceGrid.test.tsx
git commit -m "test(frontend): add AnswerPanel and SourceGrid tests"
```

---

## Task 5: Tests for `AnalyticsDashboard`

**Files:**
- Create: `web/src/components/__tests__/AnalyticsDashboard.test.tsx`

- [ ] **Step 1: Write the tests**

```tsx
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
  it("renders LLM breakdown items", () => {
    render(<AnalyticsDashboard byLLM={llmData} byPersona={null} byFlow={null} />);
    expect(screen.getByText("gpt-4o")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
  });

  it("shows heading when any data is provided", () => {
    render(<AnalyticsDashboard byLLM={llmData} byPersona={null} byFlow={null} />);
    expect(screen.getByText(/usage breakdown/i)).toBeInTheDocument();
  });

  it("renders gracefully when all data is null", () => {
    const { container } = render(
      <AnalyticsDashboard byLLM={null} byPersona={null} byFlow={null} />
    );
    expect(container.firstChild).toBeNull();
  });

  it("renders empty items list without crashing", () => {
    render(<AnalyticsDashboard byLLM={emptyData} byPersona={null} byFlow={null} />);
    expect(screen.getByText(/usage breakdown/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run all tests**

```bash
cd web && npm run test:unit -- --reporter=verbose 2>&1 | tail -30
```

Expected: ~24 tests PASS.

- [ ] **Step 3: Run full frontend type-check + tests**

```bash
cd web && npm test 2>&1 | tail -20
```

Expected: typecheck passes, all unit tests pass.

- [ ] **Step 4: Commit**

```bash
cd .. && git add web/src/components/__tests__/AnalyticsDashboard.test.tsx
git commit -m "test(frontend): add AnalyticsDashboard tests"
```

---

## Task 6: Push and open PR

- [ ] **Step 1: Push**

```bash
git push -u origin feat/frontend-component-tests
```

- [ ] **Step 2: Open PR**

```bash
gh pr create \
  --title "test(frontend): add Vitest + React Testing Library component tests" \
  --body "$(cat <<'EOF'
## Summary

Adds Vitest 3 + React Testing Library to `web/` and covers five core components:

- **SearchComposer** (8 tests): rendering, submit guard, mode selector, source selector visibility, event callbacks
- **SessionTimeline** (6 tests): empty state, message rendering, rounds/turns badges, singular/plural
- **AnswerPanel** (3 tests): empty guard, answer text, citation chips
- **SourceGrid** (3 tests): empty guard, document title, URL link
- **AnalyticsDashboard** (4 tests): breakdown items, heading, null data, empty items

`npm test` now runs `typecheck` then `vitest run` (no browser required).

## Test plan

```bash
cd web && npm test
```

Expected: typecheck passes + ~24 unit tests pass.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
