# Routed Surface Pages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the frontend's tab-button surface switching with four real pages — `/assist`, `/search`, `/chat`, `/tools` — that support deep links, refresh, and browser back/forward.

**Architecture:** A ~70-line hand-rolled router (`useSyncExternalStore` over `popstate`) replaces the `surface` `useState`. `App.tsx` drops from 466 lines to a shell that renders the topbar, the global panels, and whichever page the route selects. All Assistant state moves into a new `AssistPage`; the three other views already stand alone and are mounted directly. Deep links are made to work in Vite dev (a proxy `bypass`) and in FastAPI production (the four page paths serve the SPA shell).

**Tech Stack:** React 19, TypeScript, Vite 7, Vitest 3 + @testing-library/react, FastAPI.

**Spec:** `docs/superpowers/specs/2026-08-01-routed-surface-pages-design.md`

## Global Constraints

- **No new npm dependencies.** The router is hand-rolled. `package.json` must not change.
- **No backend endpoint renames.** Every page keeps calling the endpoint it calls today: `/api/agent` and `/api/agent/stream` (assist), `/search/send-search-message` (search), `/chat/send-chat-message` (chat), `/tool/send-tool-message` and `/tool/tool-history` (tools).
- **The four routes are exactly** `/assist`, `/search`, `/chat`, `/tools`. `/` and any unknown path resolve to `/assist`.
- **No changes to the internals of** `SearchView.tsx`, `ChatView.tsx`, `ToolAgentView.tsx`, or any file under `src/components/debug/`.
- **Frontend gate for every task:** `cd web && npm run typecheck && npm run test:unit` must pass before commit.
- Run `ruff check . --fix && ruff format .` from the repo root before committing any Python change.
- Work on branch `feat/routed-surface-pages`. Never commit to `main`.

## File Structure

| File | Status | Responsibility |
| --- | --- | --- |
| `web/src/router.tsx` | Create | Route type, `normalizeRoute`, `useRoute`, `useCanonicalRoute`, `navigate`, `NavLink` |
| `web/src/__tests__/router.test.tsx` | Create | Router unit tests |
| `web/src/pages/AssistPage.tsx` | Create | The whole Assistant surface: composer, answer, sources, session, approvals, dev console |
| `web/src/App.tsx` | Modify | Shell: topbar, nav, Tools/History panels, admin + analytics, route → page |
| `web/src/components/__tests__/App.test.tsx` | Modify | Existing assist cases stay; new navigation cases added |
| `web/src/styles.css` | Modify | `.surface-switcher` becomes `.surface-nav` link styling |
| `web/vite.config.ts` | Modify | `bypass` on the `/chat` and `/search` proxies |
| `src/internal/servers/web/app.py` | Modify | Serve the SPA shell at the four page paths |
| `tests/unit/servers/web/test_spa_page_routes.py` | Create | Page paths serve the shell; API paths still reach the API |

---

### Task 1: Router module

Self-contained. Nothing imports it yet, so it can be reviewed and merged on its own.

**Files:**
- Create: `web/src/router.tsx`
- Test: `web/src/__tests__/router.test.tsx`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `type Route = "/assist" | "/search" | "/chat" | "/tools"`
  - `const ROUTES: readonly Route[]`
  - `const DEFAULT_ROUTE: Route` (`"/assist"`)
  - `normalizeRoute(pathname: string): Route`
  - `useRoute(): Route`
  - `useCanonicalRoute(): Route` — like `useRoute`, but also rewrites the address bar to the resolved route without adding a history entry
  - `navigate(to: Route): void`
  - `NavLink({ to, className, children }: { to: Route; className?: string; children: ReactNode })`

- [ ] **Step 1: Write the failing test**

Create `web/src/__tests__/router.test.tsx`:

```tsx
import { render, renderHook, screen, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, beforeEach } from "vitest";
import { NavLink, navigate, normalizeRoute, useCanonicalRoute, useRoute } from "../router";

beforeEach(() => {
  window.history.replaceState({}, "", "/");
});

describe("normalizeRoute", () => {
  it("maps / to the default route", () => {
    expect(normalizeRoute("/")).toBe("/assist");
  });

  it("keeps a known route", () => {
    expect(normalizeRoute("/chat")).toBe("/chat");
  });

  it("strips a trailing slash", () => {
    expect(normalizeRoute("/chat/")).toBe("/chat");
  });

  it("falls back to the default for an unknown path", () => {
    expect(normalizeRoute("/nope")).toBe("/assist");
  });
});

describe("useRoute", () => {
  it("reports the route for the current path", () => {
    window.history.replaceState({}, "", "/tools");
    const { result } = renderHook(() => useRoute());
    expect(result.current).toBe("/tools");
  });

  it("updates when navigate() is called", () => {
    const { result } = renderHook(() => useRoute());
    act(() => navigate("/search"));
    expect(result.current).toBe("/search");
    expect(window.location.pathname).toBe("/search");
  });

  // Back/forward reach the app as a popstate event; this is what the browser does.
  it("updates when the browser fires popstate", () => {
    const { result } = renderHook(() => useRoute());
    act(() => {
      window.history.replaceState({}, "", "/chat");
      window.dispatchEvent(new PopStateEvent("popstate"));
    });
    expect(result.current).toBe("/chat");
  });
});

describe("useCanonicalRoute", () => {
  it("rewrites / to the default route without adding a history entry", () => {
    const before = window.history.length;
    const { result } = renderHook(() => useCanonicalRoute());
    expect(result.current).toBe("/assist");
    expect(window.location.pathname).toBe("/assist");
    expect(window.history.length).toBe(before);
  });

  it("leaves a known route alone", () => {
    window.history.replaceState({}, "", "/tools");
    const { result } = renderHook(() => useCanonicalRoute());
    expect(result.current).toBe("/tools");
    expect(window.location.pathname).toBe("/tools");
  });
});

describe("NavLink", () => {
  it("renders a real href so copy-link and new-tab work", () => {
    render(<NavLink to="/chat">Chat</NavLink>);
    expect(screen.getByRole("link", { name: "Chat" })).toHaveAttribute("href", "/chat");
  });

  it("marks the active route with aria-current", () => {
    window.history.replaceState({}, "", "/chat");
    render(
      <>
        <NavLink to="/chat">Chat</NavLink>
        <NavLink to="/tools">Tools</NavLink>
      </>,
    );
    expect(screen.getByRole("link", { name: "Chat" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("link", { name: "Tools" })).not.toHaveAttribute("aria-current");
  });

  it("navigates on a plain left click", async () => {
    render(<NavLink to="/tools">Tools</NavLink>);
    await userEvent.click(screen.getByRole("link", { name: "Tools" }));
    expect(window.location.pathname).toBe("/tools");
  });

  // A modified click means "new tab" — the browser owns it, we must not hijack it.
  it("leaves a cmd-click to the browser", async () => {
    render(<NavLink to="/tools">Tools</NavLink>);
    await userEvent.keyboard("{Meta>}");
    await userEvent.click(screen.getByRole("link", { name: "Tools" }));
    await userEvent.keyboard("{/Meta}");
    expect(window.location.pathname).toBe("/");
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd web && npx vitest run src/__tests__/router.test.tsx`
Expected: FAIL — `Failed to resolve import "../router"`.

- [ ] **Step 3: Write the implementation**

Create `web/src/router.tsx`:

```tsx
import { useCallback, useEffect, useSyncExternalStore } from "react";
import type { MouseEvent, ReactNode } from "react";

export const ROUTES = ["/assist", "/search", "/chat", "/tools"] as const;

export type Route = (typeof ROUTES)[number];

export const DEFAULT_ROUTE: Route = "/assist";

/**
 * Resolve any pathname onto one of the four pages. The navigation is a closed
 * set, so an unknown path is a typo rather than a 404 worth rendering.
 */
export function normalizeRoute(pathname: string): Route {
  const trimmed = pathname.replace(/\/+$/, "");
  return (ROUTES as readonly string[]).includes(trimmed)
    ? (trimmed as Route)
    : DEFAULT_ROUTE;
}

function subscribe(onStoreChange: () => void): () => void {
  window.addEventListener("popstate", onStoreChange);
  return () => window.removeEventListener("popstate", onStoreChange);
}

function getSnapshot(): Route {
  return normalizeRoute(window.location.pathname);
}

/** The route for the current URL, re-read whenever the history entry changes. */
export function useRoute(): Route {
  return useSyncExternalStore(subscribe, getSnapshot, () => DEFAULT_ROUTE);
}

/**
 * `useRoute`, plus an address-bar rewrite when the URL is not already the
 * resolved route (`/` and unknown paths). `replaceState` keeps Back pointing at
 * whatever preceded the app instead of looping on the redirect.
 */
export function useCanonicalRoute(): Route {
  const route = useRoute();
  useEffect(() => {
    if (window.location.pathname !== route) {
      window.history.replaceState({}, "", route);
    }
  }, [route]);
  return route;
}

/** pushState does not notify listeners, so publish the change ourselves. */
export function navigate(to: Route): void {
  if (window.location.pathname === to) return;
  window.history.pushState({}, "", to);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

export function NavLink({
  to,
  className,
  children,
}: {
  to: Route;
  className?: string;
  children: ReactNode;
}) {
  const active = useRoute() === to;

  const handleClick = useCallback(
    (event: MouseEvent<HTMLAnchorElement>) => {
      // Anything but a plain left click means the user asked the browser for
      // something we cannot do: a new tab, a new window, a download.
      if (event.defaultPrevented || event.button !== 0) return;
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      navigate(to);
    },
    [to],
  );

  return (
    <a
      href={to}
      className={`${className ?? ""}${active ? " active" : ""}`.trim()}
      aria-current={active ? "page" : undefined}
      onClick={handleClick}
    >
      {children}
    </a>
  );
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd web && npx vitest run src/__tests__/router.test.tsx`
Expected: PASS — 13 tests.

- [ ] **Step 5: Typecheck**

Run: `cd web && npm run typecheck`
Expected: no output, exit 0.

- [ ] **Step 6: Commit**

```bash
git add web/src/router.tsx web/src/__tests__/router.test.tsx
git commit -m "feat(web): add a minimal history router for the four surfaces"
```

---

### Task 2: Extract AssistPage from App

A pure move. `App.tsx` keeps its tab switcher for now and renders `<AssistPage />` where the assistant JSX used to be, so the existing `App.test.tsx` proves the move changed nothing.

**Files:**
- Create: `web/src/pages/AssistPage.tsx`
- Modify: `web/src/App.tsx`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `AssistPage()` — a zero-prop component exported as `export function AssistPage()`.

- [ ] **Step 1: Confirm the existing tests are green before the move**

Run: `cd web && npm run test:unit`
Expected: PASS. Record the passing test count — it must be identical after this task.

- [ ] **Step 2: Create AssistPage with everything the Assistant surface needs**

Create `web/src/pages/AssistPage.tsx`. Move the following out of `App.tsx` **verbatim**, changing only the import paths from `./` to `../`:

- Module constants `DEFAULT_SEARCH_URL`, `DEFAULT_BROWSER_SEARCH_URL` (App.tsx:40-41)
- `upsertTrace` (App.tsx:43-49)
- `DEV_MODE` (App.tsx:51-56)
- State: `query`, `searchUrl`, `topK`, `intent`, `route`, `routeDegraded`, `sourceProvider`, `sessionId`, `answer`, `citations`, `documents`, `messages`, `isLoading`, `streamingAnswer`, `progressSteps`, `completedSteps`, `error`, `toolCalls`, `controlFlowTrace`, `lastRequestId`, `pendingApprovals` (App.tsx:59-84, skipping `adminSummary`, `analytics*`, `showQueryHistory`, `showTools`)
- `showConsole` and `debugPanels` (App.tsx:87, 90) — the Console toggle and panel move together
- `requestRef` (App.tsx:91)
- `status` memo (App.tsx:100-107)
- `ensureSession` (App.tsx:108-117)
- `handleSubmit` (App.tsx:119-218)
- `handleApprovalDecision` (App.tsx:220-229)
- `handleNewSession` (App.tsx:231-245)
- `handleTopKChange` (App.tsx:247-249)
- `handleSourceProviderChange` (App.tsx:251-265)

Do **not** move: `adminSummary`, `analyticsByLLM`, `analyticsByPersona`, `analyticsByFlow`, their `useEffect` (App.tsx:93-98), `showQueryHistory`, `showTools`.

The file's imports:

```tsx
import { useCallback, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";
import { Bot, Gauge, MessageSquarePlus, Search } from "lucide-react";
import { createSession, streamAgent, submitToolApproval } from "../api";
import { AnswerPanel } from "../components/AnswerPanel";
import { DevConsole } from "../components/debug/DevConsole";
import { SearchComposer } from "../components/SearchComposer";
import { SessionTimeline } from "../components/SessionTimeline";
import { SourceGrid } from "../components/SourceGrid";
import { ToolApprovalCard } from "../components/ToolApprovalCard";
import { ToolCallTracePanel } from "../components/ToolCallTracePanel";
import type {
  AgentExperienceRequest,
  ChatMessageView,
  ControlFlowEventView,
  ProgressStep,
  SearchSourceProvider,
  SourceDocumentView,
  ToolApprovalView,
  ToolCallTraceView,
} from "../types";
```

Its return, which absorbs the status/route pills, the Console toggle, and the New button that used to live in the shell topbar:

```tsx
  return (
    <>
      <div className="page-actions">
        <span className="status-pill">{status}</span>
        {route && (
          <span
            className={`route-pill${routeDegraded ? " route-pill--degraded" : ""}`}
            title={
              routeDegraded
                ? `Routed to ${route} (degraded: ${routeDegraded})`
                : `Routed to ${route}`
            }
          >
            via {route}
            {routeDegraded ? " ⚠" : ""}
          </span>
        )}
        {debugPanels && (
          <button
            className={`icon-button${showConsole ? " active" : ""}`}
            type="button"
            onClick={() => setShowConsole((v) => !v)}
            title="Dev console — backend observability"
          >
            <Gauge size={18} />
            <span>Console</span>
          </button>
        )}
        <button className="icon-button" type="button" onClick={handleNewSession}>
          <MessageSquarePlus size={18} />
          <span>New</span>
        </button>
      </div>

      <SearchComposer
        query={query}
        searchUrl={searchUrl}
        topK={topK}
        sourceProvider={sourceProvider}
        isLoading={isLoading}
        showUrlField={DEV_MODE}
        showSourcePicker={DEV_MODE}
        onQueryChange={setQuery}
        onSearchUrlChange={setSearchUrl}
        onTopKChange={handleTopKChange}
        onSourceProviderChange={handleSourceProviderChange}
        onSubmit={handleSubmit}
        onExampleSelect={(q) => {
          setQuery(q);
          handleSubmit(q);
        }}
      />

      {error && <div className="error-banner">{error}</div>}

      {debugPanels && showConsole && (
        <DevConsole
          answer={streamingAnswer || answer}
          citations={citations}
          controlFlowTrace={controlFlowTrace}
          selectedRequestId={lastRequestId}
        />
      )}

      <div className={`results-layout${intent ? ` intent-${intent}` : ""}`}>
        <section className="answer-column" aria-label="Answer">
          <div className="section-heading">
            <Bot size={18} />
            <h2>Answer</h2>
          </div>
          <AnswerPanel
            answer={streamingAnswer || answer}
            citations={citations}
            intent={intent}
            documentCount={documents.length}
            progressSteps={progressSteps}
            completedSteps={completedSteps}
          />
          {pendingApprovals.map((approval) => (
            <ToolApprovalCard
              key={approval.id}
              approval={approval}
              onDecision={(decision) => handleApprovalDecision(approval.id, decision)}
            />
          ))}
          {intent === "tool" && <ToolCallTracePanel calls={toolCalls} />}
        </section>

        <section className="panel sources-panel wide" aria-label="Sources">
          <div className="section-heading">
            <Search size={18} />
            <h2>Sources</h2>
            <span className="count">{documents.length}</span>
          </div>
          <SourceGrid documents={documents} />
        </section>

        <section className="panel session-panel" aria-label="Session">
          <div className="section-heading">
            <MessageSquarePlus size={18} />
            <h2>Session</h2>
          </div>
          <SessionTimeline messages={messages} />
        </section>
      </div>
    </>
  );
}
```

- [ ] **Step 3: Strip App.tsx down to the shell plus the tab switcher**

In `App.tsx`: delete everything listed in Step 2, delete the now-unused imports, add `import { AssistPage } from "./pages/AssistPage";`, and remove the `status`/`route` pills, the Console button, and the New button from the topbar. Keep the `surface` state and the tab buttons exactly as they are — Task 3 removes them.

Two JSX edits, and they are not symmetric:

1. **Delete outright** the `{surface === "assistant" && (...)}` block at App.tsx:365-388 — the `SearchComposer` and the `{error && ...}` banner. `AssistPage` renders both now.
2. **Delete the `{debugPanels && showConsole && <DevConsole .../>}` block** at App.tsx:390-397. `AssistPage` renders it.
3. **Keep the surface ternary's shape** at App.tsx:403-453 and swap only its first branch. The `results-layout` `<div>` that spans 404-446 becomes `<AssistPage />`:

```tsx
        {surface === "assistant" ? (
          <AssistPage />
        ) : surface === "search" ? (
          <SearchView />
        ) : surface === "chat" ? (
          <ChatView />
        ) : (
          <ToolAgentView />
        )}
```

**Expect one deliberate layout shift.** Today `ToolPanel` and `QueryHistoryPanel` render between the composer and the results. The shell cannot interleave with page content, so they now render above the composer. This is the only visual change in this task; keep them ordered `{showTools && <ToolPanel />}` then `{showQueryHistory && <QueryHistoryPanel />}`, directly above the page, so their order relative to each other and to the results is unchanged.

`App.tsx`'s remaining imports:

```tsx
import { useEffect, useState } from "react";
import { ClipboardList, FileSearch, Wrench } from "lucide-react";
import {
  getAdminSummary,
  getAnalyticsByFlow,
  getAnalyticsByLLM,
  getAnalyticsByPersona,
} from "./api";
import { AdminOverview } from "./components/AdminOverview";
import { AnalyticsDashboard } from "./components/AnalyticsDashboard";
import { ChatView } from "./components/ChatView";
import { QueryHistoryPanel } from "./components/QueryHistoryPanel";
import { SearchView } from "./components/SearchView";
import { ToolAgentView } from "./components/ToolAgentView";
import { ToolPanel } from "./components/ToolPanel";
import { AssistPage } from "./pages/AssistPage";
import type { AdminSurfaceSummary, BreakdownAnalytics } from "./types";
```

- [ ] **Step 4: Add a `.page-actions` rule**

In `web/src/styles.css`, immediately after the `.surface-switcher` block (styles.css:1693):

```css
.page-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
  margin-bottom: 12px;
}
```

- [ ] **Step 5: Run the full frontend suite**

Run: `cd web && npm run typecheck && npm run test:unit`
Expected: PASS, with the same test count recorded in Step 1. No test file was edited in this task — if any test fails, the move dropped something; find it rather than editing the test.

- [ ] **Step 6: Commit**

```bash
git add web/src/pages/AssistPage.tsx web/src/App.tsx web/src/styles.css
git commit -m "refactor(web): extract the assistant surface into AssistPage"
```

---

### Task 3: Route the shell

Replaces the `surface` state and the tab buttons with the router.

**Files:**
- Modify: `web/src/App.tsx`
- Modify: `web/src/components/__tests__/App.test.tsx`
- Modify: `web/src/styles.css`

**Interfaces:**
- Consumes: `useCanonicalRoute`, `NavLink`, `Route` from `./router` (Task 1); `AssistPage` from `./pages/AssistPage` (Task 2).
- Produces: nothing new for later tasks.

- [ ] **Step 1: Write the failing navigation tests**

Append to `web/src/components/__tests__/App.test.tsx`:

```tsx
describe("page navigation", () => {
  beforeEach(() => {
    window.history.replaceState({}, "", "/");
  });

  it("renders the assistant page at /", () => {
    render(<App />);
    expect(screen.getByRole("textbox", { name: /question/i })).toBeInTheDocument();
    expect(window.location.pathname).toBe("/assist");
  });

  it("exposes the four surfaces as links, not tabs", () => {
    render(<App />);
    const nav = screen.getByRole("navigation", { name: /surfaces/i });
    expect(nav).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /assistant/i })).toHaveAttribute("href", "/assist");
    expect(screen.getByRole("link", { name: /^search$/i })).toHaveAttribute("href", "/search");
    expect(screen.getByRole("link", { name: /^chat$/i })).toHaveAttribute("href", "/chat");
    expect(screen.getByRole("link", { name: /^tools$/i })).toHaveAttribute("href", "/tools");
  });

  it("swaps the page and the URL when a link is clicked", async () => {
    render(<App />);
    await userEvent.click(screen.getByRole("link", { name: /^chat$/i }));
    expect(window.location.pathname).toBe("/chat");
    expect(screen.queryByRole("textbox", { name: /question/i })).not.toBeInTheDocument();
  });

  it("mounts the requested page on a deep link without passing through assist", () => {
    window.history.replaceState({}, "", "/search");
    render(<App />);
    expect(screen.getByRole("textbox", { name: /search query/i })).toBeInTheDocument();
    expect(screen.queryByRole("textbox", { name: /question/i })).not.toBeInTheDocument();
  });

  it("goes back to the previous page on popstate", async () => {
    render(<App />);
    await userEvent.click(screen.getByRole("link", { name: /^tools$/i }));
    expect(window.location.pathname).toBe("/tools");
    act(() => {
      window.history.replaceState({}, "", "/assist");
      window.dispatchEvent(new PopStateEvent("popstate"));
    });
    expect(screen.getByRole("textbox", { name: /question/i })).toBeInTheDocument();
  });
});
```

Add `act` to the existing `@testing-library/react` import at the top of the file.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd web && npx vitest run src/components/__tests__/App.test.tsx -t "page navigation"`
Expected: FAIL — no element with role `navigation`, and `window.location.pathname` is `/` rather than `/assist`.

- [ ] **Step 3: Replace the tab switcher with the router**

In `web/src/App.tsx`:

Add the import:

```tsx
import { NavLink, useCanonicalRoute } from "./router";
```

Delete the `surface` state (`const [surface, setSurface] = useState<...>("assistant");`) and replace it with:

```tsx
  const route = useCanonicalRoute();
```

Replace the `.surface-switcher` block with:

```tsx
            <nav className="surface-nav" aria-label="Surfaces">
              <NavLink to="/assist" className="icon-button">Assistant</NavLink>
              <NavLink to="/search" className="icon-button">Search</NavLink>
              <NavLink to="/chat" className="icon-button">Chat</NavLink>
              <NavLink to="/tools" className="icon-button">Tools</NavLink>
            </nav>
```

Replace the `{surface === "assistant" && <AssistPage />}` line with:

```tsx
        {route === "/assist" ? (
          <AssistPage />
        ) : route === "/search" ? (
          <SearchView />
        ) : route === "/chat" ? (
          <ChatView />
        ) : (
          <ToolAgentView />
        )}
```

- [ ] **Step 4: Rename the switcher style**

In `web/src/styles.css`, rename the `.surface-switcher` selector (styles.css:1693) to `.surface-nav` and add link resets, since the nav items are now anchors wearing `.icon-button`:

```css
.surface-nav {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.surface-nav .icon-button {
  text-decoration: none;
}
```

- [ ] **Step 5: Run the full frontend suite**

Run: `cd web && npm run typecheck && npm run test:unit`
Expected: PASS. The pre-existing assist tests still pass because jsdom starts at `/`, which canonicalises to `/assist` and mounts `AssistPage`.

- [ ] **Step 6: Commit**

```bash
git add web/src/App.tsx web/src/components/__tests__/App.test.tsx web/src/styles.css
git commit -m "feat(web): route the four surfaces as pages instead of tabs"
```

---

### Task 4: Deep links in dev and production

Without this, the URLs work only while the SPA is already loaded. A refresh or a pasted link 404s.

**Files:**
- Modify: `web/vite.config.ts`
- Modify: `src/internal/servers/web/app.py:1300-1304`
- Create: `tests/unit/servers/web/test_spa_page_routes.py`

**Interfaces:**
- Consumes: the four route paths from Task 1.
- Produces: nothing.

- [ ] **Step 1: Write the failing backend test**

Create `tests/unit/servers/web/test_spa_page_routes.py`:

```python
"""Page paths must serve the SPA shell, and only the page paths.

The frontend routes /assist, /search, /chat and /tools client-side. A refresh
or a pasted link hits the backend directly, so each path has to return the app
shell rather than a 404 — without shadowing the API routes that share a prefix.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.internal.db import AgenticSearchStore
from src.internal.servers.web.app import create_web_app

PAGE_PATHS = ["/assist", "/search", "/chat", "/tools"]


@pytest.fixture
def client() -> TestClient:
    # No `with`: the lifespan would load SEARCH_AGENT_MODEL, which this does not need.
    return TestClient(create_web_app(store=AgenticSearchStore(":memory:")))


@pytest.mark.parametrize("path", PAGE_PATHS)
def test_page_path_serves_the_app_shell(client: TestClient, path: str) -> None:
    response = client.get(path)

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert response.text == client.get("/").text


def test_api_paths_under_a_page_prefix_still_reach_the_api(client: TestClient) -> None:
    """The page route must not shadow /chat/* — it is a sibling, not a parent."""
    response = client.post("/chat/create-chat-session", json={})

    assert response.status_code == 200
    assert "chat_session_id" in response.json()


def test_an_unknown_path_is_still_a_404(client: TestClient) -> None:
    """Explicit page routes, not a catch-all: a typo must not return HTML."""
    response = client.get("/definitely-not-a-page")

    assert response.status_code == 404
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /Users/linghuang/Git/Agentic-Search && python -m pytest tests/unit/servers/web/test_spa_page_routes.py -q`
Expected: FAIL — the four `test_page_path_serves_the_app_shell` cases return 404.

- [ ] **Step 3: Serve the shell at the page paths**

In `src/internal/servers/web/app.py`, replace the `index` handler (app.py:1300-1304) with:

```python
    def _app_shell() -> str:
        if frontend_dist:
            return (frontend_dist / "index.html").read_text(encoding="utf-8")
        return APP_HTML

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _app_shell()

    # The frontend routes these client-side. They are listed explicitly rather
    # than served from a catch-all so a mistyped API path stays a 404 instead of
    # silently returning HTML. Keep in sync with ROUTES in web/src/router.tsx.
    @app.get("/assist", response_class=HTMLResponse)
    @app.get("/search", response_class=HTMLResponse)
    @app.get("/chat", response_class=HTMLResponse)
    @app.get("/tools", response_class=HTMLResponse)
    def spa_page() -> str:
        return _app_shell()
```

- [ ] **Step 4: Run the backend test to verify it passes**

Run: `python -m pytest tests/unit/servers/web/test_spa_page_routes.py -q`
Expected: PASS — 6 tests.

- [ ] **Step 5: Stop the Vite proxy from swallowing page navigations**

`vite.config.ts` proxies `/chat` and `/search` wholesale, so a refresh at `http://127.0.0.1:5173/chat` is forwarded to FastAPI instead of served by Vite. Replace those two string entries with objects that let HTML navigations fall through to the SPA:

```ts
      "/chat": {
        target: "http://127.0.0.1:7860",
        // A page navigation asks for HTML; that is the SPA route, not the API.
        // API calls from the app send Accept: application/json and still proxy.
        bypass: (req) =>
          req.headers.accept?.includes("text/html") ? "/index.html" : undefined,
      },
      "/search": {
        target: "http://127.0.0.1:7860",
        bypass: (req) =>
          req.headers.accept?.includes("text/html") ? "/index.html" : undefined,
      },
```

Leave `/api`, `/health`, `/auth`, `/me`, `/admin`, `/analytics`, and `/tool` as they are. `/assist` and `/tools` are not proxied at all, so Vite already serves the SPA for them.

- [ ] **Step 6: Verify the dev server by hand**

Run the stack:

```bash
python3 -m src.internal.servers.retrieval.demo --corpus_path data/corpus.jsonl   # terminal 1
PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860  # terminal 2
cd web && npm run dev                                                            # terminal 3
```

Check each of these by hand, since no automated test covers the dev proxy:
- Load `http://127.0.0.1:5173/chat` directly — the Chat page renders, no 404.
- Load `http://127.0.0.1:5173/search` directly — the Search page renders.
- From `/chat`, send a message — the request reaches `/chat/send-chat-message` and gets a reply, proving the bypass did not break the API proxy.
- Click through all four nav links, then press Back twice — the pages step backwards.

- [ ] **Step 7: Verify the production bundle**

```bash
cd web && npm run build
# restart uvicorn so it picks up web/dist, then:
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:7860/chat    # 200
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:7860/nope    # 404
```

- [ ] **Step 8: Lint and commit**

```bash
ruff check . --fix && ruff format .
git add web/vite.config.ts src/internal/servers/web/app.py tests/unit/servers/web/test_spa_page_routes.py
git commit -m "feat(web): serve the SPA at the page paths in dev and production"
```

---

### Task 5: Documentation

Six sentences across two files call the surfaces "tabs". Each is now wrong.

**Files:**
- Modify: `README.md` (lines 12, 72, 76, 80, 123)
- Modify: `docs/frontend.md` (lines 9, 23, 25, 38, 94)

- [ ] **Step 1: Update README.md**

Make exactly these replacements, leaving the rest of each line intact:

| Line | Was | Becomes |
| --- | --- | --- |
| 12 | `direct Search, Chat, and Tool Agent tabs` | `direct Search, Chat, and Tool pages` |
| 72 | `(the **Search** tab)` | `(the **Search** page, `/search`)` |
| 76 | `(the **Chat** tab)` | `(the **Chat** page, `/chat`)` |
| 80 | `(the **Tool Agent** tab)` | `(the **Tools** page, `/tools`)` |
| 123 | `The header switches between the **Assistant**, **Search**, **Chat**, and **Tool Agent** surfaces.` | `The header links to the four pages: **Assistant** (`/assist`), **Search** (`/search`), **Chat** (`/chat`), and **Tools** (`/tools`). Each has its own URL, so pages can be linked to and survive a refresh.` |

- [ ] **Step 2: Update docs/frontend.md**

| Line | Change |
| --- | --- |
| 9 | After the proxy-prefix sentence, add: `The `/chat` and `/search` proxies carry a `bypass` so a page navigation (`Accept: text/html`) is served the SPA instead of being forwarded to FastAPI — without it, a refresh on those two pages 404s in dev only.` |
| 23 | Heading `## Logging in (Search / Chat / Tool tabs)` → `## Logging in (Search / Chat / Tools pages)` |
| 25 | `The **Search** tab calls` → `The **Search** page calls` |
| 38 | `which is where the tabs need it` → `which is where the pages need it` |
| 94 | `The **New** button (`handleNewSession`)` → `The **New** button (`handleNewSession`, on the Assistant page)` |

- [ ] **Step 3: Add a routing note to docs/frontend.md**

After the paragraph at line 9, add:

```markdown
### Pages

The four surfaces are routes, not tabs. `web/src/router.tsx` holds the route
list and a ~70-line history router; `App.tsx` is a shell that renders the
topbar, the global panels, and whichever page the route selects. Each page owns
its state, so navigating away discards it.

`ROUTES` in `web/src/router.tsx` is mirrored in two other places, and all three
must be changed together to add a page: the `bypass` entries in
`web/vite.config.ts` (only for prefixes the backend also serves) and the
`spa_page` handler in `src/internal/servers/web/app.py`.
```

- [ ] **Step 4: Commit**

```bash
git add README.md docs/frontend.md
git commit -m "docs: describe the four surfaces as routed pages"
```

---

## Verification

Before opening the PR:

```bash
cd web && npm run typecheck && npm run test:unit && npm run build
cd /Users/linghuang/Git/Agentic-Search && python -m pytest tests/unit -q
ruff check . && ruff format --check .
```

`tests/unit/test_mcp_document_tools.py::test_extract_pdf_uses_the_maintained_pypdf_package` fails on `main` already and is unrelated to this work. Confirm it is the only failure; anything else is yours.

## Self-Review Notes

- Every spec section maps to a task: routes and router → Task 1; state split, topbar, and Dev Console → Task 2; navigation → Task 3; deep-link plumbing → Task 4; the spec's testing section is distributed across Tasks 1, 3, and 4.
- Names are consistent across tasks: `normalizeRoute`, `useRoute`, `useCanonicalRoute`, `navigate`, `NavLink`, `Route`, `ROUTES`, `DEFAULT_ROUTE`, `AssistPage`.
- Task 2 deliberately changes no test file, which is what makes it a verifiable pure move.
