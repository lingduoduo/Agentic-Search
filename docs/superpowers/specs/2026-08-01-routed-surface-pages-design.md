# Routed surface pages

**Date:** 2026-08-01
**Status:** Approved

## Problem

The four surfaces — Assistant, Search, Chat, Tool Agent — are tabs, not pages. A
`surface` `useState` in `App.tsx` decides which one renders; the URL never
changes. You cannot link to the Chat surface, a refresh always lands on
Assistant, and the browser's back button leaves the app entirely.

`App.tsx` carries the cost. At 466 lines it holds 27 `useState` calls, of which
roughly 23 belong to the Assistant surface alone, while the other three surfaces
are already self-contained components. The file is the Assistant surface plus a
switch statement.

## Goals

- Each surface is a page with its own URL: `/assist`, `/search`, `/chat`, `/tools`.
- Deep links and refresh work in both dev and production.
- Back and forward navigate between pages.
- `App.tsx` becomes a shell; each page owns its own state.

## Non-goals

- No backend endpoint renames. Each page keeps calling the endpoint it calls
  today.
- No changes to the internals of `SearchView`, `ChatView`, or `ToolAgentView`.
- No `/deep-research` page. The current Search surface is retrieval-only — query
  in, source cards out, no synthesis — so the name would promise more than the
  page delivers. The name is left free for a future surface backed by
  `AgenticRAGLoop`.
- No auth or per-page access control.

## Routes

| Path | Page | API (unchanged) |
| --- | --- | --- |
| `/assist` | Assistant | `POST /api/agent`, `POST /api/agent/stream` |
| `/search` | Search | `POST /search/send-search-message` |
| `/chat` | Chat | `POST /chat/send-chat-message` |
| `/tools` | Tools | `POST /tool/send-tool-message`, `GET /tool/tool-history` |

`/` redirects to `/assist` with `replaceState`, so it leaves no history entry.
Unknown paths redirect to `/assist` as well; the navigation is a closed set of
four, so a 404 page would be dead weight.

The bare paths are free. Every backend route under those prefixes is
verb-suffixed — `/chat/send-chat-message`, `/search/search-history`,
`/tool/tool-history` — so no page path collides with an API path.

## Architecture

```
src/router.tsx              NEW   useRoute() + navigate() + <NavLink>
src/App.tsx                 466 → ~110 lines, shell only
src/pages/AssistPage.tsx    NEW   receives everything App.tsx does today
src/pages/SearchPage.tsx    NEW   wraps SearchView
src/pages/ChatPage.tsx      NEW   wraps ChatView
src/pages/ToolsPage.tsx     NEW   wraps ToolAgentView
```

### Router

Hand-rolled, roughly 50 lines. Four flat routes with no params and no nesting
use almost none of `react-router`'s surface area, and the repo's frontend is
deliberately dependency-light.

- `useRoute()` — `useSyncExternalStore` over `popstate`, returning the
  normalized current route.
- `navigate(path)` — `history.pushState` followed by a synthetic `popstate` so
  subscribers re-read.
- `<NavLink>` — renders a real `<a href>`. Cmd-click, middle-click, and "copy
  link address" behave natively; `preventDefault` fires only for a plain
  left-click with no modifier keys.

### State split

`App.tsx`'s state divides along the surface boundary.

**Shell keeps (9):** `showTools`, `showQueryHistory`, `showConsole`,
`adminSummary`, `analyticsByLLM`, `analyticsByPersona`, `analyticsByFlow`.
These are page-independent and self-fetch at mount.

**AssistPage takes (~23):** `query`, `answer`, `citations`, `documents`,
`messages`, `sessionId`, `intent`, `route`, `routeDegraded`, `isLoading`,
`streamingAnswer`, `progressSteps`, `completedSteps`, `toolCalls`,
`controlFlowTrace`, `lastRequestId`, `pendingApprovals`, `error`, `searchUrl`,
`topK`, `sourceProvider`, plus `requestRef`, the `status` memo, `ensureSession`,
`handleSubmit`, `handleNewSession`, and `handleApprovalDecision`.

Navigating away unmounts a page and discards its state. That is page semantics,
and it matches how the three non-Assistant views already behave.

### Topbar

The `status` pill, the `via {route}` pill, and the **New** button are all
derived from Assistant state, so they move into AssistPage's own action row.

The shell topbar keeps the brand, the four nav links, **Tools**, and
**History** — the controls that mean the same thing on every page.

### Dev Console

`DevConsole` moves into AssistPage, and the Console toggle appears only on
`/assist`.

Three of its seven panels — `RequestInspector`, `RequestTracePanel`,
`ServerHealthGrid` — are fed by the last Assistant run through the `answer`,
`citations`, `controlFlowTrace`, and `selectedRequestId` props. The other four
self-fetch from `/api/debug/*`. Today the console renders on all four surfaces,
so on Search, Chat, and Tool those three panels display stale or empty
Assistant data. Scoping the console to the page whose data it shows is the
honest boundary.

## Deep-link plumbing

Real URLs mean a refresh or a pasted link must return the app, not a 404. Two
places need to know about the page paths.

### Vite dev server

`vite.config.ts` proxies `/chat` and `/search` wholesale to FastAPI. A hard
refresh at `http://127.0.0.1:5173/chat` would proxy to the backend, which has no
`GET /chat`, and 404. Both entries get a `bypass` that returns the SPA for
navigation requests while still proxying API calls:

```ts
"/chat": {
  target: "http://127.0.0.1:7860",
  bypass: (req) =>
    req.headers.accept?.includes("text/html") ? "/index.html" : undefined,
},
```

`/assist` and `/tools` are not proxied, so Vite already serves the SPA for them.

### FastAPI

`create_web_app` serves `web/dist/index.html` at `/` only, falling back to the
inline `APP_HTML` when no bundle is built. The same handler is registered for
each of the four page paths — an explicit list rather than a catch-all, so a
genuine API 404 stays a 404 instead of silently returning HTML. The fallback
behavior is identical to `/`: page paths and `/` always serve the same shell.

## Testing

- `router.test.ts` — `navigate` updates the route, back and forward work, `/`
  resolves to `/assist`, an unknown path resolves to `/assist`, and a modified
  click on `<NavLink>` is not intercepted.
- `App.test.tsx` — nav links render; clicking one swaps the page and the URL;
  loading directly at `/chat` mounts ChatPage without passing through Assistant.
- The existing `App.test.tsx` cases are expected to pass unchanged, because
  jsdom starts at `/`, which renders AssistPage. This is verified, not assumed;
  if a test depended on tab-switching it is updated to navigate instead.
- `SearchView`, `ChatView`, and `ToolAgentView` tests are untouched. Those
  components do not change — only what mounts them.
- Backend — `GET /chat` returns the SPA shell, and `POST
  /chat/send-chat-message` still reaches the API.

## Risks

- **The Vite proxy trap.** Dev-only breakage from proxy prefixes has bitten this
  repo twice. The `bypass` on `/chat` and `/search` is the mitigation, and a
  manual refresh on every page is part of verification.
- **Test churn in `App.test.tsx`.** The file is around 400 lines against the
  Assistant surface. Default-routing `/` to AssistPage should keep it green.
- **Moving the Dev Console** is a deliberate behavior change and the one item
  here a reviewer might reasonably want reverted. It is isolated to the shell
  and AssistPage.
