# Dev auth reachability: Vite proxy prefixes + login helper

## Problem

Clicking **Search** in the dashboard returns `Authentication required`.

`POST /search/send-search-message` resolves the caller via `resolve_request_user`
and rejects anonymous requests ([search_backend.py:55-57]) because it builds ACL
filters from the user id. That is correct behavior — the endpoint needs a user.

The problem is that **there is no way to become authenticated from the dev
frontend**:

1. The Vite dev server proxies `/api`, `/health`, `/admin`, `/analytics`,
   `/chat`, `/search`, `/tool` to the backend on `:7860` — but **not `/auth` or
   `/me`**. A `fetch("/auth/login")` from the dashboard origin hits the Vite dev
   server itself and 404s; the request never reaches FastAPI. Verified:

   ```
   POST http://127.0.0.1:5173/auth/login  → 404   (before)
   POST http://127.0.0.1:7860/auth/login  → 200   (direct to backend)
   ```

2. The frontend ships **no login UI** — nothing under `web/src/` calls
   `/auth/login`. So even with the proxy fixed, there is no in-app way to obtain
   the `fastapiusersauth` cookie.

`AGENTIC_SEARCH_DEV_ADMIN=1` does not help: that bypass lives in
`make_require_admin` and covers only admin routers. The search router does not
use it.

This is the same class of gap as the admin-panel proxy fix — an entire backend
prefix the frontend calls was missing from the dev proxy list.

## Scope

In scope — make dev auth reachable:

- Add `/auth` and `/me` to the Vite dev proxy.
- Add a dev-only login helper page so a cookie can be obtained without DevTools.
- Correct the stale proxy prefix list in `docs/frontend.md`.

Out of scope — deliberately not done here:

- A real login UI in the React app (sign-in form, session state, logout).
- A dev bypass for `resolve_request_user` mirroring `make_require_admin`.
  Both are larger design decisions; this change only restores reachability.

## Design

### 1. Proxy prefixes

`web/vite.config.ts` gains two entries alongside the existing ones:

```ts
"/auth": "http://127.0.0.1:7860",
"/me": "http://127.0.0.1:7860",
```

`/auth` covers `POST /auth/register` and `POST /auth/login`; `/me` covers
`GET /me` and `GET /me/permissions`, which the helper uses to confirm the cookie
took effect. Both are backend routers registered by `create_users_router`, so
neither prefix collides with a client-side route or a Vite-internal path
(`/src`, `/node_modules`, `/@vite`).

### 2. Login helper page

`web/dev-login.html` — a static page served by the Vite dev server at
`http://127.0.0.1:5173/dev-login.html`. One button that:

1. `POST /auth/register` with `dev@localhost` / `devpass` (a 400 "already
   registered" is treated as success, so the page is idempotent),
2. `POST /auth/login` form-encoded, which sets the `fastapiusersauth` cookie,
3. `GET /me` to display the resulting identity as confirmation.

Because Vite serves it from the dashboard's own origin, the cookie is scoped
exactly where the Search tab needs it — no CORS, no SameSite edge cases, and no
DevTools required.

**Why this is dev-only and cannot reach production:** `npm run build` emits only
`index.html` plus `assets/` (verified — `dist/` contains no other HTML).
Vite treats extra root-level HTML files as build inputs only when they are named
in `rollupOptions.input`, and this one is not. The page therefore exists solely
under the dev server.

This matters because `POST /auth/register` makes the **first** registered user an
admin ([users/api.py:185]). A page that self-registers a known credential must
never be served from a real deployment; keeping it out of the bundle is what
makes it safe. The page carries a comment saying so.

### 3. Docs

`docs/frontend.md` line 9 lists the proxied prefixes as `/api/*`, `/admin/*`,
`/analytics/*`, `/chat/*` — already stale (missing `/health`, `/search`,
`/tool`). Replace with the full current set and document the helper page in the
dev workflow section.

## Verification

- `POST http://127.0.0.1:5173/auth/login` returns 200 through the proxy.
- With the resulting cookie, `GET /me` returns the user and
  `POST /search/send-search-message` returns search hits — the exact path the
  browser takes.
- `npm run build` output contains no `dev-login.html`.
- `npm run typecheck` and the Vitest suite stay green.

## Alternatives considered

- **Dev bypass on the search endpoint** (env-gated, mirroring
  `AGENTIC_SEARCH_DEV_ADMIN`). Fewer steps for the developer, but it papers over
  the proxy bug — `/auth` would still be unreachable, breaking any future login
  UI — and it adds a second auth bypass to keep correct. Worth doing separately
  if the login step proves annoying.
- **Log in with `curl` and copy the cookie into the browser.** Works, but the
  cookie is httpOnly and Chrome's cookie store is not writable from outside, so
  it means hand-editing via DevTools — the thing this change avoids.
