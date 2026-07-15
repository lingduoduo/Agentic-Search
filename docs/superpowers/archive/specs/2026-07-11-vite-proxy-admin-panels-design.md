# Vite Dev-Proxy for Admin Panels — Design

Date: 2026-07-11
Status: Approved
Branch/PR: fix/vite-proxy-admin-panels

## Problem

In the dev server (`npm run dev`, port 5173), the admin panels — **Connectors,
Tools, History, Admin overview, Analytics** — silently do nothing. Root cause
(verified end-to-end):

`web/vite.config.ts` proxies only `/api` and `/health` to the FastAPI backend
(:7860). But the admin API client (`web/src/api.ts`) calls three **unprefixed**
router groups the backend mounts:

- `/admin/*` — `/admin/connectors`, `/admin/tools`, `/admin/chat-session-history`,
  `/admin/observability/summary`, `/admin/query-history/audit`
- `/analytics/*` — `/analytics/by-llm|by-persona|by-flow`
- `/chat/*` — `/chat/create-chat-message-feedback`

In dev these requests hit the Vite server (which has no such routes) instead of
the backend, and each panel's `.catch(() => undefined)` swallows the failure — so
the buttons appear inert. `/api/*` features (search, sessions, Dev Console,
feedback) work because they *are* proxied. In production the bug is invisible:
FastAPI serves the built bundle on its own origin, so there is no proxy.

Confirmed the only prefixes `api.ts` uses are `/api` (proxied), `/admin`,
`/analytics`, `/chat`.

## Second, separate blocker (not fixed here): admin auth

Even reached, `/admin/*` and `/analytics/*` are gated by `make_require_admin`
(`src/internal/servers/_auth.py`) → `401` without an admin identity. The bundled
UI sends `credentials: "same-origin"` but no token and has no login flow, so the
panels 401 even once proxied. This is existing admin/enterprise gating, out of
scope for a dev-proxy fix. The usage recipe (mint a dev admin JWT, present it as
the `fastapiusersauth` cookie) is documented in the PR body, not code.

## Approach

Add `/admin`, `/analytics`, `/chat` to the Vite `server.proxy` map, each →
`http://127.0.0.1:7860`. One-file change (`web/vite.config.ts`) with a comment
explaining why. No frontend/route collision: the SPA has no client-side routes at
those prefixes (they are fetch-only).

## Success criteria

- Verified: with the backend up and an admin token, `/admin/tools`,
  `/admin/connectors`, `/admin/observability/summary`, `/analytics/by-llm` return
  `200` (cookie and bearer); no-auth returns `401` (proving the routes exist and
  the remaining gate is auth, not the proxy).
- Proxy config lists exactly the four prefixes `api.ts` calls.

## Non-goals

- No auth bypass / dev-login (separate, larger design).
- No change to the backend routers or `api.ts`.
