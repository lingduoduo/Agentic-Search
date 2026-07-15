# Local dev admin-auth bypass — Design

Date: 2026-07-11
Status: Approved
Branch/PR: feat/dev-admin-bypass

## Problem

The admin dashboard panels (Connectors, Tools, History, Admin overview,
Analytics) are gated by `make_require_admin`, which requires an admin JWT in the
`fastapiusersauth` cookie (or bearer). The bundled UI has no login flow, so
locally the panels 401 unless you hand-mint a token and set the cookie in
devtools. (Separately, the Vite dev proxy must forward `/admin` & `/analytics` —
PR #401 — for the requests to even reach the backend.)

Goal: let the panels load locally without minting a cookie, via an explicit,
default-off, dev-only flag.

## Approach

**Flag.** New `AuthSettings.dev_admin_bypass` (`app_configs.py`), read from env
`AGENTIC_SEARCH_DEV_ADMIN` (default `False`). Independent of `DEV_MODE` (which
only affects OAuth redirects) so enabling it is a single-purpose, deliberate act
— no surprise coupling.

**Bypass.** In `make_require_admin` (`src/internal/servers/_auth.py`), a guard at
the top of the dependency: when `app_settings.auth.dev_admin_bypass` is true,
return a fixed dev admin (`id="dev-admin"`, `metadata={"role": "admin"}`,
`is_anonymous=False`) before any header/JWT check. Because every admin/analytics
router depends on this one factory, the flag unlocks all panels at once, and no
cookie/token is needed. When off, behavior is byte-identical to today.

**Guardrail.** `create_web_app` emits one loud `logger.warning(...)` at startup
when the flag is on ("ADMIN AUTH BYPASSED … dev only, never set in production").
Default off ⇒ zero production impact.

## Non-goals (YAGNI)

- No frontend change, no `/auth/dev-login` endpoint, no cookie minting.
- No host/secret interlock (refuse-if-prod). The explicit default-off flag +
  startup warning is the agreed guardrail.
- Does not change the real JWT/super-user path in any way.

## Dependency (noted, not implemented here)

End-to-end the panels also need the Vite proxy fix (PR #401) so `/admin/*` and
`/analytics/*` reach the backend in dev.

## Success criteria

- `make_require_admin` with the flag on returns the dev admin for a request with
  no auth header; with the flag off, an anonymous request still raises 401.
- `GET /admin/tools` returns 200 with the flag on and no auth; 401 by default.
- Existing admin-auth tests (`test_connector_api.py`, etc.) stay green.
- `ruff` clean.
