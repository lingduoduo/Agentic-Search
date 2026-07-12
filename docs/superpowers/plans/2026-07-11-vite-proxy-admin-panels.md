# Vite Dev-Proxy for Admin Panels — Plan

Spec: [2026-07-11-vite-proxy-admin-panels-design.md](../specs/2026-07-11-vite-proxy-admin-panels-design.md)

## Steps

1. **Add missing proxy prefixes** (`web/vite.config.ts` → `server.proxy`): add
   `/admin`, `/analytics`, `/chat` → `http://127.0.0.1:7860`, with an explanatory
   comment. → verify: `git diff` shows only these additions.

2. **Confirm prefix set is complete** → verify: `grep -oE '"/[a-z]+' web/src/api.ts`
   yields only `/api`, `/admin`, `/analytics`, `/chat`.

3. **End-to-end auth/route check** (done during investigation) → verify: backend
   up, admin JWT minted via `generate_user_jwt_token(..., extra={'role':'admin'})`;
   `/admin/*` + `/analytics/by-llm` return `200` via cookie and bearer, `401`
   without. Proves the routes exist and the proxy is the dev blocker.

4. **Commit, push, PR** → verify: PR opened with the auth recipe in the body.
