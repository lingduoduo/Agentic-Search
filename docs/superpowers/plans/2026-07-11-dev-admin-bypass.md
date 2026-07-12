# Local dev admin-auth bypass — Plan

Spec: [2026-07-11-dev-admin-bypass-design.md](../specs/2026-07-11-dev-admin-bypass-design.md)

## Steps

1. **Config field** — add `dev_admin_bypass: bool = False` to `AuthSettings` and
   load it from `AGENTIC_SEARCH_DEV_ADMIN` in `load_app_settings`
   (`src/internal/configs/app_configs.py`). → verify: env true → field true.

2. **Bypass** — in `make_require_admin` (`src/internal/servers/_auth.py`), return a
   fixed dev admin (`role=admin`) when the flag is on, before the header check.
   → verify: unit test — flag on returns dev admin without headers; off → 401.

3. **Startup warning** — in `create_web_app`, `logger.warning(...)` when the flag
   is on. → verify: manual/log check (no assertion needed).

4. **HTTP test** — `GET /admin/tools` → 200 with flag on/no auth; 401 by default.
   → verify: `pytest tests/unit/servers/test_dev_admin_bypass.py`.

5. **Lint + land** — ruff; rebase `--onto origin/main`; push; PR. Note the PR #401
   proxy dependency in the body.
