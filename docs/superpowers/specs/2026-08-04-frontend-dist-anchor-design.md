# The backend serves the frontend bundle it was built to serve

**Date:** 2026-08-04
**Status:** Approved

## Problem

`_frontend_dist_path` (`src/internal/servers/web/app.py`) locates the built
frontend by counting parents up from its own file:

```python
dist = Path(__file__).resolve().parents[3] / "web" / "dist"
```

From `src/internal/servers/web/app.py` the parents are `web` → `servers` →
`internal` → `src`. `parents[3]` is therefore `src/`, and the function looks for
`src/web/dist` — a path nothing creates. `npm run build` writes
`<repo>/web/dist`, which is `parents[4]`.

The guard then fails, the function returns `None`, and every caller falls back to
the inline `APP_HTML` shell. **The built frontend has never been served.** There
is no error: a miscounted index is indistinguishable from "no build yet", which
is a legitimate state, so the fallback looks like it is working as designed.

Two documents promise otherwise. `.claude/CLAUDE.md`: *"`npm run build` produces
`web/dist`; the FastAPI app serves it automatically when the bundle exists."*
`docs/frontend.md`: port 7860 *"serves the last `npm run build` bundle from
`web/dist`"*.

## Goals

- A built bundle at `<repo>/web/dist` is served by the backend.
- No build still falls back to the inline shell, silently and correctly.

## Non-goals

- No change to the `StaticFiles` mount or the SPA route handler. Both were
  already correct; they were simply never reached with a real bundle.
- No new configuration. The path is derived, and should stay derived — an env var
  would be a workaround for an arithmetic error.

## Design

`parents[3]` → `parents[4]`, with a comment naming each level, since the next
person to move this file has to redo the count and the failure is silent.

## Verification

The bundle is gitignored and absent in CI, so the tests fabricate a checkout —
`pyproject.toml`, `src/internal/servers/web/app.py`, `web/dist/{index.html,assets/}`
— and point the module's `__file__` at it. That exercises the real arithmetic
without depending on anyone having run `npm run build`.

- A bundle at the fabricated root is found. This fails before the change with
  `assert None == …/web/dist` — the bug itself, not a missing helper.
- No bundle returns `None` (the fallback must survive).
- `index.html` without `assets/` returns `None`: a broken build is not servable.

End-to-end against a real build: `/` returns the built `index.html` (matching
hashed asset names), `/assets/<hashed>.js` returns 200, and the SPA routes
`/search` and `/tools` return 200.

## Risks

- Serving the bundle is a behaviour change for anyone who has a stale `web/dist`
  in their checkout: port 7860 will start showing that stale build instead of the
  inline shell. That is the documented contract, and `docs/frontend.md` already
  warns that `:7860` serves the last build and to use `:5173` for live source.
