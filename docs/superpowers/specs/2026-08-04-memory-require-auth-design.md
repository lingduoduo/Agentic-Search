# An opt-in switch for the shared anonymous memory bucket

**Date:** 2026-08-04
**Status:** Approved

## Problem

`/api/memory/*` is mounted unconditionally and resolves its caller with:

```python
user = user_from_headers(request.headers)
return user.id if user is not None else default_user_id   # "default_user"
```

Every anonymous caller therefore reads and writes **the same** memory bucket.
The MCP memory tools do the same: `_resolve_user_id()` reads the bearer token's
`sub` and falls back to `DEFAULT_MEMORY_USER_ID`.

This is deliberate, not an oversight. It was reviewed on 2026-07-22 and accepted
for research use, and `docs/cli.md` documents it as a feature: *"Unauthenticated
is fine for local/research use — with no token the backend treats the caller as
the default user."* The single-operator local flow depends on it.

It is wrong for any deployment with more than one person, where "anonymous" is
not one person but everyone, and one user's remembered facts are readable and
overwritable by the next.

## Goals

- A deployment can require authentication for memory.
- The documented unauthenticated local flow keeps working untouched by default.
- The switch governs every door into the store, not just the HTTP one.

## Non-goals

- No change to the default. Off is today's behaviour, exactly.
- No per-anonymous-caller bucketing. There is no stable anonymous identity to key
  on; a session id is not one, and inventing one would be a new identity model.
- No change to `curate`'s session-ownership check (#494), which is orthogonal and
  applies whether or not this flag is set.

## Design

`create_memory_router(db, llm, *, require_auth=False)`. When set, `_uid` raises
`401 Authentication required.` instead of returning `default_user_id`. The
authenticated path is untouched.

Plumbed as `ServiceSettings.memory_require_auth` from
`AGENTIC_SEARCH_MEMORY_REQUIRE_AUTH`, alongside `debug_panels`, and passed to
`_register_routers`.

**The MCP tools honour the same variable**, read via `os.getenv` at call time
rather than through `ServiceSettings`, because the MCP server is a separate
process that never constructs one. Enforcing only in the router would leave the
flag half-honoured: an operator who set it would still have an unauthenticated
MCP client writing into the shared bucket. A half-honoured security switch is
worse than none, because it is believed.

`_resolve_user_id` raises `PermissionError`, which the tools' existing
`except Exception` turns into `{"status": "error", "message": …}` — no new error
shape.

## Verification

- Strict on: every `/api/memory/*` route returns 401 to an anonymous caller.
- Strict on: an authenticated caller still gets **their own** bucket, so the
  switch withholds rather than refusing everyone. Without this control a blanket
  break would pass.
- Strict off: an anonymous caller saves and lists as before — the July ruling,
  pinned by a test so it cannot be eroded silently.
- MCP: anonymous raises under strict mode; a token still resolves to its `sub`.

## Risks

- **Two enforcement sites for one flag**, in two processes, which can drift. They
  are covered by tests on both sides, and the coupling is stated in `mcp.md`.
- Turning the flag on strands whatever is already in the `default_user` bucket:
  it stays in the store, readable by nobody, since no authenticated caller
  resolves to that id. Migration is out of scope; the flag is aimed at
  deployments that have not been accumulating anonymous memories.
- The web UI has no login flow for memory, so enabling this on a browser-facing
  deployment requires the caller to present a token by other means.
