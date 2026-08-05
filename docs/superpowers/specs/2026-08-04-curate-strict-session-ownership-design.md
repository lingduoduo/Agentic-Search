# Curate reads only sessions you own

**Date:** 2026-08-04
**Status:** Approved

## Problem

#494 stopped `curate --session-id` from reading another user's session, but left
ownerless sessions readable by anyone holding the id:

```python
return session is not None and session.user_id in (None, user_id)
```

The justification was the rule this codebase applies to documents —
`SearchFilters.matches` and `demo.py`'s `_allowed_by_acl` both treat "declares no
ACL" as public — and that it preserved `curate --session-id` for anonymous
callers.

A session is not a document. An ownerless session is somebody's actual
conversation, recorded before they signed in, not an unclassified corpus file.
Anyone who learns the id can distil it into their own memories.

## Goals

- `curate --session-id` reads only a session whose `user_id` matches the caller.
- The capability this removes is visible in the response, not silent.

## Non-goals

- No change to the no-flag path. `list_sessions_for_user` is already scoped.
- No retro-assignment of owners to existing NULL sessions. Ownership was never
  recorded for them; inventing one would be a guess.

## Design

`_readable` becomes `session.user_id == user_id`.

**The cost, stated plainly.** Sessions created without a signed-in user are
stored with `user_id = NULL`. The no-flag path is scoped by `WHERE user_id = ?`,
which never matches NULL — so `-session-id` was an unauthenticated caller's
*only* route to their own conversations. Closing it leaves them none: an
anonymous caller can no longer curate from conversations at all. `memory add`,
`list`, `search`, `consolidate` and `profile` are unaffected.

That is a larger loss than "ownerless sessions become unreadable" suggests, so it
is pinned by a test rather than left to be rediscovered.

**The loss is reported, not silent.** #494 chose to skip a denied session and let
`curate` return its generic `{"status": "empty", "message": "no conversations or
notes yet"}`. Reusing that here would make a removed capability read as "nothing
to do" — the invisible-loss shape #490 shipped and #491 had to undo. When an
explicit `session_id` yields nothing, the message becomes `session not found, or
not readable by you`.

One message covers both causes — not yours, and does not exist — so it still
confirms nothing about another user's sessions. No new status, no new error path,
so both callers (the HTTP router and the MCP tool) are unaffected.

## Verification

- Another user's session: not read, LLM never invoked.
- An ownerless session: not read either — the change itself.
- The caller's own session: still read. Without this control a blanket refusal
  would pass.
- An explicit unreadable `session_id` returns the specific message; the no-flag
  path keeps the generic one.
- An anonymous caller has no route left: `list_sessions_for_user("default_user")`
  is empty *and* the by-id call returns empty.

The LLM double raises on any call, so the guarantee is about what enters the
prompt rather than what comes back.

## Risks

- ~~**Anonymous conversation-curation is gone.**~~ This was the cost of strict
  ownership on its own, and the reason for the Addendum below: anonymous
  sessions now carry the anonymous identity, so signed-out callers curate their
  own conversations again. Strict ownership stands; only NULL-owned rows are
  unreachable.
- Existing NULL-owned sessions become permanently uncurateable. They remain
  readable through the chat surfaces; only memory curation loses them.
- A user who curated an ownerless session before this change keeps the memories
  already derived from it. Nothing is retracted.

---

## Addendum: anonymous sessions gain the anonymous identity

Strict ownership left signed-out callers with nothing to curate, because their
sessions were stored with `user_id = NULL` and both routes into a session are
keyed by owner. Rather than accept that, sessions now carry the same identity
their memories already use.

**Why sessions were NULL in the first place.** `chat_sessions.user_id` is a
foreign key to `users(id)`; `user_memories.user_id` is not. So the anonymous
memory bucket worked with an id nothing had provisioned, while a session could
not. `AgenticSearchStore` now provisions a `default_user` row at schema init, and
`create_chat_session` defaults a missing owner to `ANONYMOUS_USER_ID`.

`ANONYMOUS_USER_ID` lives in `db/models.py` and `DEFAULT_MEMORY_USER_ID` aliases
it, so the session owner and the memory bucket cannot drift apart — if they did,
curate would read a different bucket than it writes and silently find nothing.

### What provisioning that row broke

A row in `users` is visible to everything that reads `users`, and three of those
were counting or listing *accounts*:

| Surface | Effect | Resolution |
| --- | --- | --- |
| `/auth/register` | `role = "admin" if not all_users` — the list is never empty, so **no first user ever became admin**, silently, on a fresh deployment | excluded from `list_users()` |
| Daily-active-users analytics | one phantom user per day; the old `user_id IS NOT NULL` filter *was* the anonymous exclusion, and giving anonymous an id defeated it | explicit `user_id != ANONYMOUS_USER_ID` |
| Admin "Users/groups" metric and the admin user list | counted and displayed a synthetic account | excluded from `list_users()` |

The register regression is the reason `list_users()` excludes the row rather than
each call site filtering: three of four callers manage accounts, and the fourth
(integration-test reset) must not delete it. `get_user(ANONYMOUS_USER_ID)`
reaches it deliberately.

**The whole suite passed while first-user-becomes-admin was broken.** Nothing
covered it. A test does now.

### Not migrated

Existing NULL rows stay NULL. `chat_sessions.user_id` is `ON DELETE SET NULL`, so
a NULL row is either a legacy anonymous session *or* an orphaned session whose
owner was deleted — indistinguishable in the data. Adopting them would hand a
deleted user's conversations to every anonymous caller.
