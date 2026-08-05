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

- **Anonymous conversation-curation is gone.** Deliberate and the point of the
  change, but it is a capability removal on a documented workflow, not a
  hardening no one can feel. `docs/cli.md` says so in a callout.
- Existing NULL-owned sessions become permanently uncurateable. They remain
  readable through the chat surfaces; only memory curation loses them.
- A user who curated an ownerless session before this change keeps the memories
  already derived from it. Nothing is retracted.
