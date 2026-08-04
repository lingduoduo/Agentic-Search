# Curating from a session you may not read

**Date:** 2026-08-04
**Status:** Approved

## Problem

`POST /api/memory/curate` takes an optional `session_id`. `_gather_sources`
(`src/internal/memory/service.py`) resolves it without checking who owns it:

```python
sessions = (
    [store.get_chat_session(session_id)]        # any session in the table
    if session_id
    else store.list_sessions_for_user(user_id)  # WHERE user_id = ?
)
```

The two branches disagree about whose data the caller may see. The unfiltered
branch has always been scoped by SQL. The by-id branch returns any row, and its
messages then go into the curation prompt, from which the model writes memories
into **the caller's** bucket. Naming another user's session is therefore a
cross-user read of a full conversation transcript, laundered into the caller's
own memory store.

Both entry points reach it: the HTTP router (`memory/router.py:95`) and the MCP
tool (`mcp_server/tools/memory.py:89`). Neither checks either. `_uid()` falls
back to a shared `default_user` bucket when no credential is present, so an
unauthenticated caller can do it too.

Session ids are `session_<uuid4 hex>` and so not guessable in bulk. An
unguessable identifier is not an access control: ids appear in logs, in the Dev
Console, and in anything a user pastes.

## Goals

- A caller may only curate from sessions they could already read.
- The check sits where the sessions are read, so both call sites inherit it.

## Non-goals

- **The shared `default_user` bucket stays.** Anonymous callers pooling into one
  memory bucket is plausibly the intended single-user dev default, and the MCP
  tool depends on it (`_uid` there is unconditionally `DEFAULT_MEMORY_USER_ID`).
  Changing it is a separate decision about what anonymous memory means.
- No new error path. See below.
- No change to `list_sessions_for_user`, which is already correct.

## Design

`_gather_sources` filters the by-id result through an ownership predicate:

```python
def _readable(session, user_id: str) -> bool:
    return session is not None and session.user_id in (None, user_id)
```

A denied session is **skipped**, not reported. `curate` then finds no sources and
returns its existing `{"status": "empty"}`. This is preferred over a 404/403
because it needs no new error plumbing in either caller, and because the response
is then identical to the one for a session id that does not exist — nothing
confirms that another user's session is real. The `None` return from
`get_chat_session` already flowed to the same place, so this reuses a path that
exists rather than adding one.

**Ownerless sessions stay readable**, which is the one judgment call here. The
codebase already resolves this exact question for documents: `SearchFilters.matches`
and `demo.py`'s `_allowed_by_acl` both treat "declares no ACL" as public. Sessions
created without a signed-in user are stored with `user_id = NULL`, and anonymous
callers share the `default_user` bucket, so requiring strict equality would
silently delete `curate --session-id` for every anonymous caller — the same
invisible capability loss #490 took on the auto route and #491 had to undo. The
leak that matters, reading an *identified* user's transcript, is closed either
way.

Placing the check in `_gather_sources` rather than in the router follows the
invariant this codebase learned the hard way in #488 and #490: enforce where the
data is read, not at each call site that remembers to.

## Verification

- Curating with another user's `session_id` returns `empty` **and the LLM is
  never called** — asserted with a double that raises on any use, so the test
  fails if the transcript reaches the prompt even when the output looks clean.
- The caller's own `session_id` still yields the transcript. Without this control
  a blanket refusal would pass.
- An ownerless session stays readable.

## Risks

- Ownerless sessions remain readable by any caller who knows the id. That is the
  status quo for anonymous content, and consistent with how undeclared-ACL
  documents are treated. Tightening it would need the shared-bucket question
  settled first.
- A user who mistypes their own session id gets `empty` rather than an error.
  Accepted: it is the response they already get for an id that does not exist.
