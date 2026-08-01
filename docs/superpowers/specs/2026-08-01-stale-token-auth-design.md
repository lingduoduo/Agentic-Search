# Stale identity must not be trusted (and must not crash)

## Problem

Local auth behaved non-deterministically: the same browser session would search
fine, then fail on chat, then work again after a re-login, then fail after a
restart. It reads as "authentication is broken at random". It isn't random.

A JWT stays valid until it expires (7 days). The row it names does not: the web
store defaults to `:memory:` and is rebuilt on every backend restart. So the
normal state after a restart is **a validly-signed token naming a user that no
longer exists** — and the browser keeps sending it.

Such a token was worse than sending no credential at all. Two crashes:

1. **Foreign key.** `chat_sessions.user_id` references `users(id)`. Creating a
   session for a token whose user is gone raises
   `sqlite3.IntegrityError: FOREIGN KEY constraint failed` → 500.

2. **Unique email.** `_ensure_session` tried to auto-provision the token's user
   with `store.upsert_user(UserRecord(id=auth_user.id, email=auth_user.email))`.
   After re-registering, the same email exists under a *new* id, so inserting it
   under the old id raises `sqlite3.IntegrityError: UNIQUE constraint failed:
   users.email` → 500. This is why "I log in, then rerun, and it fails": logging
   in again mints a new id for the same email, and every request still carrying
   the previous cookie collides with it permanently.

Endpoints that only *read* the user were unaffected. Measured, one cookie:

| Endpoint | no cookie | valid | stale |
|---|---|---|---|
| `/api/agent` | 200 | 200 | **500** |
| `/chat/create-chat-session` | 200 | 200 | **500** |
| `/search/send-search-message` | 401 | 200 | **200** |
| `/tool/tool-history` | 200 | 200 | 200 |

Search succeeding while chat 500s, from the same cookie, is the whole complaint.

## Design

One rule: **the store is the source of truth for identity. A token whose user is
not in the store is not authenticated.** Stale then collapses onto the existing
"no credential" path — 401 where auth is required, anonymous where it is not —
and stops being a third, crashing behaviour.

`resolve_active_user(request, store)` in `servers/users/api.py` wraps
`resolve_request_user` with that existence check. Every site that turns a request
into a persisted user id now goes through it:

- `chat_backend._get_user`
- `search_backend._get_user` / `_authenticated_search_filters`
- `tool_backend` send + history
- `app._optional_user_from_request` (used by `/api/agent`, tool approvals)

`_ensure_session` no longer fabricates a user. It validates instead: a `user_id`
it cannot find is logged and dropped, and the session is created anonymously.
This matters beyond tokens — `user_id` also arrives in the `/api/agent` request
body, so a client could previously trigger the same 500 with an arbitrary id.

### Behaviour change worth calling out

Auto-provisioning a user row from token claims is **removed**, not repaired. A
bearer token signed with the shared secret (or a future external IdP) for a user
absent from the store no longer creates that user as a side effect of asking a
question; it is simply unauthenticated. Provisioning belongs to registration and
the SCIM router, which already exist. Anything relying on implicit provisioning
must provision explicitly.

`chat_backend` previously resolved callers with `user_from_headers`, which reads
only the `Authorization` header — chat session listing ignored the login cookie
entirely. Routing it through the shared resolver fixes that inconsistency too.

## Verification

Against the running stack, with the cookie that used to 500:

```
                        before        after
/api/agent              500     →     200 (anonymous)
/chat/create-session     500     →     200 (anonymous)
/search/send-search      200     →     401 (clean)
```

With a valid cookie, everything is 200 — and now survives a backend restart when
`AGENTIC_SEARCH_WEB_DB_PATH` points at a file, which is the actual fix for having
to log in repeatedly.

`tests/unit/servers/web/test_stale_token_auth.py` covers both crashes, the
"stale behaves exactly like no credential" invariant, the anonymous-session
attribution, and the request-body `user_id` path.

## Not in scope

- **A dev bypass that skips login entirely** (`AGENTIC_SEARCH_DEV_USER`, seeding
  a real dev user so no cookie is needed). Reasonable for local work and a
  natural follow-up, but it is a second auth bypass and a deliberate decision,
  not a bug fix — it should be opted into on its own merits.
- **Making the anonymous policy uniform.** Assistant, Chat and Tool accept
  anonymous callers; Search demands auth. That inconsistency is real and
  user-visible, but it is a product decision about which surfaces require login,
  not something to change while fixing crashes.
- **Changing the default DB path** away from `:memory:`.
