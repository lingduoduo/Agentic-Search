# Plan: stale identity must not be trusted

Spec: [2026-08-01-stale-token-auth-design.md](../specs/2026-08-01-stale-token-auth-design.md)

## Task 1 — Reproduce both crashes deterministically

Register a user, then present a validly-signed token for a *different* id
carrying the same email.

**Verify:** `/api/agent` → 500 (`UNIQUE constraint failed: users.email`),
`/chat/create-chat-session` → 500 (`FOREIGN KEY constraint failed`). ✅ done

## Task 2 — Add `resolve_active_user`

In `servers/users/api.py`, wrap `resolve_request_user` with a store-existence
check; return `None` when the row is absent.

**Verify:** unit test asserts `None` for an unknown id and the user once the row
is created. ✅ done

## Task 3 — Route every persisting call site through it

`chat_backend`, `search_backend`, `tool_backend`, and
`app._optional_user_from_request` (3 call sites). Drop imports left orphaned
(`user_from_headers`, `UserRecord` in `app.py`).

**Verify:** `grep resolve_request_user src/` shows only the resolver itself and
`/me`'s `_require_auth`; `ruff check .` clean. ✅ done

## Task 4 — Stop `_ensure_session` fabricating users

Remove the `upsert_user` call; validate `request.user_id` against the store and
fall back to an anonymous session, logging the dropped id. Drop the now-unused
`auth_user` parameter and update its caller.

**Verify:** posting `/api/agent` with `user_id: "nobody_by_that_name"` returns
200 and creates a session with `user_id is None`. ✅ done

## Task 5 — Regression tests

`tests/unit/servers/web/test_stale_token_auth.py`: both crashes, stale ≡ no
credential, anonymous attribution, search 401, valid login still works, and the
body-supplied `user_id` path.

**Verify:** 8 tests pass. ✅ done

## Task 6 — Repair tests that relied on the old behaviour

Expected fallout — these authenticated users that were never in the store:
- `test_chat_backend.py`, `test_tool_backend.py`: repoint monkeypatch targets to
  `resolve_active_user`.
- `test_search_backend.py` (2), `test_sse_streaming.py`,
  `test_web_experience_app.py`: seed the caller so the intent (ACL from store,
  approval flow enabled) is preserved rather than the assertion weakened.

**Verify:** full `pytest` green. ✅ done — 2765 passed.

## Task 7 — End-to-end against the running stack

Restart the backend on the fixed code and replay the cookie that used to 500.

**Verify:** stale → 200/200/401 (was 500/500/200); valid → all 200 and survives a
restart with a file-backed DB. ✅ done
