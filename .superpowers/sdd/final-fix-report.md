# Final review fix report

## Root cause

- `/me` resolved identity through the users router's request resolver, while
  `/search/send-search-message` independently called `user_from_headers`.
  Any opaque/PAT-capable identity path accepted by `/me` was therefore bypassed
  by the endpoint used by MCP retrieval.
- Search ACL construction trusted `AuthenticatedUser.group_ids`, which are token
  claims and can become stale after a membership grant or revocation.
- Two pre-existing query/chat tests called the now-authenticated search endpoint
  without credentials; production correctly returned `401`.

## RED

`pytest tests/unit/servers/query_and_chat/test_search_backend.py tests/unit/servers/query_and_chat/test_query_and_chat.py -q`

- 2 failed, 24 passed.
- The opaque-token regression failed because search had no shared request resolver.
- The group regression showed `group:engineering` from stale claims instead of
  `group:current-group` from the authoritative store.

## GREEN

- Promoted the `/me` request resolver to `resolve_request_user` and reused it in
  search routes, keeping one credential-resolution boundary.
- Search ACLs now query `store.list_group_ids_for_user(user.id)` on every request.
- Caller-supplied ACLs remain discarded and anonymous requests remain `401`.
- Streaming and non-streaming compatibility tests now send valid authenticated
  headers rather than weakening production authentication.

## Verification

- Focused search/query tests: `26 passed`.
- MCP/search/context regression suite: `174 passed`.
- Ruff check: passed.
- Ruff format check: passed.
- `git diff --check`: passed.

## Concerns

- Live PAT integration was not run because it requires the external PAT API and
  service stack. The unit regression exercises an opaque PAT-equivalent identity
  returned by the same resolver used by `/me`.
- Pre-existing edits to `.superpowers/sdd/task-3-report.md` and
  `.superpowers/sdd/task-4-report.md` were intentionally left out of this fix.
