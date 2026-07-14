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

## Second review follow-up: opaque credential exchange

The first fix correctly unified the endpoint resolver but did not make an opaque
PAT locally resolvable: the MCP verifier retained the original credential after
`/me` accepted it, while the downstream endpoint's local resolver accepts signed
JWTs/cookies. The original consumer test hid that mismatch by monkeypatching the
resolver.

The verifier now treats `/me` as the authoritative credential-validation and
identity boundary, validates its response, and exchanges any accepted credential
for a five-minute locally signed downstream JWT. The derived token contains user
identity and role but no group claims; search still loads current memberships from
the store on every request. The original PAT is used only for the `/me` request and
is neither retained in the FastMCP access token nor embedded in the derived JWT.

Strict TDD evidence:

- RED: `tests/unit/test_mcp_auth.py` had 8 failures. The original opaque token was
  retained and all malformed `/me` identities were accepted.
- GREEN: all 8 verifier tests pass. The production `resolve_request_user` resolves
  the derived JWT without consumer monkeypatching; malformed/invalid JSON fails
  closed.
