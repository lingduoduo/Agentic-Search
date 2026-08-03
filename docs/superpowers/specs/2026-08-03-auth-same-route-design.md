# Signing in narrows results, it does not change the route

**Date:** 2026-08-03
**Status:** Approved

## Problem

The same query behaved differently once a user signed in. Measured on `/api/agent`
with `RAG` against the same corpus:

| | Route | Docs | Time |
| --- | --- | --- | --- |
| Anonymous | `search_mode: direct`, `tier: semantic` | 2 | 0.56s |
| Signed in | `search_mode: filtered_pipeline` | 5 | 8.17s |

Signed in, the query was silently expanded into five (`RAG`, `RAG model`,
`RAG architecture`, `RAG applications`, `RAG benefits`) and web results were
mixed in. Same input, different answer, 15× slower, no indication why.

`_run_search_direct_or_escalate` diverted every filtered query to
`_auto_search_pipeline`. `filters` is non-`None` for *any* authenticated user,
because `build_user_only_filters` always returns at least `["public"]` — so the
divert fired on every signed-in request regardless of what the ACL actually
restricted.

The intended contract is the opposite: **authentication should narrow results to
what the user may see and add what they have special access to, not swap the
engine.**

## The trap

The guard's comment justified itself with:

> `_run_direct_search` and `_run_search_agent` do not thread per-user access
> filters into retrieval — `search_tool` and the loop take no `filters`.

That is outdated — #407 threaded `filters` end-to-end and every call below the
guard already passed them. But removing the guard on that basis alone would have
opened an access-control hole, because **passing filters is not enforcing them**:

- `demo.py` and `hybrid.py` accept the `filters` field and **ignore it**. Only
  the full `RetrievalService` honours it. Verified directly: the demo server
  returns a document ACL'd to another user.
- Enforcement for the pipeline path lives client-side in
  `search_runner._apply_filters`, which the direct path never reaches.
- `SearchPage` — which every direct-path result passes through — had **no
  metadata field at all**, so the document's ACL was discarded before anything
  could check it.

So the guard was load-bearing for the wrong stated reason. The fix is to make
enforcement real on the route, not to trust the server.

## Design

**Carry the ACL.** `SearchPage` gains `metadata`, populated from
`SearchResult.metadata`; `_documents_from_search_pages` merges it into the
`ContextDocument` beneath the labels it adds, so a document's own `acl` survives
to the caller.

**Enforce on the route.** `_enforce_access(documents, filters)` drops documents
whose ACL does not intersect the caller's, applied to the direct path before the
match gate and to the escalated path's documents. Documents that declare no ACL
stay public, matching `SearchFilters.matches`. This holds whatever the retrieval
server does.

**Remove the divert.** With enforcement real, filtered queries take the same
route as anonymous ones.

## Verification

Two documents, identical titles, served by `demo.py` (which ignores filters):
`pub_1` ACL `["public"]`, `sec_1` ACL `["user:someone_else"]`.

| | Titles | Confidential text present |
| --- | --- | --- |
| Anonymous | `Zebra Handbook` ×2 | **yes** |
| Signed in | 5 web results | **no** |

And the original symptom, same corpus, query `RAG`:

```
ANON: mode=direct tier=semantic docs=2 queries=['RAG']
AUTH: mode=direct tier=semantic docs=2 queries=['RAG']
```

8.17s → 0.027s, identical routing. 2830 tests pass.

## Risks

- **Anonymous requests are unfiltered.** With no user there is no ACL, so an
  anonymous caller sees restricted documents — visible in the table above. That
  is pre-existing and unchanged here, but it means anonymous access and
  restricted documents must not be combined. Making anonymous imply
  `["public"]` is a policy decision, not a bug fix, and is not in this change.
- Enforcement is a post-filter: a restricted document still costs retrieval
  bandwidth and can displace an accessible one from top-k before being dropped.
  Correct, but not equivalent to filtering in the index.
- `SearchPage.metadata` now flows into `ContextDocument.metadata` for every
  search path. Label keys still win on a clash, so existing consumers are
  unaffected, but retrieval-side keys are newly visible downstream.
