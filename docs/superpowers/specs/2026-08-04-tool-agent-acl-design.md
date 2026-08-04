# The tool agent's corpus search obeys the caller's ACL

**Date:** 2026-08-04
**Status:** Approved

## Problem

Every `/api/agent` retrieval route now enforces the caller's ACL (#487, #488).
The tool agent does not. `build_search_routing_tool`
(`src/internal/tools/routing_tools.py`) closes over `search_url` and `top_k`
only; its body calls `search_tool(provider="retrieval")` with **no filters and
no post-filter**, so the `search` tool offered to `ToolAgentLoop` returns
whatever the corpus holds, to whoever asked.

It reaches an agent by two paths:

| Path | Instance | Filtered |
| --- | --- | --- |
| Seeded at process start into the global registry (`knowledge_base.py:45`) | shared, built where no request identity exists | no |
| Rebuilt per request in `_run_tool_agent` when `with_search_tool=True` | request-bound | no — but could be |

The auto route passes `with_search_tool=False` (`app.py:1100`), so the **default
path receives the shared, unfiltered instance**. #488 recorded this as a
Non-goal; this spec closes it.

The per-request rebuild already exists. It rebinds the retrieval URL, not the
identity.

## Goals

- The corpus `search` tool returns only documents the caller may read.
- No unfiltered corpus-search tool can reach an agent loop by construction, not
  by each call site remembering.
- Retrieval backends shipped in this repo honour the `access_acl` they are sent.

## Non-goals

- No change to `web_search`. Web results carry no ACL; filtering them is meaningless.
- No change to MCP-sourced tools. They authenticate to their own server with the
  caller's bearer token, which is that server's business.
- No per-document sharing model. ACLs remain whatever retrieval returns.
- No filtering inside `ToolAgentLoop`. By the time the loop holds a tool result
  the model has already been handed it — see Rejected alternatives.

## Design

### 1. The builder takes filters, and enforces them

```python
build_search_routing_tool(*, search_url, top_k, name="search", filters=None)
```

Inside `_execute`, both halves, in one place:

- send `filters.to_payload()` to `search_tool`, for backends that honour it;
- drop pages failing `filters.matches(page.metadata)` before serializing.

Both, because `demo.py` and `hybrid.py` accept the field and ignore it. This is
the invariant established by #488's post-mortem: a serialization without a
paired enforcement converts a fail-closed crash into a silent cross-user read.

This is only possible because #487 added `metadata` to `SearchPage`. Before
that the ACL was discarded before any tool could see it.

`filters=None` keeps today's unfiltered behaviour, so non-web callers
(training scripts, evals) are unaffected.

### 2. The runner never hands out the shared instance

`_run_tool_agent` drops `_CORPUS_SEARCH_NAME` from the registry-derived list
**unconditionally**, then adds back a request-bound tool when
`with_search_tool` is set. Today the name is displaced only when that flag is
true, which is exactly how the auto route ends up with the shared one.

The runner gains `filters: SearchFilters | None`. It takes filters rather than
`RequestCapabilities` because the tool needs an ACL, not an identity, and the
tools package should not import the access package for this. `user_present`
stays as it is — already established and tested.

### 3. The shared instance is not agent-callable

`seed_tools` registers the corpus search with `agent_callable=False`. It stays
listed and invocable through `/admin/tools` and the Dev Console, but no agent
loop can ever be handed it. This reuses the mechanism already in `ToolEntry`
(`agent_callable`, `user_scoped`) and makes the guarantee structural: the unsafe
object cannot reach a loop even if a future call site forgets step 2.

### 4. The bundled retrieval servers honour `access_acl`

`demo.py` and `hybrid.py` filter their results by the `access_acl` in the
request payload, treating a document with no declared ACL as public — matching
`SearchFilters.matches`. Defense in depth, not the primary control: the web
layer keeps enforcing regardless, because a third-party backend can still
ignore the field.

## Rejected alternatives

**Filter inside `ToolAgentLoop`.** The loop feeds tool results to the model;
anything filtered after that point has already been read. Enforcement has to
happen before the tool returns.

**Server-side enforcement alone.** Necessary but not sufficient. The tool sends
no filters today, so there is nothing for a server to honour, and any backend
outside this repo remains free to ignore them.

## Verification

- A document ACL'd to another user is absent from the tool's result for a
  signed-in caller and for an anonymous one, against `demo.py` *before* step 4
  lands — proving the web layer enforces without server cooperation.
- The agent's tool list never contains an unfiltered corpus search: with
  `with_search_tool=False`, `search` is absent entirely; with it true, the
  instance present is the request-bound one.
- `filters=None` returns every document, so non-web callers are unchanged.
- After step 4, `demo.py` and `hybrid.py` drop restricted documents on their own,
  verified by calling `/retrieve` directly with an `access_acl` payload.

## Risks

- **The tool agent's recall drops for restricted corpora.** Intended, but a
  corpus whose documents all declare owner-specific ACLs will return little to
  an anonymous caller. Documents with no ACL stay public, so the demo corpus is
  unaffected.
- Enforcement remains a post-filter: a restricted document still consumes
  retrieval bandwidth and can displace an accessible one from `top_k` before
  being dropped. Correct, but not equivalent to filtering in the index. Step 4
  reduces the exposure for the bundled backends without eliminating it.
- `agent_callable=False` on the seeded instance means a future contributor who
  adds a new agent surface gets no corpus search rather than an unfiltered one.
  Failing closed is the intent; the failure mode is a missing capability, which
  is visible, rather than a silent leak, which is not.
