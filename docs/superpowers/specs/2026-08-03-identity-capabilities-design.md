# Identity shapes capability

**Date:** 2026-08-03
**Status:** Approved

## Problem

Signing in should narrow results to what a user may see and add what they have
special access to — data, tools, and their own memory. Today it does none of
that consistently.

**Anonymous is unfiltered.** With no user there is no ACL, so an anonymous
caller reads documents restricted to someone else. Verified against `demo.py`
with two identically-titled documents, one ACL'd to `user:someone_else`: the
anonymous caller received its confidential text. Signing in *narrows* correctly
(#487), but signing out removes the fence entirely.

**Memory is dormant.** `memory_preamble(db, user_id)` is wired at
`app.py:1414` behind `AGENTIC_SEARCH_MEMORY_INJECTION`, which defaults to off.
The feature is built and invisible — the same pattern that left the MCP client
unreachable until it was deliberately looked for.

**Tools ignore identity.** Every caller is offered the same registry. The
memory tools are inherently per-user, so an anonymous `save_memory` writes into a
shared `default_user` bucket and pools unrelated people's memories together.

Underneath all three: identity is derived independently at each site. That is how
#487's defect arose — one path enforced the ACL, another did not.

## Goals

- Anonymous means `["public"]`, not "unfiltered".
- A signed-in user's memory is injected because they are signed in, with no flag
  to discover.
- User-scoped tools are offered only when there is a user to scope them to.
- One place decides what an identity is entitled to.

## Non-goals

- No change to *which route* runs. #487 established that signing in must not
  switch pipelines; this builds on it.
- No new authentication mechanism, login UI, or token format.
- No per-document sharing UI. ACLs are whatever retrieval already returns.
- No anonymous memory. Logged-out callers get none rather than a session bucket.
- **The tool agent's corpus `search` tool stays unfiltered.**
  `build_search_routing_tool` (`src/internal/tools/routing_tools.py`) calls
  `search_tool` with no filters and applies no post-filter, so the `search`
  offered by `ToolAgentLoop` reads the whole corpus regardless of who is asking.
  The ACL guarantee in this spec covers the `/api/agent` retrieval routes
  (`search_tool`, `hybrid_search`, `search_agent`, `chat_loop`, and the auto
  route's retrieval strategies) — it does not cover tools the agent chooses to
  call, and that exception includes the auto route's own `RouteStrategy.TOOL`
  branch, which runs `ToolAgentLoop` against the seeded tool. Closing this needs
  per-request tool construction: the leaking instance is seeded into the global
  `ToolRegistry` at process start (`knowledge_base.tool_knowledge_base`), where
  no request identity exists, and `tool_agent_runner` only rebinds it when
  `with_search_tool=True`. Threading identity through the registry is a change
  of its own and is tracked separately.

## Design

### One resolver

`src/internal/access/capabilities.py`:

```python
@dataclass(frozen=True)
class RequestCapabilities:
    user_id: str | None
    access_acl: list[str]      # never empty; ["public"] when anonymous
    memory_preamble: str       # "" when anonymous


def resolve_capabilities(user, store) -> RequestCapabilities: ...
```

`store` is required, not optional: the preamble is read from it. Passing it
explicitly keeps the resolver a plain function with no global state, so it can be
called from the agent loops and MCP paths that a FastAPI dependency could not
reach.

Anonymous stops being the absence of a user and becomes an identity with exactly
`["public"]`. `access_acl` is never empty, so no caller can accidentally mean
"unfiltered" by passing an empty list — the distinction that made the current
hole possible.

Three sites read from it instead of deriving identity themselves:

1. `app.py`'s filter construction, which today calls `build_user_only_filters`
   only when a user exists and passes `None` otherwise. It becomes
   `SearchFilters(access_acl=capabilities.access_acl)` unconditionally.
2. The memory injection at `app.py:1414`.
3. `_run_tool_agent`, which asks the registry for the tools matching
   `user_present=capabilities.user_id is not None`.

`SearchFilters` is therefore never `None` on these paths. #487 already removed
the routing branch that keyed off `filters` being truthy, so a non-`None`
anonymous filter changes what is retrieved, never which route runs.

### Tools stay in the registry

Capabilities deliberately do **not** carry a tool list. `ToolEntry` gains
`user_scoped: bool`, set at MCP registration from `AGENTIC_SEARCH_MCP_USER_SCOPED`
(defaulting to `save_memory`, `update_memory_from_conversation`,
`generate_user_profile`, `get_user_profile`, `search_memories`,
`consolidate_memories`) — the same shape as the existing `agent_exclude`. The
registry answers `tools_for(user_present=...)`.

This keeps the registry the single source of truth about tools and keeps the
resolver ignorant of tool names, so adding a user-scoped tool is a registration
concern rather than an edit in two places.

### Memory

`AGENTIC_SEARCH_MEMORY_INJECTION` is removed. The preamble is built whenever a
user resolves, and is `""` otherwise. One less dormant flag, and the behaviour
follows from identity rather than configuration.

### Failure

A user id that no longer resolves degrades to anonymous — public-only, no
memory, no scoped tools — rather than erroring, matching #476's "the store is the
source of truth". An absent MCP server registers no scoped tools, which is
already how it behaves.

## Verification

- Anonymous receives exactly `["public"]`; signed-in receives public plus their
  own `user:` / `email:` / `group:` entries; an unresolvable id degrades to
  anonymous.
- The end-to-end leak test from #487, extended: a document ACL'd to another user
  must be invisible to an **anonymous** caller. It fails today — that failure is
  the point of this change.
- A signed-in request carries a non-empty preamble with no flag set; an anonymous
  one carries none.
- `tools_for(user_present=False)` omits every user-scoped tool;
  `tools_for(user_present=True)` includes them.

## Risks

- **Anonymous becomes strictly less capable.** Intended, but a breaking change
  for any deployment relying on open read access. Documents that declare no ACL
  remain public, so a corpus without ACL metadata — including the demo corpus —
  is unaffected.
- **Memory injection changes prompts for every signed-in request**, spending
  tokens and shifting answers. It is what the feature is for, but it is no longer
  possible to turn off without signing out.
- Enforcement stays a post-filter: a restricted document still consumes retrieval
  bandwidth and can displace an accessible one from top-k before being dropped.
  Correct, but not equivalent to filtering in the index.
- This branches from #487, which is unmerged. If #487 changes in review, this
  must be rebased rather than merged alongside.
