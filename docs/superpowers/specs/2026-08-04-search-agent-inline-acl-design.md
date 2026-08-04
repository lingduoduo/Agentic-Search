# The search agent filters before the model reads, not after

**Date:** 2026-08-04
**Status:** Approved

## Problem

`SearchAgentLoop` retrieves documents across multiple turns and feeds them into
the model's context. Its access filters are sent to the retrieval server but
never applied to what comes back:

```
app.py                _run_search_agent(..., filters=_filters_payload(filters))   → a dict
SearchAgentLoopConfig filters: dict | None
search.py:574         _retrieve_many sends the dict to the client, NO post-filter
                      ↓ documents enter the loop's context
                      ↓ the model reads them and writes its answer
app.py:903            _enforce_access(docs, filters)   ← filters the returned list
```

The returned **documents list** is therefore clean while the **answer text** can
quote a document the caller may not read. Enforcement happens after generation,
which is exactly the failure the tool-agent spec's Rejected Alternatives named
when it refused to filter inside `ToolAgentLoop`:

> The loop feeds tool results to the model; anything filtered after that point
> has already been read.

`demo.py` and `hybrid.py` honour `access_acl` since #490, so the bundled stack
is currently covered by that second layer. A retrieval backend outside this repo
is free to ignore the field, and then nothing stands between it and the model —
the same assumption that produced a false pass in #487 and a live bypass in #488.

## Goals

- The search agent's model never sees a document the caller may not read.
- Enforcement holds whatever the retrieval server does.

## Non-goals

- No change to the web retriever. Web results carry no ACL.
- No change to `app.py`'s post-hoc `_enforce_access` on the returned documents.
  It becomes redundant for this path rather than wrong, and removing it would
  drop the belt while the braces are still being fitted.
- No change to `SearchFilters.matches`.

## Design

`SearchAgentLoopConfig.filters` changes from the wire dict to a `SearchFilters`
object, and `_retrieve_many` does what `build_search_routing_tool` already does:
serialize at the wire, enforce on return.

```python
filters = self.search_config.filters if retriever is not Retriever.WEB else None
# Built OUTSIDE the try below: _retrieve_many swallows exceptions into an empty
# result set, so a mis-typed filter in here would degrade to "no results" with
# only a log line — the same silent shape that hid #487's broken enforcement.
# Let it raise instead.
payload = filters.to_payload() if filters is not None else None
try:
    results = await self._client_for(retriever).retrieve(queries, filters=payload)
except Exception as exc:
    logger.warning("Search failed for queries %r: %s", queries, exc)
    return [[] for _ in queries]
if filters is not None:
    results = [[r for r in row if filters.matches(r.metadata)] for row in results]
return results
```

This upholds the project invariant — a serialization paired with an enforcement
in the same place — and moves the enforcement point before the model's context
rather than after its answer.

**The type change has exactly one producer.** Every `SearchAgentLoopConfig(...)`
construction in the tree was checked: only `app.py`'s `_run_search_agent` passes
`filters=`, and it currently wraps the value in `_filters_payload(...)`. That
wrapping is dropped; the object is passed through. The training and eval scripts
(`run_bamboogle_eval`, `run_retriever_aware_grpo`, `run_bamboogle_synthetic_grpo`,
`run_agentic_search`) construct the config without filters and are unaffected.

`filters=None` keeps the loop unfiltered, as today.

## Verification

- A retrieval server that returns a document outside the caller's ACL: the
  document must be absent from what the loop puts in the model's context, not
  merely absent from the returned list. Assert on what the *client mock was
  asked for and what the loop kept*, so the test fails if the post-filter is
  dropped even though `app.py`'s later `_enforce_access` would still clean the
  returned documents.
- The payload reaching the client is JSON-serializable — the crash that masked
  #487's enforcement for months came from passing an object here.
- `filters=None` returns every document.
- Web retrieval is never filtered.

## Risks

- **`app.py`'s `_enforce_access` on this path becomes redundant.** Deliberate:
  two layers, and the outer one also covers the degraded no-local-model branch
  that does not use this loop at all.
- The loop's filtering is per retrieval call, so a restricted document still
  costs retrieval bandwidth and can displace an accessible one from `top_k`
  before being dropped — the same post-filter limitation the rest of this
  system has.
- `SearchAgentLoopConfig` is part of the agents package that training code
  imports. The field's type changes, so an out-of-tree caller still passing a
  dict now raises on `.to_payload()`. That is deliberate and is why the payload
  is built outside the `try`: inside it, the failure would be swallowed into an
  empty result set and read as "the corpus had nothing". Only one in-tree
  producer exists, and it is updated here.
