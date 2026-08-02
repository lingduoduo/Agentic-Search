# web_search reports why it failed

**Date:** 2026-08-02
**Status:** Approved

## Problem

Invoking `web_search` returned:

```json
{"response": "No results found.", "raw": [], "errors": []}
```

It looked like a tool that ran fine over a topic with no hits. It was not. On the
machine that reported it:

```
SERP_API_KEY set: True
serpapi → error: 429, message='Too Many Requests'
AGENTIC_SEARCH_BROWSER_SEARCH_URL: not set
```

The SerpAPI quota was exhausted and no fallback leg was configured, so the tool
could not run at all — and said nothing.

`make_web_cascade_search` tries SerpAPI, then the browser server, and ends with
`return []` when neither is usable. Both legs' errors are discarded on the way.
`format_search_pages` renders an empty list as "No results found.", so a missing
key, an exhausted quota, an unreachable browser server and a genuinely empty
search were all indistinguishable.

This is what "no tools can be triggered to run" looked like from the outside.
`search` and `rag_routing_tool` were working the whole time.

## Goals

- An unusable cascade says which leg failed and why.
- A genuinely empty search still reports empty — no invented failures.
- Secrets never appear in the surfaced error.

## Non-goals

- No retry, backoff, or quota tracking for SerpAPI.
- No new provider legs.
- No change to what counts as a usable result (`_pages_are_usable` is untouched).

## Design

The cascade accumulates a `failures: list[SearchPage]` as it goes: error pages
from the SerpAPI leg, error pages or the raised exception from the browser leg.
When neither leg yields a usable result it returns those failures instead of `[]`.
`format_search_pages` already renders an error page as `Error: …`, so the reason
reaches the caller with no changes downstream.

When a leg failed **and** no browser fallback is configured, one more line names
the missing setting — a concrete next step rather than a dead end. That line is
added only when something actually errored: a search that legitimately found
nothing is not a configuration problem, and saying so would be a new lie in the
opposite direction.

Redaction already exists (`_redact_secret_params`), so the surfaced 429 carries
`api_key=[REDACTED]`.

## Verification

The same invocation that returned "No results found.":

```
Error: 429, message='Too Many Requests', url=URL('https://serpapi.com/search.json?…api_key=%5BREDACTED%5D&num=5&start=0')

Error: No browser fallback is configured; set AGENTIC_SEARCH_BROWSER_SEARCH_URL to add one.
```

2824 tests pass (4 new).

## Risks

- The error text reaches the model as a tool result when the agent calls
  `web_search`. That is the intent — a model told the search failed can say so
  instead of inventing an answer — but it is more text than an empty result.
- Only `web_search`'s cascade is fixed. Other tools that swallow failures into
  empty results have the same class of problem and were not audited.
