# Regex-first intent routing — design

## Problem

The entry-point router `route_query` (`src/internal/servers/web/intent_routing.py`)
sends every non-bare query to the **LLM classifier** `classify_route` on the happy
path (when an LLM is available). The classifier is stochastic in spirit and
over-routes obvious cases — e.g. `What is FAISS?` → `search` instead of `chat` —
which then runs the weak local search agent and can return no answer. Meanwhile a
deterministic 3-way rule router (`_rule_based_route`) already exists but is
relegated to the no-LLM fallback, so its correct decision for such queries never
runs.

We want cheap, deterministic **surface-feature checks** that decide the obvious
search / chat / tool cases up front, and only defer the genuinely ambiguous middle
to the LLM.

## Goal

Add a high-precision, anchored regex pass that runs **before** the LLM classifier.
Obvious cases are decided deterministically and for free; anything not confidently
matched falls through to the existing LLM classifier (or the existing lenient
rule-based fallback when no LLM is present). No new dependencies, no ML.

## Non-goals

- Not replacing the LLM classifier — it stays as the fallback for ambiguous input.
- Not a scoring/weighting model — this is a small, ordered rule list, deliberately
  kept to a handful of anchored patterns.
- No change to capability-aware dispatch/degradation in `app.py`.
- Not touching the separate internal retriever router (`src/internal/routing/`).

## Approach

One new function; a three-line change to `route_query`.

```
_regex_route(query) -> RouteStrategy | None     # anchored, confident-only; None = "not sure"

route_query:
  explicit_source            → SEARCH        (unchanged)
  r = _regex_route(query)
  if r is not None: return r                  # NEW deterministic pre-LLM pass
  if llm:  return classify_route(query, llm)  (unchanged)
  else:    return _rule_based_route(query)    (unchanged, lenient no-LLM fallback)
```

- `_regex_route` is **high-precision**: returns a strategy only on a confident
  match, else `None`.
- `classify_route` (LLM) and `_rule_based_route` (lenient no-LLM fallback) keep
  their current roles unchanged. `_is_bare_lookup` is reused inside `_regex_route`.

## Rules (`_regex_route`, in precedence order)

Anchoring `^` means the cue must be at the **start** of the (stripped) query — this
is what separates a command from a description (`send an email` = tool;
`how to send an email` = not a tool command).

1. **TOOL** — starts with an imperative action cue. Split by ambiguity to keep
   precision:
   - **Unambiguous action verbs (bare):**
     `^(send|email|schedule|book|cancel|deploy|assign|notify|remind|invoke|trigger|subscribe|unsubscribe)\b`
     → `TOOL`.
   - **Ambiguous verbs — only when object-qualified** (a bare `open`/`file`/`run`/`post`/`add`/`update`/`create`/`delete` misfires on `open source models`, `file formats`, `post office`):
     `^(create|delete|remove|update|add|open|close|file|post|run|execute) (?:a |an |the )?(ticket|issue|pr|pull request|task|event|meeting|reminder|calendar|record|entry|api|job|workflow|deployment|message|email)\b`
     → `TOOL`.
2. **SEARCH** —
   - `_is_bare_lookup(query)` (short, verb-less term/entity) → `SEARCH`; or
   - starts with a lookup imperative:
     `^(find|search for|look up|look for|retrieve|fetch|pull|list|locate|show me|get me)\b`
     → `SEARCH`.
3. **CHAT** —
   - starts with a question/explain word:
     `^(what|why|how|explain|describe|summarize|compare|tell me about|difference between)\b`; or
   - starts with a generative verb:
     `^(write|draft|translate|rephrase|reword|brainstorm|compose|generate)\b`; or
   - the query **ends with `?`**
   → `CHAT`.
4. **Conflict guard** — if a start-cue for one intent co-occurs with a strong cue
   for another, return `None` (defer to the LLM). The only cross-cue checked is a
   small fixed set of **currency/fact cues** that turn a chat-form question into a
   likely search: `_CURRENCY_RE = \b(latest|current|recent|news|price|stock|weather|today|now)\b`.
   Concretely: if rule 3 (CHAT) would fire but `_CURRENCY_RE` also matches, return
   `None`. This keeps `what is the latest price of NVDA` out of a wrong
   deterministic CHAT and lets the LLM decide.
5. Nothing matched → `None` → LLM (or lenient fallback).

Precedence is strict top-to-bottom: the first confident rule wins. Because tool /
search / chat cues are start-anchored, a query generally starts with exactly one
intent's cue, so conflicts are rare and rule 4 handles the known one.

## Data flow / example

`What is FAISS?` → rule 3 (starts with `what`, ends with `?`), `_CURRENCY_RE` does
not match → `CHAT`, deterministically, **without** an LLM call → reliable grounded
synthesis path. This is the exact query that previously misrouted to `search`.

`FAISS` → rule 2 bare lookup → `SEARCH`. `send an email to Bob` → rule 1 → `TOOL`.
`explain how to send an email` → rule 3 (starts with `explain`) → `CHAT`.
`what is the latest price of NVDA` → rule 3 matches but `_CURRENCY_RE` matches →
`None` → LLM.

## Testing

- **`_regex_route` unit tests**, one behavior each: tool imperatives → TOOL; bare
  term + lookup imperative → SEARCH; question/explain/generative/trailing-`?` →
  CHAT; currency-conflict → None; genuinely ambiguous phrase → None.
- **`route_query` integration** (reuse `_FakeLLM` that records calls): a confident
  regex case returns the strategy and the **LLM is never consulted**
  (`llm.calls == []`); a non-confident case with an LLM **does** consult
  `classify_route`; the no-LLM path still resolves via `_rule_based_route`.
- **Regression:** update the `test_agent_router.py` cases that assumed the LLM
  classifier ran for now-deterministic inputs (e.g. `What is FAISS?` used to reach
  the classifier; now short-circuits to CHAT). Behavior for ambiguous inputs is
  unchanged.

## Success criteria

- `What is FAISS?` (and other `what/why/how … ?` questions without a currency cue)
  route to `chat` deterministically, with no LLM classifier call.
- Clear tool imperatives and bare-term lookups route to `tool` / `search`
  deterministically.
- Ambiguous / currency-conflicted queries fall through to the LLM classifier
  unchanged.
- No behavior change to dispatch/degradation; full suite green.
