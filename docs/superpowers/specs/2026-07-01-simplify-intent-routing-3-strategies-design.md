# Simplify Intent Routing: 4 Strategies → 3 — Design

Status: draft
Date: 2026-07-01

## Goal

Collapse the entry-point router's **4** `RouteStrategy` values to the **3**
user-facing intents the README advertises (`search` / `chat` / `tool`), by
merging `direct_llm` and `agentic_rag` into a single grounded `chat` strategy.
This removes a two-vocabulary mismatch (4 internal strategies vs 3 surfaced
intents), deletes an ungrounded parametric path, and drops one orphaned helper.

## Problem

Today `route_query` (`src/internal/servers/web/intent_routing.py`) picks one of
four strategies:

- `DIRECT_LLM` — bare `llm.complete()`, no retrieval (fires only on `_DIRECT_RE`:
  `write`, `translate`, `hello`, `joke`, `poem`, …).
- `AGENTIC_RAG` — `AgenticRAGLoop`, decompose + HyDE + grounded synthesis (the
  default when no signal dominates).
- `SEARCH_AGENT` — multi-turn `SearchAgentLoop`.
- `TOOL_AGENT` — `ToolAgentLoop` function calling.

But the frontend only ever sees **3** intents (`search` / `chat` / `tool`), and
both `DIRECT_LLM` and `AGENTIC_RAG` surface as `chat`. So the internal enum
carries a distinction the product never exposes, and readers must mentally map 4
strategy names onto 3 intent labels. Separately, `_rule_based_is_search` — an
older 2-way (search-vs-not) helper — is now referenced only by tests; the 4-way
`_rule_based_route` superseded it.

## Approach

### 1. Collapse the enum to 3 intent-aligned values

`RouteStrategy` becomes:

| New value | Replaces | Dispatched loop |
| --- | --- | --- |
| `CHAT = "chat"` | `AGENTIC_RAG` + `DIRECT_LLM` | `AgenticRAGLoop` (grounded synthesis) |
| `SEARCH = "search"` | `SEARCH_AGENT` | `SearchAgentLoop` |
| `TOOL = "tool"` | `TOOL_AGENT` | `ToolAgentLoop` |

The values are renamed to `chat` / `search` / `tool` so `extra["route"]` speaks
the same vocabulary as the user-facing `intent`. These two fields remain
distinct in meaning:

- `route` = the strategy we **chose** for the query.
- `intent` = what **actually ran** after any capability-aware degradation.

They can legitimately differ — e.g. `route="tool"` but the tool backend is
unavailable → degrades to RAG → `intent="chat"`; `route_degraded` records why.
Sharing one vocabulary makes that relationship legible rather than obscuring it.

### 2. Delete the `DIRECT_LLM` path

- Remove the `_DIRECT_RE` regex, the `direct_llm` bullet from `_ROUTE_PROMPT`,
  and the `direct_llm` branch in `_rule_based_route`.
- Remove the `DIRECT_LLM` dispatch block in `_run_auto_routed`
  (`src/internal/servers/web/app.py`), including its parametric
  `plain_generation` local-only fallback.
- New `_rule_based_route` precedence: **tool > search > bare-lookup > chat
  (default)**. The default is `CHAT`, so when no signal dominates a query gets a
  grounded answer — same safe default as before, just under the merged name.

**Accepted tradeoff:** generative / conversational asks ("write a poem",
"hello") now route to `CHAT` → grounded RAG instead of a fast parametric reply.
This is the intended behavior (always-grounded); `AgenticRAGLoop` must answer
sensibly when retrieval returns nothing relevant (see Tests).

### 3. Remove dead code

Delete `_rule_based_is_search` (production-unused, superseded by
`_rule_based_route`) and its dedicated tests.

## Out of scope (unchanged)

- `_infer_intent_from_output` — still used by the explicit tool-agent finalize
  path (`app.py:619`); keeps mapping the first tool call to a surfaced intent.
- The LLM classifier `classify_route` — kept; only its label set shrinks (drops
  `direct_llm`). The user chose to keep classification, not go pure-rule-based.
- Capability-aware degradation in `_run_auto_routed`.
- `IntentPipeline` (`src/model/intent_classifier.py`) — a separate ML model that
  tunes retrieval settings; unrelated to the entry-point router.

## Data flow (after)

```
POST /api/agent (mode=None)
  └─ _run_auto_routed(query)
       ├─ route_query() → RouteStrategy ∈ {CHAT, SEARCH, TOOL}
       │    ├─ explicit_source            → SEARCH
       │    ├─ _is_bare_lookup            → SEARCH
       │    ├─ classify_route(llm)        → 3-label LLM classify (rule-based on error)
       │    └─ _rule_based_route          → tool > search > bare-lookup > chat
       └─ dispatch (capability-aware, degrades as today)
            ├─ TOOL   → ToolAgentLoop        (degrade → CHAT/RAG)
            ├─ SEARCH → SearchAgentLoop      (degrade → _auto_search_pipeline)
            └─ CHAT   → AgenticRAGLoop        (degrade → _auto_search_pipeline)
```

## Tests

- `test_intent_routing.py`: drop the `_rule_based_is_search` tests; update
  route-classification tests so former `direct_llm` cases (`write`, `translate`,
  greetings) now assert `RouteStrategy.CHAT`.
- `test_sse_streaming.py`: the `route == "direct_llm"` assertion becomes the new
  vocabulary (`chat` for that query, or an updated fixture query).
- New: a generative query (e.g. "write a haiku") routes to `CHAT` and
  `_run_auto_routed` returns a non-crashing answer even when retrieval yields
  zero relevant documents (guards the accepted-tradeoff behavior).

## Doc updates

- README: the Intent Routing table (strategy→intent), the `extra["route"]`
  value list, and the SSE `done`-event `route` field.
- Note in the routing section that `route` and `intent` share a vocabulary but
  differ in meaning (chosen vs actually-ran).

## Success criteria

- `RouteStrategy` has exactly three members: `CHAT`, `SEARCH`, `TOOL`.
- No `direct_llm` references remain in `src/` (grep-clean).
- `_rule_based_is_search` is gone.
- `pytest tests/unit/test_intent_routing.py tests/unit/servers/web/test_sse_streaming.py`
  passes, plus the new generative-query test.
- Full `pytest` suite green.
