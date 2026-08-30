# Async fan-out concurrency — design

## Problem

An audit of every `async def` / `await` in `src/` (253 coroutines, 331 awaits)
looked for the shapes that actually cost latency: blocking calls inside
coroutines, and independent I/O awaited one call at a time.

There are no blocking calls inside coroutines — no `requests`, no `time.sleep`,
no synchronous file I/O on any request path. Every known blocking boundary is
already offloaded with `asyncio.to_thread` or `run_in_executor`.

Three sites did await independent I/O sequentially, so their latency was the
**sum** of the calls rather than the slowest one:

| Site | Independent work | Sequential cost |
|---|---|---|
| `AgenticRAGLoop.run` | one retrieval per enhanced query, up to 5 per round × 3 rounds | `n × retrieval_latency` |
| `collect_tool_evidence` | up to `max_calls` read-only tool calls, each with its own `timeout_seconds` | `max_calls × timeout_seconds` |
| `register_mcp_tools` | one discovery session per configured MCP server | each unreachable server's connect timeout delays the next server |

`SearchAgentLoop._retrieve_many` already gets this right — it issues one
*batched* multi-query call on a persistent client. `AgenticRAGLoop` is the loop
that did not, and it is on the Assist request path.

## Approach

Replace each sequential `for … await` with a single `asyncio.gather(...,
return_exceptions=True)`, then consume the results **in the original order** so
that ordering-dependent state is unchanged:

- **AgenticRAG** — documents accumulate into `accumulated` in `novel_queries`
  order, so the stable `D1..DN` re-numbering and the recorded `search/retrieve`
  stages come out exactly as before. Per-query failures stay isolated: a raised
  exception is re-raised inside the existing per-query `try` so it is logged and
  skipped, as it was.
- **Tool evidence** — selection screening (eligibility, `rejected` callbacks,
  the `max_calls` cut-off) stays a sequential pass, because it is pure and order
  matters. Only the accepted invocations fan out. `succeeded`/`failed` callbacks
  and `T1..Tn` evidence ids are emitted in selection order.
- **MCP** — discovery fans out; registration still walks `specs` in order, so
  registry contents and log lines are unchanged.

## Deliberately not changed

- **`_run_direct_search`'s provider fan-out** (`src/internal/servers/web/app.py`).
  The providers are independent, but each leg's document numbering is threaded
  through `start_index` / `existing_count`, and `_run_browser_search` is
  monkeypatched by three test modules with that signature. Parallelising it means
  re-numbering documents after the fact — in the citation-label code path that
  has broken before (#427). The payoff does not justify it: the only setting
  where legs are both slow is `source_provider="all"`, and there the browser leg
  (~48 s) dominates SerpAPI (~12 s), so the saving is ~20% of one uncommon mode.
  Every other mode has one slow leg and one ~50 ms leg.
- **The `auto` escalation loop** (`app.py`, external-provider fallback) is a
  first-success cascade: it exists precisely so a successful SerpAPI call avoids
  paying for the browser. Running the legs together would defeat its purpose.
- **Multi-turn agent loops** (`SearchAgentLoop.run`, memory curation) are
  sequential by definition — each turn's input is the previous turn's output.
- **`warm_up_connections`** opens 20 async connections one await at a time, but
  it has no callers anywhere in the repo. Noted, not touched.

## Verification

A latency benchmark stubs each site's I/O with a fixed 50 ms sleep, so the number
measured is the scheduling shape alone:

| Site | Before | After |
|---|---|---|
| AgenticRAG round, 5 queries | 256 ms | 51 ms |
| Tool evidence, 4 tools | 254 ms | 52 ms |
| MCP discovery, 4 servers | 205 ms | 51 ms |

Each site gets a regression test asserting peak in-flight concurrency > 1. All
three were mutation-checked: forcing the call sites back to sequential turns
exactly those tests red and leaves the rest of the suite green.
