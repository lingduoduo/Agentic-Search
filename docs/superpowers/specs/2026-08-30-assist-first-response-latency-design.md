# Assist first-response latency — design

## Problem

A real-time conversational budget looks roughly like this:

| Stage | Budget |
|---|---|
| Context / retrieval | 50–100 ms |
| LLM time-to-first-token | 150–300 ms |
| TTS first audio chunk | 100–200 ms |
| Time to first response | ~400–800 ms |

Measuring this stack against that budget found the latency is not where the
budget assumes it is.

**Retrieval is not the problem.** Against a live demo server on BEIR scifact
(5,183 documents, TF-IDF backend):

```
retrieve_context (fresh session per call)   p50=  4.5ms  p95=  5.0ms
persistent client, 1 query                  p50=  3.8ms  p95=  4.2ms
persistent client, 5 queries batched        p50=  6.3ms  p95=  8.5ms
AgenticRAG round: 5 queries gathered        p50= 13.5ms  p95= 14.8ms
```

Every figure is inside the 50–100 ms budget by 4–10×.

**The event loop freezes before retrieval starts.** `QueryEnhancer.enhance()`
issues three sequential blocking LLM calls — `decompose`, `hyde`, `step_back` —
and `AgenticRAGLoop.run` calls it inline, with no `asyncio.to_thread` offload.
`llm.complete` is a blocking `requests.Session.post`. Measured with a 5 ms
heartbeat coroutine and an LLM stubbed at an optimistic 100 ms per call:

```
LLM calls made by one run   : 3  (decompose, hyde, step_back)
longest single stall        : 320 ms
total time loop was blocked : 315 ms
```

For those 320 ms nothing else on the process runs — every other in-flight
session stalls behind one user's query preparation. This is the same defect
#547 fixed for `generate_answer`; the comment above that offload in
`agentic_rag.py` describes it exactly ("awaiting it inline froze the whole event
loop ... the worst one to leave unprotected"). `enhance()` was missed. At real
hosted-LLM latency (300–800 ms per call) the freeze is 0.9–2.4 s per request.

**Nothing measures first response.** `EvalTimings.llm_first_token_ms` exists and
is printed by the local eval provider, but `evals/eval.py` hardcodes it to
`None` — and that field belongs to the Onyx-heritage `src/internal/chat/` stack,
not the Assist path. `add_latency_logging_middleware` is written but has zero
call sites. Time-to-first-token and time-to-first-claim are both unmeasured on
the path users actually hit.

**The first request per process pays ~1.4–1.6 s of lazy imports** (`.safety`,
`request_capture`), attributed to whichever user arrives first.

## Scope

Four changes. Text-to-speech is out of scope: there is no audio subsystem in this
repository at all, so that budget row is a new subsystem rather than an
optimization.

### 1. Un-block and parallelize query enhancement

Add `QueryEnhancer.enhance_async()` beside the existing sync `enhance()`:
`asyncio.gather` over three `asyncio.to_thread` calls, one per strategy.
`AgenticRAGLoop.run` awaits it.

The sync `enhance()` stays exactly as it is — `query_transform.py:98` builds a
`QueryEnhancer` from synchronous code and must keep working. `enhance_async` is
additive, not a replacement.

Three concurrent calls are safe: the provider's session is an
`HTTPAdapter(pool_connections=4, pool_maxsize=16)`, and this changes no call
count — the same three requests simply overlap rather than queue.

Each strategy already degrades to a fallback on any exception, so gathering
without `return_exceptions` is correct: nothing propagates.

### 2. Instrument first response on the Assist stream

Two measurements, both on the path the benchmark above exercises. Both are taken
inside `_generate_guarded_answer`, which already drives the streaming loop
itself — so neither needs a change to `providers.py`, and neither adds per-delta
work:

- **Time to first token** — elapsed from immediately before iterating
  `stream_fn(...)` to its first `delta`. That delta is the first token out of the
  provider, so this is true TTFT.
- **Time to first claim** — elapsed from the same origin to the first `_commit()`,
  i.e. the first content a user can see.

Both are `None` when the provider does not stream (the `llm.complete` branch),
and time-to-first-claim is `None` when no claim was ever committed.

**Plumbing.** `_generate_guarded_answer` already returns an 8-tuple. It gains one
element — a `GenerationTimings` dataclass — rather than two more loose floats, so
the call site grows by one name and the timings stay typed and grouped.
`generate_answer` puts it on `AnswerGenerationResult`; `AgenticRAGLoop.run`
reads it off the result and adds both fields to the `answer_generator`
control-flow trace event it already emits. From there they ride the SSE stream
the frontend already consumes and appear in the Dev Console, with no new
endpoint and no new transport.

`EvalTimings.llm_first_token_ms` is left alone. Populating it would instrument a
stack that is not the request path, and the field's only reader is an offline
provider.

### 3. Warm lazy imports at startup

A `_warm_imports()` call in the web app's `lifespan` that imports
`src.context.safety` and `src.internal.servers.web.request_capture`. Both are
deferred purely for laziness — `request_capture`'s deferral in the agent loops is
to break an import cycle *from those modules*, which importing from the web app's
own startup does not reintroduce, because by then the web package is fully
loaded.

Model loading stays out: it is already conditional on
`search_agent_server_url` and is orders of magnitude larger than an import.

### 4. Batch the AgenticRAG round's queries

Add `retrieve_contexts(queries, ...)` to `src/context/pipeline.py`: one
`SearchClient.retrieve(queries)` call over a single session, returning one
`SearchContextBundle` per query in input order — identical to what N separate
`retrieve_context` calls produce, because `build_context_bundle` is applied
per query with the same `top_k`.

Retrieval-provider only. AgenticRAG never uses another provider, and the
multi-query request shape is specific to `/retrieve`.

This supersedes the `asyncio.gather` of single-query calls added in #560 at that
one call site. The gather is correct and stays correct; batching just replaces N
round trips and N sessions with one. Measured 13.5 ms → ~6.3 ms.

Access filters are enforced exactly as the single-query path enforces them:
filters are forwarded in the payload and re-applied to each returned row, so the
invariant from #487–#492 (never forward without enforcing) holds per query.

## Testing

Each item gets a test that fails without it:

1. A heartbeat coroutine ticking every 5 ms during `AgenticRAGLoop.run` with a
   blocking stub LLM — asserts no stall above a threshold, and that the three
   strategy calls overlap.
2. An assertion that the `answer_generator` trace event carries both
   `llm_first_token_ms` and `time_to_first_claim_ms`.
3. An assertion that the warmed modules are in `sys.modules` after lifespan
   startup.
4. An assertion that one AgenticRAG round issues exactly one retrieval request
   carrying all of the round's queries.

Every test is mutation-checked: reverting the change must turn its test red and
leave the rest of the suite green.

## Risks

- Item 4 rewrites a function changed by #560. If batching interacts badly with
  access filters, the gather is kept and the batching is dropped — the other
  three items are independent of it.
- Item 2 measures inside the streaming loop. It must cost one `perf_counter`
  before the loop plus one `is None` check per delta — never a timestamp or any
  other work per delta.
- Item 2 grows `_generate_guarded_answer`'s return tuple from 8 elements to 9.
  That function has a single call site, so the change is contained, but the
  tuple is already large enough that the next addition should convert it to a
  dataclass rather than growing it again.
