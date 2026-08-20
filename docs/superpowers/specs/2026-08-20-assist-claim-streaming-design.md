# Assist claim-level streaming

Date: 2026-08-20
Status: approved, not implemented

## Problem

The Assist surface emits its answer as a single SSE event after the whole run
finishes, so time-to-first-token equals time-to-completion. PR #546 fixed this
for the Chat surface by streaming model tokens, but Assist was left untouched
and the note closing that work described the remaining job as "wire
`LLM.stream()` through `LLMClient` with the same probe-and-fallback pattern."

That description is wrong, and the reason it is wrong determines this design.

Assist's chat route runs `_run_agentic_rag` (`app.py:640`) which calls
`generate_answer` (`pipeline.py:51`) with a default `AnswerGenerationRequest`.
The default `GroundedGenerationConfig.enabled` is `True` (`models.py:284`), so
the live path is `_generate_guarded_answer` (`pipeline.py:155`). On that path
**the model does not produce the answer**. It produces a JSON *answer draft*:

```json
{"claims": [{"text": "...", "evidence_ids": ["R1"]}],
 "missing_information": [],
 "abstain": false}
```

Each claim is verified against the evidence, unsupported claims are **dropped**,
and the answer the user sees is `render_verified_answer` (`safety.py:187`) — a
join of only the surviving claims, or a canonical abstention if none survive.

Streaming raw tokens from that call would emit JSON scaffolding, and would show
the user claims the verifier is about to delete. Token streaming is not merely
unimplemented here; it is the wrong unit. The correct unit is the **verified
claim**.

## The invariant

Text sent to the browser cannot be recalled. Every design decision below exists
to serve one property:

> **Append-only.** Once a claim has been emitted, nothing later in the run may
> contradict, retract, or rewrite it. The final answer is exactly the join of
> the emitted claims, in emission order.

Two mechanisms in the current code violate this.

### Hazard 1 — `abstain` is decided after the claims are read

`abstain: true` discards every claim regardless of individual support
(`safety.py:95`). In the current schema `claims` is the first property, so a
streaming reader learns the draft abstained only after it has already streamed
the entire claim list.

### Hazard 2 — the corrective retry replaces the whole draft

`_generate_guarded_answer` loops up to `1 + min(max(max_retries, 0), 1)` = 2
attempts by default, and retries whenever **any** claim is unsupported:

```python
if draft.abstain or not result.unsupported_claims:
    break
feedback = _verifier_feedback(result)
```

A partially-grounded draft is the common case, not an edge case, so the retry
fires often. Attempt 2 is generated from a corrective prompt and is an entirely
fresh draft with different claim text, invalidating anything attempt 1 streamed.

## Design

### H1 — order the draft so `abstain` precedes `claims`

Reorder both `required` and `properties` in `_ANSWER_DRAFT_JSON_SCHEMA`
(`structured_output.py:34`) to `abstain`, `missing_information`, `claims`, and
reorder the corresponding sentence in `build_structured_answer_prompt`
(`prompts.py:103`) to match. Under OpenAI strict structured output, schema
property order drives generation order; the prompt sentence carries the same
ordering for the `PROMPT_ONLY` capability path.

`parse_answer_draft` (`safety.py:47`) compares key **sets**, so it is already
order-insensitive and needs no change. Reordering is therefore backward
compatible with drafts produced by the old ordering.

The incremental reader treats a claim as unreachable until `abstain` has been
seen and is `false`. If a draft arrives with `claims` first anyway — an older
provider, or a prompt-only model that ignored the ordering — the reader yields
nothing and the run falls back to non-streaming behaviour. Ordering is an
optimisation for the streaming path, never a correctness requirement.

### H2 — commit-only-supported, and make the retry append-only

A claim is emitted **only after it verifies as supported**. Unsupported claims
are never streamed. The corrective retry is then reframed: attempt 1's supported
claims are already committed and permanent, and attempt 2 supplies replacements
only for the unsupported ones, which are appended.

```
attempt 1: [A supported] [B unsupported] [C supported]
           emit A, emit C
attempt 2: repair B
           emit B'
final answer = "A C B'"
```

**This is a deliberate semantic change and it is accepted.** Today attempt 2
replaces attempt 1 wholesale, so a streamed answer can differ from what the same
query returns unstreamed: streaming has already committed a claim that today's
retry might have rewritten. The alternative — buffering attempt 1 — preserves
exact semantics but yields no improvement in the common clean-first-draft case,
which is the case that matters. The alternative of suppressing the retry
entirely produces visibly thinner answers.

Claim order changes as a consequence: repaired claims land at the end rather
than in their original position. `render_verified_answer` joins claims with a
space and imposes no ordering contract, so this is a presentation change, not a
correctness one.

### The seam: an optional `on_claim` callback

`generate_answer` gains `on_claim: Callable[[str], None] | None = None`. When it
is `None`, the entire mechanism is inert and behaviour is byte-identical to
today, including whole-draft replace-retry. Streaming is opt-in per call.

This follows #546's discipline. `LLMClient` (`models.py:33`) is a Protocol
declaring only `complete`, so adding a required streaming method would break
every structural implementation and test double — including the non-streaming
MCP and search callers. Streaming support is instead probed with `getattr`,
exactly as `structured_output_capability` is already probed at `pipeline.py:171`:

```python
stream_fn = getattr(llm, "stream_complete", None)
```

When `on_claim` is set but the LLM has no `stream_complete`, or the incremental
parse fails, the code falls back to the existing blocking `complete` path and
emits nothing incrementally. The final answer is unaffected in every case.

### Component boundaries

**`verify_claim` (new, `safety.py`).** `verify_answer_draft` (`safety.py:117`)
already loops over claims with no cross-claim dependency: each verdict is a
function of the claim, the evidence map and the overlap threshold. Extract that
body as `verify_claim(claim, evidence_by_id, *, overlap_threshold) ->
ClaimVerdict` and have `verify_answer_draft` call it per claim. Pure refactor,
no behaviour change, pinned by the existing verification tests. The aggregate
concerns — `draft.abstain`, `confidence`, `status` — stay in
`verify_answer_draft` and remain whole-draft.

**Incremental draft reader (new module, `src/context/streaming_draft.py`).** The
one genuinely new component. Accepts text chunks and yields, in order: the
`abstain` boolean once decidable, then each complete claim object as its closing
brace arrives. Strictly advisory: on any malformed or unrecognised prefix it
stops yielding permanently and reports that it has given up. It never decides
the final answer. The authoritative parse remains whole-text
`parse_answer_draft` on the accumulated string, exactly as today.

**Streaming transport (`providers.py`).** `OpenAICompatibleLLM` — the class
Assist actually constructs (`app.py:1258`) — already has `stream()`
(`providers.py:150`), returning `Iterator[ModelResponseStream]`. But `stream()`
is not the method the pipeline probes for, and deliberately so: its signature
belongs to the raw provider API (`structured_response_format: dict`, provider
chunk objects) while `complete`'s belongs to `LLMClient`
(`structured_output: StructuredOutputRequest`). Probing `stream` directly would
push provider-shaped arguments into `pipeline.py`.

So `OpenAICompatibleLLM` gains a thin `stream_complete(messages, *,
structured_output=None) -> Iterator[str]` adapter: it translates the
`StructuredOutputRequest` into `structured_response_format`, delegates to
`stream()`, and yields plain text deltas. That is the method `pipeline.py`
probes, and it mirrors `complete`'s signature exactly — which is what makes the
`getattr` probe safe against every other `LLMClient` implementation.

The returned iterator is synchronous over blocking `requests`, so it is consumed
inside the existing `asyncio.to_thread` offload, with each claim marshalled back
to the event loop via `loop.call_soon_threadsafe`. The offload added in #547 is
preserved, not bypassed.

**Loop and route.** `AgenticRAGLoop.run` (`agentic_rag.py:323`) threads
`on_claim` into its `to_thread(generate_answer, ...)` call. `_run_agentic_rag`
(`app.py:640`) accepts and forwards it. `stream_agent` (`app.py:1886`) supplies
a callback that puts `{"type": "claim", "text": ...}` on the existing SSE queue,
alongside today's `progress` and `trace` events. The terminal `answer` and
`done` events are unchanged, so a client that ignores `claim` events behaves
exactly as it does now.

**Frontend.** `AssistPage.tsx:144` gains a `claim` branch that appends the claim
text to the in-progress assistant message; the existing `answer` branch then
reconciles the full text on completion. `types.ts:384` gains the event variant.

## Data flow

```
OpenAICompatibleLLM.stream()      sync iterator, in a worker thread
  -> incremental draft reader     yields abstain, then complete claims
  -> verify_claim                 per claim, against the evidence map
  -> on_claim(text)               supported claims only
  -> call_soon_threadsafe         back onto the event loop
  -> SSE queue                    {"type": "claim", "text": ...}
  -> AssistPage                   appended into the assistant bubble

(whole accumulated text) -> parse_answer_draft -> verify_answer_draft
  -> render_verified_answer -> the authoritative final answer
```

## Error handling

Every failure degrades to today's behaviour rather than to a broken stream.

- LLM has no `stream_complete`: fall back to blocking `complete`.
- Malformed or unexpected JSON prefix: reader stops yielding; the whole-text
  parse still runs and the final answer is unaffected.
- `claims` arrives before `abstain`: reader yields nothing, run completes
  normally.
- `abstain: true`: no claim is ever emitted, so there is nothing to retract; the
  canonical abstention is delivered by the final `answer` event.
- Stream errors mid-draft: the accumulated partial text fails `parse_answer_draft`,
  which is already handled as a parse failure and drives the existing retry.
- Timeout: `LLMTimeoutError` handling at `pipeline.py:204` is unchanged.

## Testing

The invariant is the test target, not the plumbing.

1. **Append-only property.** For a draft with supported and unsupported claims
   across both attempts, assert the final answer equals the join of emitted
   claims in emission order, and that no emitted claim is absent from the final
   answer.
2. **Abstain ordering.** A draft with `abstain: true` emits zero claims.
3. **Claims-before-abstain.** A draft in the old key order emits zero claims and
   still returns the correct final answer.
4. **`on_claim=None` is inert.** Existing `_generate_guarded_answer` tests pass
   unchanged, including whole-draft replace-retry.
5. **Fallback.** An `LLMClient` double with no `stream_complete` streams nothing
   and returns the same answer as today.
6. **Malformed prefix.** Truncated and corrupt JSON yield no claims and no
   exception.
7. **`verify_claim` refactor.** Existing verification tests pass unchanged.
8. **SSE shape.** `stream_agent` emits `claim` events before `answer`, and a
   client ignoring them sees the current event sequence.
9. **Frontend.** `AssistPage` appends claims and reconciles on `answer`.

## Out of scope

- `SearchAgentLoop`, excluded for the same reason as in #546: its XML protocol
  would leak `<plan>` and `<searches>` without incremental `<answer>` boundary
  detection.
- The MCP `ask_agentic_search` tool, which is not on the request path.
- The extractive (`llm is None`) and ungrounded (`enabled=False`) branches of
  `generate_answer`, which Assist does not reach.
- Token-level streaming within a single claim. The claim is the unit that can be
  verified, and therefore the smallest unit that can be safely shown.
