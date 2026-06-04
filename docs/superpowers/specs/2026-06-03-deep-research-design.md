# Deep Research — Design Spec

**Date:** 2026-06-03
**Status:** Approved

## Overview

Add a "Deep Research" mode to Agentic Search that produces in-depth, structured
markdown reports by running a multi-step plan → search → synthesize flow. Users
trigger it via a toggle in the UI; the server streams progress events so they
can follow each phase in real time.

---

## Goals

- Multi-step research flow: explicit planning, iterative retrieval, structured synthesis
- Streaming progress feedback (SSE) so users see each phase as it happens
- Output is a structured markdown report (Overview / Findings / Conclusion / References) stored as a normal assistant message in the chat timeline
- Minimal surface-area change: new mode on the existing `/api/agent` endpoint, no new endpoint

## Non-goals

- Background job / async polling (no job queue needed)
- Per-sub-question retrieval budgets (shared pool across all sub-questions)
- Fine-tuning or reward shaping for the deep research loop

---

## Architecture

### New files

| File | Purpose |
|---|---|
| `src/agents/deep_research.py` | `DeepResearchLoop` — three-phase agent with async streaming |
| `tests/unit/test_deep_research.py` | Unit tests for all three phases |

### Modified files

| File | Change |
|---|---|
| `src/backend/servers/web/app.py` | Detect `mode="deep_research"`, return `StreamingResponse(text/event-stream)` |
| `web/src/types.ts` | Add `mode` to `AgentExperienceRequest`; add `DeepResearchProgressEvent` |
| `web/src/api.ts` | Add `runDeepResearch(req, onProgress, onResult)` using fetch + ReadableStream |
| `web/src/components/SearchComposer.tsx` | "Deep Research" toggle; relabels submit button to "Research" when active |
| `web/src/components/AnswerPanel.tsx` | Vertical stepper for progress; renders structured markdown report |

---

## `DeepResearchLoop`

### Config

```python
@dataclass(frozen=True)
class DeepResearchConfig:
    max_rounds: int = 5           # total retrieval rounds across all sub-questions
    max_sub_questions: int = 5    # max LLM-generated research branches
    topk: int = 5
    retrieval_url: str = "http://localhost:8000/retrieve"
```

### Result

```python
@dataclass
class DeepResearchResult:
    report: str                    # full markdown report
    citations: list[str]
    documents: list[ContextDocument]
    rounds_used: int
    sub_questions: list[str]
```

### Progress events (newline-delimited JSON, sent as SSE `data:` lines)

```
{"step": "planning",     "message": "Generating research plan…"}
{"step": "searching",    "message": "Researching: <sub-question>", "round": N}
{"step": "synthesizing", "message": "Writing report…"}
{"step": "done",         "message": ""}
```

After `done`, the server sends a final `event: result` line containing the full
`AgentExperienceResponse` JSON.

On error: `event: error\ndata: {"detail": "…"}\n\n` then close.

### Three-phase execution

#### Phase 1 — Plan

Single LLM call with a prompt asking for a JSON list of 3–5 focused
sub-questions derived from the original query. Falls back to `[original_query]`
if the LLM call fails or the response is not valid JSON.

```
Prompt: "Break the following research question into 3–5 focused sub-questions.
Return a JSON array of strings, nothing else.\n\nQuestion: {question}"
```

#### Phase 2 — Search

Round-robin dispatch across sub-questions until `max_rounds` is exhausted:

- Round 0 → sub-question 0
- Round 1 → sub-question 1
- … wraps back to sub-question 0 after the last sub-question

Each round calls `retrieve_context(sub_question, search_url, top_k)` and merges
results into a shared `accumulated: dict[str, ContextDocument]` keyed by
document `id` (deduplication). Evidence is shared across sections, so the
synthesis step can cite a document regardless of which sub-question found it.

Retrieval errors on a round are logged and skipped; the loop continues.

#### Phase 3 — Synthesize

Single LLM call with all accumulated documents as context. The prompt instructs
the model to produce a markdown report with these exact top-level sections:

```markdown
## Overview
## Findings: <sub-question 1>
## Findings: <sub-question 2>
…
## Conclusion
## References
```

The report is stored as an ordinary assistant message in `AgenticSearchStore` so
the chat timeline remains consistent with the standard and agentic_rag modes.

---

## Web Layer — SSE Streaming

`/api/agent` detects `mode="deep_research"` and returns:

```python
StreamingResponse(
    _deep_research_stream(request, db, settings, llm, hook_metadata),
    media_type="text/event-stream",
    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
)
```

The async generator `_deep_research_stream`:
1. Creates `DeepResearchLoop`, calls `run()` with an `on_progress` callback
2. Each `on_progress` call yields `event: progress\ndata: {json}\n\n`
3. After `run()` completes, persists the assistant message, yields `event: result\ndata: {json}\n\n`
4. On exception yields `event: error\ndata: {json}\n\n` and exits

Existing `mode="standard"` and `mode="agentic_rag"` code paths are untouched.

---

## Frontend

### `types.ts` additions

```typescript
export type AgentMode = "standard" | "agentic_rag" | "deep_research";

// Add to AgentExperienceRequest:
mode?: AgentMode;

export interface DeepResearchProgressEvent {
  step: "planning" | "searching" | "synthesizing" | "done";
  message: string;
  round?: number;
}
```

### `api.ts` — `runDeepResearch`

Uses `fetch` + `response.body.getReader()` to parse the SSE stream (not
`EventSource`, which only supports GET). Calls `onProgress` for each
`event: progress` line and `onResult` for `event: result`.

### `SearchComposer.tsx`

Adds a "Deep Research" toggle (off by default). When enabled:
- Submit button label changes from "Search" to "Research"
- Request is dispatched via `runDeepResearch` instead of the normal `POST /api/agent`

### `AnswerPanel.tsx`

While the stream is active, renders a vertical stepper above the answer area:

```
✓ Planning
⟳ Searching (round 2 / 5): What is FAISS?
  Synthesizing
  Done
```

Once the result arrives, the stepper collapses and the markdown report is
rendered with section headers, body text, and inline citations.

---

## Error Handling

| Failure | Behaviour |
|---|---|
| Planning LLM error / bad JSON | Fall back to `[original_query]`; research continues |
| Retrieval timeout on a round | Log warning, skip round, continue |
| Synthesis LLM error | Yield `event: error`, return HTTP 502 in stream |
| Client disconnects mid-stream | FastAPI disconnect terminates the generator cleanly |

---

## Tests (`tests/unit/test_deep_research.py`)

- **Plan parsing**: valid JSON list, malformed JSON (fallback to original query), empty list (fallback)
- **Search accumulation**: deduplication by `id` across rounds; partial retrieval failure does not abort loop
- **Synthesis**: correct section headers present; citations list populated from accumulated docs
- **Full `run()` with mocks**: mock `LLMClient` + mock `retrieve_context` — verifies all three phases fire in order; result fields populated correctly
- **Progress event sequence**: asserts `planning → searching × N → synthesizing → done` order

---

## Open Questions

None — all design decisions resolved during brainstorming.
