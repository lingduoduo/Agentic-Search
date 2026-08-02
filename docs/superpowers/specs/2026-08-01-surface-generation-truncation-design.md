# Surfacing a truncated generation

**Date:** 2026-08-01
**Status:** Approved

## Problem

The wall-clock stop cuts long local answers mid-word. #482 made
`generation_timeout_seconds` configurable, but left the symptom untouched: the
user still sees half a sentence with no indication anything went wrong.

The information exists. `LocalServerManager` already detects the case and prints

```
Warning : generation stopped by --generation_timeout_seconds=120.0 after 298 token(s).
```

to **stdout**. Nothing carries it into the agent loop, the API response, or the
UI. A fragment is indistinguishable from a complete answer, so a user's only
recourse is to guess that the model is bad — which it is not; it ran out of time.

Diagnosing this from the outside took three wrong attempts even with the source
open, because the one signal that would have said so was on a stream nobody
reads.

## Goals

- A truncated answer is labelled as truncated, everywhere it surfaces.
- The label names the knob that fixes it.
- Managers that cannot truncate (OpenAI-compatible, test doubles) need no changes.

## Non-goals

- No automatic retry or continuation of a truncated answer.
- No change to the default timeout, or to when truncation happens.
- Only the tool surface reports it. The search and chat surfaces share the
  mechanism but are not wired up here.

## Design

### Recording, keyed by request

`generate()` already receives a `request_id`. `LocalServerManager` records
truncation into `_truncated[request_id]` at the same point it prints the warning,
and exposes `pop_truncated(request_id) -> bool`, which consumes the record.

Keying by request matters: one manager serves concurrent requests, so a flag on
the manager would attribute one request's truncation to another.

The dict is bounded by `_TRUNCATION_RECORD_LIMIT` (256, oldest evicted). The
agent loop always pops, but training scripts call `generate()` directly and never
do, so an unbounded dict would leak in exactly the path nobody watches.

### Carrying it out

`AgentLoopBase.generate_response_ids` pops after each generation and sets
`self.generation_truncated`. The lookup is `getattr(manager, "pop_truncated",
None)`, so a manager without the hook is simply not asked — no Protocol change,
no updates to the OpenAI-compatible manager or to test doubles.

`AgentLoopOutput` gains `truncated: bool = False`; `ToolAgentLoop.run` fills it.
`_run_tool_agent` puts it in `extra`, `tool_backend` returns it on
`ToolAgentMessageResponse` and in the SSE `done` event, and `ToolAgentView`
renders an advisory notice naming `AGENTIC_SEARCH_GENERATION_TIMEOUT`.

The notice is styled as advice, not an error: the answer is real, just
incomplete.

## Verification

Live model, timeout forced to 20s to make truncation deterministic:

```
Warning : generation stopped by --generation_timeout_seconds=20.0 after 86 token(s).
[tight timeout] timeout=20.0s truncated=True tokens=86
  tail: 'rucial but computational resources are limited.\n\n**Sparse Retrieval**:'
```

2821 backend tests, 167 frontend tests.

## Note on test design

The first version of this change broke the call between `generate()` and
`_generate_sync()` — a positional-argument mismatch — and **every unit test still
passed**, because they all mock the manager. Only running the real model caught
it.

`test_generate_records_truncation_through_the_real_call_path` exercises
`generate()` end to end against a stub model instead of a stub manager. Verified
by reintroducing the bug: that test fails, the mocked ones do not.

## Risks

- Only the wall-clock stop is reported. An answer cut by `max_tokens` or by the
  rollout budget is still silent; those paths would each need their own signal.
- `pop_truncated` consumes, so a second reader sees `False`. The agent loop is
  the only reader today.
