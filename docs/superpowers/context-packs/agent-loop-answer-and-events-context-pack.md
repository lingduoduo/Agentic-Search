# Generated Context Pack

# Agent Loop Answer And Events

## Sources

- [Specification: 2026-07-11-agent-loop-answer-and-events-design.md](../specs/2026-07-11-agent-loop-answer-and-events-design.md)
- [Plan: 2026-07-11-agent-loop-answer-and-events.md](../plans/2026-07-11-agent-loop-answer-and-events.md)

## Specification Context

### Non-goals / invariants

- No behavior change to the happy path (tool calls present, no caps hit).
- Token-level truncation from `response_length` is unchanged; A1 only guarantees
  the (possibly truncated) answer is recorded rather than lost.
- No new config fields.

### Tests

- A1: caps (`max_assistant_turns=1`, `response_length`) fire on the first turn →
  `final_answer` equals the generated content and `trajectory_messages` ends with
  the assistant message.
- A2: `_is_stopped` monkeypatched True → consumer receives a STOP event before the
  generator ends; `_LISTEN_TIMEOUT_SECONDS` monkeypatched to 0 → TIMEOUT event.
- A3: a `complete` that sleeps past a tiny `sufficiency_timeout_s` → returns `[]`
  without hanging; a normal response → parsed to queries.

## Implementation Plan Context

### Overview

Date: 2026-07-11
Spec: docs/superpowers/specs/2026-07-11-agent-loop-answer-and-events-design.md

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
