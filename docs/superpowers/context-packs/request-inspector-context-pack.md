# Generated Context Pack

# Request Inspector — full-stage capture for the Dev Console — design

## Sources

- [Specification: 2026-07-05-request-inspector-design.md](../specs/2026-07-05-request-inspector-design.md)

## Specification Context

### Goal

A **developer instrument** that captures the **full raw payload** of every stage
of a single request and presents it as **one request inspector** (all stages,
top-to-bottom, for a chosen run) inside the existing Dev Console. Recent runs are
inspectable via a **rolling in-memory history**. The capture path is entirely
gated behind the existing debug flag and is a **separate channel** from the
sanitized control-flow trace — the sanitized trace is untouched.

### Verification / success criteria

- With `AGENTIC_SEARCH_DEBUG_PANELS` on, running a query and opening the Dev
  Console "Request Inspector" shows that run's intent, search, LLM, and final
  stages with full raw payloads; recent runs are selectable.
- With the flag off, `active()` is always `None`, no snapshot is stored, and no
  new endpoints do work.
- The sanitized `ControlFlowRecorder` output and existing tests are unchanged.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
