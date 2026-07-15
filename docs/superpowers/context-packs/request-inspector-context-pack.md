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

### Testing

- **Unit:** `record_stage` no-ops when inactive; `start_capture` + emits produce
  the expected snapshot; ring buffer evicts past N; endpoints return snapshot /
  404.
- **Integration:** one auto-routed request with `debug_panels` on yields a
  snapshot with all reached stages populated (raw prompt + retrieved docs
  present); with the flag off, no capture and zero added work.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
