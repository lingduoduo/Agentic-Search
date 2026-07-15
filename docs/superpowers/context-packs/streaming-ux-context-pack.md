# Generated Context Pack

# Streaming UX Design Spec

## Sources

- [Specification: 2026-06-16-streaming-ux-design.md](../archive/specs/2026-06-16-streaming-ux-design.md)

## Specification Context

### 2. Architecture

The agent loop receives an `on_turn` async callback. It calls it after each completed turn. The stream endpoint converts each call into an SSE `progress` event. This is the only coupling point — the rest of the agent loop is unchanged.

---

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
