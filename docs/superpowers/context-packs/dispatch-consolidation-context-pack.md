# Generated Context Pack

# Dispatch Consolidation

## Sources

- [Specification: 2026-06-29-dispatch-consolidation-design.md](../specs/2026-06-29-dispatch-consolidation-design.md)
- [Plan: 2026-06-29-dispatch-consolidation.md](../plans/2026-06-29-dispatch-consolidation.md)

## Specification Context

### Goal

One place builds and runs each agent loop; one place assembles the response.

### Out of scope

- M10 `Router`-into-loop wiring (PR B); `route_query` / auto-search internals.

## Implementation Plan Context

### Overview

Spec: 2026-06-29-dispatch-consolidation-design.md
Status: shipped (consolidated in PR #347, alongside the router and the
deterministic auto-search).

**Goal:** One place builds/runs each agent loop; one place assembles the response.
`_run_auto_routed` and the explicit-mode chain in `app.py` both dispatch through
shared runners and `_finalize_response`. Behavior preserved except additive
convergences listed in the spec. TDD; commit per task.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
