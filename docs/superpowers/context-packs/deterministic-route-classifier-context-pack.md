# Generated Context Pack

# Deterministic Route Classifier

## Sources

- [Specification: 2026-07-05-deterministic-route-classifier-design.md](../specs/2026-07-05-deterministic-route-classifier-design.md)
- [Plan: 2026-07-05-deterministic-route-classifier.md](../plans/2026-07-05-deterministic-route-classifier.md)

## Specification Context

### Scope / non-goals

- **In scope:** deterministic *routing* only.
- **Out of scope:** pinning answer-generation temperature (answer wording may
  still vary slightly); forcing a single fixed source; cross-restart model-load
  flapping. These were considered and explicitly deferred per user decision
  ("deterministic routing").

### Verification

- Unit: `complete()` forwards `temperature`; `classify_route` passes
  `temperature=0.0`.
- Regression: web-server, routing, LLM-provider, and execution-fallback suites
  stay green.

## Implementation Plan Context

### Overview

Spec: `docs/superpowers/specs/2026-07-05-deterministic-route-classifier-design.md`

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
