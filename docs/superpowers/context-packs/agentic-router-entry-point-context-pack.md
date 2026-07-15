# Generated Context Pack

# Agentic Router Entry Point

## Sources

- [Specification: 2026-06-28-agentic-router-entry-point-design.md](../specs/2026-06-28-agentic-router-entry-point-design.md)
- [Plan: 2026-06-28-agentic-router-entry-point.md](../plans/2026-06-28-agentic-router-entry-point.md)

## Specification Context

### Goal

Make the no-mode (`mode=None`) entry point a real **strategy router**: pick *how*
to answer, then dispatch the matching agent loop — instead of the old 2-way
search-vs-chat branch that always ran a one-shot retrieval.

| Strategy | Loop | Behavior |
|---|---|---|
| `direct_llm` | `llm.complete` / `PlainGenerationLoop` | parametric answer, no retrieval |
| `agentic_rag` | `AgenticRAGLoop` | query decompose + HyDE + grounded synthesis |
| `search_agent` | `SearchAgentLoop` | multi-turn search until evidence suffices |
| `tool_agent` | `ToolAgentLoop` | OpenAPI / MCP function calling |

### Out of scope

- Wiring the M10 retriever-backend router (`src/internal/routing/`) inside the
  chosen loop's retrieval — a separate follow-up (PR B). M10 routes a query to a
  *retriever backend*; this routes to an *agent strategy*.

## Implementation Plan Context

### Overview

Spec: 2026-06-28-agentic-router-entry-point-design.md
Status: shipped (consolidated in PR #347).

**Goal:** `mode=None` picks a `RouteStrategy` via `route_query`, then dispatches
capability-aware. Response contract `(answer, citations, documents, intent, extra)`
unchanged; no degradation path worse than the old binary router.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
