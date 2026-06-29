# Agentic Router Entry Point — Design

Status: shipped (consolidated in PR #347, alongside the deterministic auto-search
and the dispatch consolidation). This doc covers the **router** piece.
Date: 2026-06-28

## Goal

Make the no-mode (`mode=None`) entry point a real **strategy router**: pick *how*
to answer, then dispatch the matching agent loop — instead of the old 2-way
search-vs-chat branch that always ran a one-shot retrieval.

| Strategy | Loop | Behavior |
|---|---|---|
| `direct_llm` | `llm.complete` / `PlainGenerationLoop` | parametric answer, no retrieval |
| `agentic_rag` | `AgenticRAGLoop` | query decompose + HyDE + grounded synthesis |
| `search_agent` | `SearchAgentLoop` | multi-turn search until evidence suffices |
| `tool_agent` | `ToolAgentLoop` | OpenAPI / MCP function calling |

## Problem (before)

The auto path was binary: the "search" route ran a one-shot
`_run_hybrid_search`, the "chat" route ran plain `answer_with_retrieval` (never
`AgenticRAGLoop`'s decompose + HyDE), there was no direct-LLM route, and tool use
was only implicit.

## Approach

1. **`route_query`** (`intent_routing.py`) returns a `RouteStrategy` via a cascade:
   explicit non-`auto` source → `SEARCH_AGENT`; else an **LLM 4-way classifier**
   (`classify_route`) when an LLM is available; else the **rule-based**
   `_rule_based_route` (tool verbs → `TOOL_AGENT`; search verbs → `SEARCH_AGENT`;
   conversational → `DIRECT_LLM`; default → `AGENTIC_RAG`, since grounded is safest).
2. **Capability-aware dispatch** in `_run_auto_routed`, degrading so no path is
   worse than before:

   | Route | Primary (requirement) | Degrades to |
   |---|---|---|
   | `TOOL_AGENT` | `ToolAgentLoop` (local model) | `AGENTIC_RAG` |
   | `SEARCH_AGENT` | `SearchAgentLoop` (local model) | hybrid-search pipeline |
   | `AGENTIC_RAG` | `AgenticRAGLoop` (llm) | hybrid-search pipeline |
   | `DIRECT_LLM` | `llm.complete` (llm) | `PlainGenerationLoop` → else 400 |

   Response contract unchanged: `(answer, citations, documents, intent, extra)`,
   `intent ∈ {search, chat, tool}`. `extra["route"]` records the strategy;
   `extra["route_degraded"]` records any degradation.

## Relationship to the rest of #347

The actual loop construction + dispatch was subsequently unified: both
`_run_auto_routed` and the explicit-mode chain now run loops through shared
runners — see the dispatch-consolidation design
(`2026-06-29-dispatch-consolidation-design.md`). `route_query` is unchanged by
that work; it remains the decision function for the auto path.

## Out of scope

- Wiring the M10 retriever-backend router (`src/internal/routing/`) inside the
  chosen loop's retrieval — a separate follow-up (PR B). M10 routes a query to a
  *retriever backend*; this routes to an *agent strategy*.

## Tests

- `tests/unit/servers/web/test_agent_router.py` — pure decision-logic tests for
  the cascade, rule-based routing, and the classifier (incl. garbage → default).
- Auto-route regression in `test_web_experience_app.py` (dispatch contract).
