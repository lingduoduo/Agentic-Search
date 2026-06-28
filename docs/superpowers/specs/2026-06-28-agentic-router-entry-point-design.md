# Agentic Router Entry Point Design

Status: accepted
Date: 2026-06-28

## Goal

Make the no-mode (`mode=None`) entry point a true **strategy router** that picks
*how* to answer a query and dispatches to the matching agent loop, rather than a
2-way search-vs-chat branch that always runs a one-shot retrieval. The four
strategies, as discussed across the routing specs:

| Strategy | Loop | Behavior |
|---|---|---|
| `direct_llm` | `PlainGenerationLoop` / `llm.complete` | answer from parametric knowledge, no retrieval |
| `agentic_rag` | `AgenticRAGLoop` | query decomposition + HyDE + multi-round grounded synthesis |
| `search_agent` | `SearchAgentLoop` | multi-turn search until evidence is sufficient |
| `tool_agent` | `ToolAgentLoop` | OpenAPI / MCP function calling |

## Background — what exists today

The intent-routing design (`docs/superpowers/plans/2026-06-15-intent-routed-search-chat.md`)
introduced `_run_auto_routed` in `src/internal/servers/web/app.py:300` with a
three-tier cascade:

1. **Tier 1** — `ToolAgentLoop` as a universal router with
   `search_routing_tool` + `rag_routing_tool` as callable tools; intent is
   inferred from the first tool the model calls
   (`_infer_intent_from_output`, `intent_routing.py:44`).
2. **Tier 2** — LLM binary classifier `classify_is_search_flow`
   (`secondary_llm_flows/search_flow_classification.py:22`).
3. **Tier 3** — rule-based `_rule_based_is_search` (`intent_routing.py:26`).

Gaps relative to the goal:

- The **search route** runs a *one-shot* `_run_hybrid_search`/`search_routing_tool`,
  not the multi-turn `SearchAgentLoop`.
- The **chat route** calls plain `answer_with_retrieval`, not `AgenticRAGLoop`
  (so decomposition + HyDE never fire on the auto path).
- There is **no `direct_llm`** route — every non-search query retrieves.
- The decision is **binary** (search vs chat); tool is only implicit via Tier 1.

## Chosen Approach

Introduce an explicit strategy enum and a single decision function, then rewire
`_run_auto_routed` to dispatch capability-aware.

### 1. `RouteStrategy` + `route_query` (in `intent_routing.py`)

```python
class RouteStrategy(str, Enum):
    DIRECT_LLM = "direct_llm"
    AGENTIC_RAG = "agentic_rag"
    SEARCH_AGENT = "search_agent"
    TOOL_AGENT = "tool_agent"

def route_query(
    query: str,
    *,
    llm,
    has_local_model: bool,
    explicit_source: bool,
) -> RouteStrategy: ...
```

Decision cascade:

1. `explicit_source` (user picked a non-`auto` source provider) →
   `SEARCH_AGENT` (an explicit search command).
2. `llm` available → LLM 4-way classifier `classify_route(query, llm)`
   returning one of the four labels; on error → `_rule_based_route`.
3. no `llm` → `_rule_based_route(query)`.

`_rule_based_route` (pure, deterministic, unit-tested):

- imperative tool verbs (`send`, `create`, `open a ticket`, `call the api`,
  `run`, `schedule`) → `TOOL_AGENT`
- search verbs (`find`, `list`, `look up`, `latest`, `search for`, `who/when/
  where` short factual) → `SEARCH_AGENT`
- conversational / no-retrieval (`write`, `translate`, `rephrase`, `hello`,
  `explain this code`) → `DIRECT_LLM`
- everything else (default, grounded is safest) → `AGENTIC_RAG`

`classify_route` mirrors `classify_is_search_flow`: one LLM call, a constrained
prompt that returns exactly one label, defaulting to `AGENTIC_RAG` on an
unexpected/empty response.

### 2. Capability-aware dispatch in `_run_auto_routed`

Loops have different runtime requirements:

- `SearchAgentLoop`, `ToolAgentLoop`, `PlainGenerationLoop` need a **local
  model** (`manager` + `tokenizer`).
- `AgenticRAGLoop`, `answer_with_retrieval` need the **LLM client** (`llm`).

Dispatch with graceful degradation (never worse than today):

| Route | Primary | Degrades to (when requirement missing) |
|---|---|---|
| `TOOL_AGENT` | `ToolAgentLoop` (local model) | `AGENTIC_RAG` |
| `SEARCH_AGENT` | `SearchAgentLoop` (local model) | `_run_hybrid_search` pipeline |
| `AGENTIC_RAG` | `AgenticRAGLoop` (llm) | `_run_hybrid_search` pipeline |
| `DIRECT_LLM` | `llm.complete` (llm) | `PlainGenerationLoop` (local model) → else 400 |

The returned `intent` (frontend contract, unchanged: `search` / `chat` / `tool`)
maps: `SEARCH_AGENT→search`, `TOOL_AGENT→tool`, `AGENTIC_RAG`/`DIRECT_LLM→chat`.
`extra["route"]` records the chosen `RouteStrategy` for observability, and
`extra["route_degraded"]` is set when a degradation fired.

All existing fallback safety nets (hybrid-search failure → RAG → raw search,
`No LLM configured` 400) are preserved.

### 3. Tool-route execution detail

`TOOL_AGENT` keeps the existing Tier-1 mechanism: build the real registered
tools (`tool_registry.list_tools()` + MCP/OpenAPI tools) and run `ToolAgentLoop`.
The `search_routing_tool`/`rag_routing_tool` markers are no longer needed for the
decision because `route_query` decides up front; the loop is given the *actual*
tools to call. Search and RAG become full `SearchAgentLoop` / `AgenticRAGLoop`
runs instead of the inline one-shot routing tools.

## Alternatives Rejected

### Keep ToolAgentLoop-as-router and re-dispatch from its tool choice
The Tier-1 loop already *executes* `search_routing_tool` inline (a one-shot
retrieval). Re-dispatching a full `SearchAgentLoop`/`AgenticRAGLoop` afterward
would retrieve twice. A standalone `route_query` decides once, then dispatches
once.

### Wire the M10 retrieval router (`src/internal/routing/`) here
M10 routes a query to a *retriever backend* (SQL/graph/API/hybrid) and is
attached to `RetrievalService`. That is an orthogonal, lower layer. This spec
routes to an *agent strategy*; M10 integration inside the chosen loop's
retrieval is a separate follow-up.

## Testing

### Unit (`tests/unit/servers/web/test_agent_router.py`, new)

`route_query` / `_rule_based_route` are pure and directly tested:
- explicit source → `SEARCH_AGENT`
- tool verbs → `TOOL_AGENT`
- search verbs → `SEARCH_AGENT`
- conversational → `DIRECT_LLM`
- ambiguous default → `AGENTIC_RAG`
- `llm=None` path uses the rule-based route
- `classify_route` parses each label and defaults to `AGENTIC_RAG` on garbage
  (with a stub LLM)

### Integration / regression
- Existing `tests/unit/servers/web/test_web_experience_app.py` auto-route tests
  must still pass (dispatch contract unchanged: returns
  `(answer, citations, documents, intent, extra)`).

## Files Touched

- `src/internal/servers/web/intent_routing.py` — add `RouteStrategy`,
  `route_query`, `_rule_based_route`, `classify_route`.
- `src/internal/servers/web/app.py` — rewire `_run_auto_routed` to 4-way
  capability-aware dispatch (reusing existing loop constructions).
- `tests/unit/servers/web/test_agent_router.py` — new unit tests.

## Out of Scope

- M10 retriever-backend routing inside loops.
- Changing the explicit-mode dispatch paths (`search_agent`, `chat_loop`, …);
  they remain the single-source-of-truth registry/pipeline dispatch.
