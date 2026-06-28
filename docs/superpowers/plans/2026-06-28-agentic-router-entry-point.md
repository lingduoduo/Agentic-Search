# Agentic Router Entry Point Implementation Plan

Spec: [docs/superpowers/specs/2026-06-28-agentic-router-entry-point-design.md](../specs/2026-06-28-agentic-router-entry-point-design.md)
Date: 2026-06-28

## Global Constraints

- The auto-route response contract is unchanged:
  `_run_auto_routed` returns `(answer, citations, documents, intent, extra)`
  with `intent ∈ {search, chat, tool}`.
- Never regress: every degradation path must be at least as capable as today's
  binary router; preserve all existing fallbacks.
- Reuse existing loop constructions (`SearchAgentLoop`, `AgenticRAGLoop`,
  `ToolAgentLoop`, `PlainGenerationLoop`, `_run_hybrid_search`,
  `answer_with_retrieval`) — do not re-implement them.
- TDD for the pure decision logic; rely on the existing web-experience suite for
  dispatch regression.

## File Map

- `src/internal/servers/web/intent_routing.py`
  - add `RouteStrategy` enum
  - add `_rule_based_route(query) -> RouteStrategy`
  - add `classify_route(query, llm) -> RouteStrategy`
  - add `route_query(query, *, llm, has_local_model, explicit_source) -> RouteStrategy`
- `src/internal/servers/web/app.py`
  - rewrite `_run_auto_routed` body to: decide via `route_query`, then dispatch
    capability-aware with degradation; set `extra["route"]` / `extra["route_degraded"]`.
- `tests/unit/servers/web/test_agent_router.py` (new) — decision-function tests.

## Execution Order

### Task 1: Failing unit tests for the decision logic

`tests/unit/servers/web/test_agent_router.py`:

- `test_explicit_source_routes_to_search_agent`
- `test_rule_based_tool_verbs_route_to_tool_agent`
- `test_rule_based_search_verbs_route_to_search_agent`
- `test_rule_based_conversational_routes_to_direct_llm`
- `test_rule_based_ambiguous_defaults_to_agentic_rag`
- `test_route_query_without_llm_uses_rule_based`
- `test_classify_route_parses_each_label` (stub LLM returns each label)
- `test_classify_route_defaults_to_agentic_rag_on_garbage`

Verify: `pytest tests/unit/servers/web/test_agent_router.py -q` → all fail
(symbols absent).

### Task 2: Implement the decision logic in `intent_routing.py`

Add `RouteStrategy`, keyword tables, `_rule_based_route`, `classify_route`
(mirroring `classify_is_search_flow`'s single-call + constrained-parse shape),
and `route_query` (the cascade from the spec). Keep `_rule_based_is_search` /
`_infer_intent_from_output` intact (still used elsewhere).

Verify: decision tests pass.

### Task 3: Rewire `_run_auto_routed` dispatch

Replace the Tier-2 binary branch with:

```python
has_local_model = manager is not None and tokenizer is not None
strategy = route_query(
    query, llm=llm, has_local_model=has_local_model, explicit_source=explicit_source
)
extra["route"] = strategy.value

# TOOL_AGENT
if strategy is RouteStrategy.TOOL_AGENT and has_local_model:
    -> existing ToolAgentLoop run with real tools (tool_registry + MCP);
       intent = _infer_intent_from_output(output) or "tool"
elif strategy is RouteStrategy.TOOL_AGENT:
    strategy = RouteStrategy.AGENTIC_RAG; extra["route_degraded"] = "no_local_model"

# SEARCH_AGENT
if strategy is RouteStrategy.SEARCH_AGENT and has_local_model:
    -> SearchAgentLoop run; documents from output.context.turns; intent "search"
elif strategy is RouteStrategy.SEARCH_AGENT:
    -> _run_hybrid_search pipeline (existing); intent "search"
       (extra["route_degraded"] = "no_local_model")

# AGENTIC_RAG
if strategy is RouteStrategy.AGENTIC_RAG and llm is not None:
    -> AgenticRAGLoop(AgenticRAGConfig(max_rounds=3, topk=top_k,
       retrieval_url=search_url), llm=llm).run(query, chat_history=history);
       intent "chat"
elif strategy is RouteStrategy.AGENTIC_RAG:
    -> _run_hybrid_search pipeline; intent "search"; degraded

# DIRECT_LLM
if strategy is RouteStrategy.DIRECT_LLM and llm is not None:
    -> llm.complete([... history ..., user query]); no retrieval; intent "chat"
elif strategy is RouteStrategy.DIRECT_LLM and has_local_model:
    -> PlainGenerationLoop run; intent "chat"
else:
    -> 400 "No LLM configured" (existing message)
```

Keep Tier-1 explicit-source short-circuit and all existing
hybrid→RAG→raw-search fallbacks reachable. Factor each branch into a small
helper if `_run_auto_routed` grows too long, but keep behavior identical to the
inline version.

Verify: web-experience suite passes.

### Task 4: Regression + lint

- `pytest tests/unit/servers/web/test_agent_router.py tests/unit/servers/web/test_web_experience_app.py tests/unit/servers/web/test_sse_streaming.py -q`
- `ruff check src/internal/servers/web/intent_routing.py src/internal/servers/web/app.py tests/unit/servers/web/test_agent_router.py`

## Final Acceptance Checklist

- [ ] `route_query` returns the right `RouteStrategy` across the cascade.
- [ ] Auto-route dispatches `SearchAgentLoop` (multi-turn) for search when a local model is present.
- [ ] Auto-route dispatches `AgenticRAGLoop` (decompose + HyDE) for RAG when an LLM is present.
- [ ] `DIRECT_LLM` answers without retrieval.
- [ ] `TOOL_AGENT` runs `ToolAgentLoop` with real tools.
- [ ] Every route degrades gracefully when its backend is absent; no path is worse than today.
- [ ] Response contract `(answer, citations, documents, intent, extra)` unchanged; `intent ∈ {search,chat,tool}`.
- [ ] `extra["route"]` recorded; `extra["route_degraded"]` set on degradation.
- [ ] Existing web-experience + SSE tests pass; `ruff` clean.
