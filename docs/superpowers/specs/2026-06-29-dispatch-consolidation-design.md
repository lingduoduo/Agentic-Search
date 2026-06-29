# Agent Dispatch Consolidation Design

**Date:** 2026-06-29
**Status:** Approved for implementation planning
**Scope:** Collapse the two duplicated agent-loop dispatch sites in the web backend
(`src/internal/servers/web/app.py`) into shared loop runners plus one response
tail. No change to the public API, the routing decision (`route_query`), or
retrieval behavior.

This is PR A of a two-PR control-flow consolidation. PR B (out of scope here)
wires the M10 per-query `Router` (`src/internal/routing/`) into the search loop's
retriever selection and builds on the runners introduced here.

## Background

PR 346 (4-way agentic router) added `_run_auto_routed` for `mode=None` requests.
It builds and runs `SearchAgentLoop`, `AgenticRAGLoop`, `ToolAgentLoop`, and a
direct-LLM / `PlainGenerationLoop` path. The explicit-mode `if/elif` chain in
`_run_agent_impl` already builds and runs the same loops for `search_agent`,
`chat_loop`, and `tool_agent`. PR 345 (deterministic auto-search) is unrelated to
this change but is part of the same control-flow workstream.

The result is a **second loop-dispatch site** that duplicates the first:

| Loop | `_run_auto_routed` | explicit-mode chain |
|---|---|---|
| `SearchAgentLoop` | ~534–568 | `search_agent` ~1045–1115 |
| `AgenticRAGLoop` | ~584–598 | `chat_loop` ~1011–1043 |
| `ToolAgentLoop` | ~492–531 | `tool_agent` ~1117–1177 |
| direct LLM / `PlainGenerationLoop` | ~613–631 | (auto only) |

The `output.context.turns → documents` extraction is verbatim duplicated, each
loop's construction is duplicated, and the
`db.add_chat_message → list_chat_messages → AgentExperienceResponse` tail is
repeated about six times.

## Goal

One place builds and runs each loop; one place assembles the response. Both
entry points (`_run_auto_routed` and the explicit-mode chain) call the same
runners and the same tail.

## Chosen Approach

Extract per-loop **runners** that return the canonical tuple
`_run_auto_routed` already speaks, plus one **response tail**. Capability policy
stays at the call sites. (Approach 1 of three considered; see Alternatives.)

### Components (all in `app.py`)

- **`_run_search_agent(query, *, manager, tokenizer, search_url, top_k, on_turn, on_trace)`**
  Builds `SearchAgentLoop` via `get_registered_agent_loop(resolve_agent_name("search_agent"))`
  with `SearchAgentLoopConfig(search_url=search_url, topk=top_k, max_turns=3)`,
  runs it on `[{"role": "user", "content": query}]` with
  `sampling_params={"temperature": 0.0, "max_tokens": 256}`, extracts and dedupes
  documents from `output.context.turns`, and returns
  `(answer, citations, documents, "search", extra)` where
  `extra = {"control_flow_trace": output.control_flow_trace, "num_turns": output.num_turns}`.

- **`_run_agentic_rag(query, *, llm, search_url, top_k, history)`**
  Builds `AgenticRAGLoop(AgenticRAGConfig(max_rounds=3, topk=top_k, retrieval_url=search_url), llm=llm)`,
  runs `await rag_loop.run(query, chat_history=history)`, returns
  `(rag.answer, rag.citations, rag.context.documents, "chat", {"rounds_used": rag.rounds_used})`.

- **`_run_tool_agent(query, *, manager, tokenizer, search_url, history, resolved, on_turn, with_search_tool)`**
  Builds `ToolAgentLoop` via the registry with
  `ToolAgentLoopConfig(tool_parser_format=resolved.tool_agent_parser)`. The tool
  set depends on `with_search_tool`: explicit `tool_agent` mode prepends
  `build_search_tool(search_url=...)` to `tool_registry.list_tools()`; the
  auto-route uses `tool_registry.list_tools()` only (preserving today's two tool
  sets). Runs with `sampling_params={"temperature": 0.0, "max_tokens": 512}`,
  extracts tool calls + documents via `_extract_tool_calls_and_docs`, and returns
  `(answer, citations, documents, _infer_intent_from_output(output), extra)` where
  `answer = output.final_answer or ""` (no fallback applied inside the runner) and
  `extra = {"tool_calls": tool_calls, "num_turns": output.num_turns, "_assistant_fallback": <last assistant message>}`.
  The two callers apply opposite policies on an empty `final_answer`, so the runner
  must not pick one: the auto-route **degrades to RAG** when `answer` is empty;
  explicit `tool_agent` mode applies the **last-assistant fallback**
  (`answer = answer or extra.pop("_assistant_fallback", "")`) before finalizing.
  The `_assistant_fallback` key is popped by both callers and never reaches
  `_finalize_response`/metadata.

- **`_finalize_response(db, session_id, *, answer, citations, documents, intent, hook_metadata, extra, mode)`**
  The single response tail. Builds metadata
  `{"citations": citations, "document_ids": [d.id for d in documents], "hooks": hook_metadata, "mode": mode, "intent": intent, **extra}`,
  popping `tool_calls` out of `extra` (it is a response field, not metadata) and
  converting `control_flow_trace` raw events to `_control_flow_event_view` for both
  the response field and the persisted `model_dump` payload. Persists the
  assistant message, reads back `list_chat_messages`, and returns
  `AgentExperienceResponse` with `tool_calls` and `control_flow_trace` populated
  from `extra`.

`_run_direct_llm` is **not** extracted: it has a single caller (`_run_auto_routed`)
and is not duplicated. It stays inline.

### Call sites after consolidation

- **`_run_auto_routed`** keeps `route_query` and all capability-aware
  **degradation** (TOOL_AGENT → RAG when no local model or empty answer;
  SEARCH_AGENT → `_auto_search_pipeline` when no local model; AGENTIC_RAG →
  `_auto_search_pipeline` when no LLM; DIRECT_LLM inline). Where it runs a loop, it
  calls the matching runner instead of inline construction. It still returns the
  canonical tuple to its caller.

- **Explicit-mode chain** keeps its capability **guards** (raise `HTTPException
  400` when `manager`/`tokenizer` is missing for `search_agent` / `tool_agent`)
  before calling the runner. `search_tool` and `hybrid_search` build their tuple
  from the existing `_run_direct_search` / `_run_hybrid_search` helpers. Every
  branch — including the final `answer_with_retrieval` default — ends by calling
  `_finalize_response`.

## Behavior convergence (intentional, additive)

Because both entry points call the same runner, three behaviors converge. All are
additive (more populated response fields, never fewer):

1. **Auto-routed `search_agent`** now includes `control_flow_trace` + `num_turns`
   in its response (today only explicit `search_agent` does).
2. **Explicit `tool_agent`** now populates `tool_calls`, `documents`, and an
   inferred `intent` (today returns `intent="tool"` with empty documents and no
   tool calls).
3. `on_trace` is a runner parameter; the auto-route passes `None` and explicit
   `search_agent` passes its sink — no divergence.

These change a few existing test assertions; those tests are updated to the
converged contract. No `AgentExperienceResponse` field is removed or renamed.

## Preserved behavior

- `/api/agent` and `/api/agent/stream` request/response contracts unchanged.
- `route_query` cascade and `_run_auto_routed` degradation chain unchanged.
- Explicit-mode 400 guards for missing local model unchanged.
- The two tool sets (auto vs explicit `tool_agent`) preserved via `with_search_tool`.
- Sampling params, `max_turns`, config values, document dedup, and citation
  ordering unchanged.
- The canonical tuple `(answer, citations, documents, intent, extra)` and the
  `intent ∈ {search, chat, tool}` contract unchanged.

## Alternatives Rejected

- **One dispatcher** (explicit modes pin a `RouteStrategy` and route through
  `_run_auto_routed`): `search_tool` and `hybrid_search` do not map to the four
  strategies, and it would change explicit-mode error semantics from raise to
  degrade. More behavior risk for no extra dedup.
- **Minimal** (extract only the doc-extraction helper): leaves loop construction
  and the response tail duplicated; not meaningful consolidation.

## Testing

TDD: pin behavior with focused tests, then refactor under them.

- **New** `tests/unit/servers/web/test_loop_runners.py` — each runner's tuple
  contract and `extra` population, driven by fake loops (no model required).
- **Update** `test_web_experience_app.py`, `test_tool_trace.py`,
  `test_agent_router.py` to the converged contract.
- **Regression**: full `pytest` green; `ruff check` + `ruff format` clean.

### Verification commands

```bash
pytest tests/unit/servers/web/ -v
pytest
ruff check src/internal/servers/web/app.py tests/unit/servers/web/
ruff format --check src/internal/servers/web/app.py tests/unit/servers/web/
```

## Success Criteria

1. Each agent loop is constructed and run in exactly one place (`_run_search_agent`,
   `_run_agentic_rag`, `_run_tool_agent`).
2. The `output.context.turns → documents` extraction exists once.
3. Response assembly exists once (`_finalize_response`); every branch of
   `_run_agent_impl` ends there.
4. `_run_auto_routed` and the explicit-mode chain dispatch through the runners.
5. Public API, streaming, routing, and retrieval behavior unchanged; converged
   fields are additive only.
6. Focused and full suites pass; lint/format clean.

## Out of Scope

- M10 `Router`-into-loop wiring (PR B).
- Any change to `route_query` / `classify_route` / `_rule_based_route`.
- Changes to `SearchAgentLoop`, `AgenticRAGLoop`, `ToolAgentLoop` internals or the
  `auto_search_on_deadend` behavior from PR 345.
- New agent modes, endpoints, or response fields.
```
