# Agent Invocation Surface

This document is the single reference for every agent mode and retrieval pipeline surface in this codebase. It supersedes any ad-hoc lists in other files.

The registry (`get_registered_agent_loop` + `resolve_agent_name`, defined in `src/agents/base.py` and exported from `src`) is the **single source of truth** for the four `AgentLoopBase` loops. `resolve_agent_name` raises `KeyError` for any name not in the registry, so callers that need non-registry paths keep dispatching those on their own existing paths. Construction is per-loop — the registry supplies only the class; instantiation kwargs differ by loop.

## Mode Table

| Mode | Category | Canonical name | CLI flag | Web mode | Scenario |
|------|----------|----------------|----------|----------|----------|
| PlainGenerationLoop | registry loop | `plain_generation` | `--mode single` | — | one-shot generation, no retrieval |
| SingleTurnAgentLoop | registry loop | `single_turn_agent` | (none) | — | one-shot RAG (reachable by canonical name) |
| SearchAgentLoop | registry loop | `search_agent` | `--mode search` | `search_agent` | multi-turn retrieval QA |
| ToolAgentLoop | registry loop | `tool_agent` | `--mode tool` | `tool_agent` | generic function calling |
| AgenticRAGLoop | non-registry loop | (none) | — | `chat_loop` | iterative RAG; different constructor (`config`, `llm`) + `run(question) -> AgenticRAGResult` |
| search_tool | retrieval pipeline | (none) | — | `search_tool` | raw search via `src.tools.search_tool` |
| hybrid_search | retrieval pipeline | (none) | — | `hybrid_search` | hybrid retrieval pipeline |
| chat_once | retrieval pipeline | (none) | — | `chat_once` | single-shot RAG answer via `answer_with_retrieval` |

## Accuracy Notes

- **CLI `--mode single` maps to `plain_generation` (PlainGenerationLoop), NOT `single_turn_agent`.** The CLI alias `single` resolves to the canonical name `plain_generation` via `resolve_agent_name`.
- `AgenticRAGLoop` is **not registered** because it does not conform to the `AgentLoopBase` contract: its constructor takes `(config, llm)` instead of the standard kwargs, and its `run` method returns `AgenticRAGResult` rather than the base return type. It is dispatched directly in the web layer.
- Non-registry rows (`AgenticRAGLoop`, `search_tool`, `hybrid_search`, `chat_once`) are not reachable through `get_registered_agent_loop`; they live on their own dispatch paths in `src/internal/servers/web/app.py`.

## Aliases

The following short aliases are resolved by `resolve_agent_name` before the registry lookup:

| Alias | Resolves to |
|-------|-------------|
| `single` (CLI) | `plain_generation` |
| `search` (CLI) | `search_agent` |
| `tool` (CLI) | `tool_agent` |

Web modes `search_agent` and `tool_agent` use their canonical names directly and do not go through the alias resolver.
