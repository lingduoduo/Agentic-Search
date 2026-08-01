# Tool-agent tool selection

**Date:** 2026-08-01
**Status:** Approved

## Problem

On `/tools`, the tool agent does not call tools. Reproduced against the live
backend with `Qwen/Qwen2.5-1.5B-Instruct`:

| Request | Result |
| --- | --- |
| `What is FAISS?` | `tool_calls: []`, `num_turns: 2` — answered from parametric memory, answer terminated by a literal `<\|im_end\|>` |
| `Use the search tool to find documents about FAISS in the corpus.` | `search` called with `{"query": "FAISS"}`, completed in 1064 ms, grounded answer |

Parsing, dispatch, execution, the SSE stream and the UI all work. The model
simply does not choose to call a tool unless the user names one.

Two causes, confirmed by experiment:

1. **No steering.** `_run_tool_agent` builds the conversation as history plus the
   user message with no system prompt at all, so nothing tells the model that
   tools exist to be preferred over memory.
2. **Four overlapping tools.** The registry is seeded with `search`,
   `web_search`, `search_routing_tool` and `rag_routing_tool`. Three are corpus
   retrieval; `search` and `search_routing_tool` take the same argument and hit
   the same corpus, and `rag_routing_tool` ("Answer a question using
   retrieval-augmented generation") answers the whole question rather than
   returning evidence. Its description is the likely source of the reported reply
   *"a question that you would like answered using RAG?"*.

Adding a system prompt alone was tested and was **not** sufficient: with an
explicit "you MUST call `search` first, do not answer from memory" system
message, the 1.5B model still emitted no tool call while all four tools were
present. The ambiguous tool set has to go too.

A third, independent defect surfaced during the investigation: `ToolParser._decode`
calls `tokenizer.decode(response_ids)` without `skip_special_tokens`, so the
template's EOS marker lands in the user-visible answer.

## Goals

- The tool agent sees exactly one corpus-search tool.
- No tool in its set generates an answer instead of returning evidence.
- The model is told to prefer tools over memory.
- Special tokens never reach the user-visible answer.
- Source cards and intent classification keep working.

## Non-goals

- No change to the registry itself. `/admin/tools`, the Dev Console and MCP keep
  seeing every seeded tool; only what the *agent* is offered narrows.
- No forced first-turn tool call, and no retry-if-no-tool-call loop.
- No model change. A tool-tuned model behind `SEARCH_AGENT_SERVER_URL` remains
  the larger win and is out of scope here.
- The 512-token answer cap stays as is; truncated long answers are a separate issue.

## Design

### Tool set

`_run_tool_agent` filters two names out of the registry listing:

- `search` — duplicate of `search_routing_tool`: same corpus, same single
  `query` argument.
- `rag_routing_tool` — generates an answer; the RAG path already exists as its
  own route, so exposing it here gives the model a way to skip its own job.

Everything else the registry holds is passed through untouched, so tools a user
registers via OpenAPI or MCP still reach the agent.

`search_routing_tool` is kept as the surviving corpus search rather than
`search` because it returns a JSON list of `{title, content, url}`, which is what
`_extract_tool_calls_and_docs` turns into source cards, and what
`_infer_intent_from_output` maps to the `search` intent. Keeping `search`
instead would have required re-parsing `format_search_pages` text back into
documents.

When `with_search_tool` is set, the corpus tool is built per request via
`build_search_routing_tool(search_url=...)` so it binds to the caller's
retrieval URL instead of the one the registry was seeded with; the registry's
copy is then dropped to avoid a duplicate.

### System prompt

`TOOL_AGENT_SYSTEM_PROMPT` is prepended as a system message, but only when
neither the history nor the caller already supplies one — a caller-provided
system prompt wins.

### Special tokens

`ToolParser._decode` gains `skip_special_tokens`, defaulting to `True`.
`Llama3ToolParser` opts out with `False`: `<|python_tag|>` and `<|eom_id|>` are
genuine special tokens in the Llama 3 tokenizer and are the markers that parser
matches on, so stripping them would erase the tool calls it exists to find.
Qwen's `<tool_call>` tags are added-but-not-special tokens and survive the strip
(verified against the tokenizer), so `HermesToolParser` is unaffected.

## Verification

End-to-end against the real model and a live retrieval server, on the query that
previously produced no tool call:

```
===== What is FAISS? =====
tool_calls: [('search_routing_tool', {'query': 'FAISS definition'}, 'completed')]
intent: search | docs: 3 | turns: 4
special token leak: False
```

## Risks

- The steer is a prompt, not a constraint. A small model may still answer from
  memory on some questions; this narrows the failure, it does not close it.
- Dropping `search` from the agent's view changes which tool name appears in
  traces for tool-mode requests. `_infer_intent_from_output` already maps
  `search_routing_tool` to the `search` intent, so classification improves
  rather than breaks.
