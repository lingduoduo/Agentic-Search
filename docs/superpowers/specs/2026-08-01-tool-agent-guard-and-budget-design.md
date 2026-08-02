# Tool-agent recursion guard and answer budget

**Date:** 2026-08-01
**Status:** Approved

## Problem

Two loose ends left open by #479 and #480.

### 1. The recursion guard was a name match in the wrong layer

`_SHADOWED_TOOL_NAMES` in the web runner filtered tools by name. Two of those
names describe properties of the *tools*, not of the runner:

- `rag_routing_tool` generates a whole answer rather than returning evidence.
- `ask_agentic_search` runs an agent, so offering it to the agent lets the agent
  call itself. **It arrives from a remote MCP server**, which can rename it
  without touching this repo — and a rename silently disables the guard.

### 2. Answers were cut mid-word

Reported as truncation at `max_tokens: 512`. Investigation found that diagnosis
incomplete: there are **three** caps, and the one that actually bites is the one
nobody configured.

| Cap | Bounds | Was |
| --- | --- | --- |
| `max_tokens` | one generation | 512 |
| `response_length` | the whole rollout, and truncates each response | 512 |
| `generation_timeout_seconds` | wall-clock per generation | 120s, hardcoded |

`response_length` is not an answer budget. Tool results are fed back in and
counted against it (they enter `response_mask` as zeros), so raising `max_tokens`
and `response_length` together to the same value made things *worse*: three
corpus searches returning 15 documents exhausted the budget before the model
wrote anything, and the run ended on the tool-calling turn. Its "answer" was the
leftover markup:

```
tool_calls: [('search', 'completed'), ('search', 'completed'), ('search', 'completed')]
answer tokens: 11
tail: '<tool_call>\n\n</tool_call><tool_call>\n\n</tool_call>'
```

That exposed a second defect: `JSONToolParser` removes the JSON object from a
Hermes-style `<tool_call>` block but leaves the tags, so the scaffolding reads as
the answer.

With both fixed, the answer still stopped mid-word — and the real cause finally
surfaced:

```
Warning : generation stopped by --generation_timeout_seconds=120.0 after 298 token(s).
```

MPS runs a 1.5B model at a few tokens per second, so a 120-second wall clock cuts
a long answer around 300–400 tokens regardless of any token budget. The manager
already detects this and prints a warning; it goes to stdout, so the API caller
and the UI see only a half sentence.

## Goals

- "May an agent call this tool?" is a property set at registration, not a name
  matched downstream.
- A tool renamed on a remote MCP server cannot silently disable its own guard.
- Long answers are not cut by a budget the operator cannot see or change.

## Non-goals

- No change to `ToolAgentLoop`'s rollout accounting. `response_mask` counting
  tool responses is the VERL convention the training path depends on; this sizes
  the budget correctly rather than redefining it.
- Truncation is not yet surfaced to the API response. The warning still only
  reaches stdout — see Risks.
- No change to the default timeout. It becomes configurable, not longer.

## Design

### Agent-callable as a registration property

`ToolEntry` gains `agent_callable: bool = True`; `ToolRegistry.register` takes it
and `agent_tools()` filters on it. The decision is made where the knowledge
lives:

- `seed_tools` marks `rag_routing_tool` via `NOT_AGENT_CALLABLE`.
- `register_mcp_tools` marks per-server names from `McpServerSpec.agent_exclude`,
  defaulting to `DEFAULT_AGENT_EXCLUDE` and overridable with
  `AGENTIC_SEARCH_MCP_AGENT_EXCLUDE`.

The runner drops its name list entirely and calls `tool_registry.agent_tools()`.
Excluded tools stay registered, listed and directly invocable; only agent loops
are denied them. `agent_callable` is exposed in the tool summaries so
`/admin/tools` and the Dev Console can show it.

The MCP name is still a name — the protocol exposes no "this runs an agent"
signal — but it is now per-server configuration next to the server it describes,
not a literal in the web runner.

### Budget

`tool_agent_max_tokens` (env `TOOL_AGENT_MAX_TOKENS`, default 1024) caps one
generation. `response_length` is that times `_ROLLOUT_BUDGET_MULTIPLIER` (4), so
tool traffic cannot starve the answer.

`generation_timeout_seconds` becomes `AGENTIC_SEARCH_GENERATION_TIMEOUT`
(default unchanged at 120.0, `0` disables) and is passed to the local server
manager, which previously hardcoded it.

### Parser

`JSONToolParser` strips `</?tool_call>` from the content when it extracted calls,
so leftover scaffolding never reads as an answer. Only when calls were found —
prose that merely mentions the tag is untouched.

## Verification

Live model plus retrieval, question chosen to force a long answer:

| | Before | After |
| --- | --- | --- |
| answer | 11 tokens of `<tool_call>` markup | 298 tokens of prose |
| generations | 1 (loop bailed) | 2 (tools, then answer) |
| documents | 15 | 15 |

`python3 -m pytest` — 2808 passed.

## Risks

- **Truncation is still silent to the caller.** With the default 120s the answer
  is cut around 300 tokens on this hardware and the user sees a half sentence
  with no explanation. Operators can raise the timeout now, but surfacing
  "answer truncated" in the response is the real fix and is not in this change.
- `_ROLLOUT_BUDGET_MULTIPLIER = 4` is a heuristic. It is sized so several
  `max_tool_response_length` (2048-char) responses fit alongside a full answer;
  a workload with more parallel tools could still exhaust it.
