# Tool-Agent Tool Selection Implementation Plan

**Goal:** Make the `/tools` surface actually call tools by narrowing what the tool agent is offered to one unambiguous corpus-search tool and steering it with a system prompt, and stop special tokens leaking into answers.

**Architecture:** Two small changes in `tool_agent_runner.py` (tool filtering + system prompt) and one in `parsers.py` (`skip_special_tokens`). The registry is untouched, so `/admin/tools`, the Dev Console and MCP still see every seeded tool.

**Tech Stack:** Python 3.12, FastAPI, transformers, pytest.

**Spec:** `docs/superpowers/specs/2026-08-01-tool-agent-tool-selection-design.md`

## Global Constraints

- Work on branch `fix/tool-agent-tool-selection`. Never commit to `main`.
- No changes to `ToolRegistry`, `knowledge_base.seed_tools`, or the seeded tool set.
- No changes to `ToolAgentLoop` control flow.
- `python3 -m pytest` must pass before commit.
- Run `ruff check . --fix && ruff format .` before committing.

## Tasks

- [x] **Task 1 — Narrow the agent's tool set.**
      Filter `search` and `rag_routing_tool` out of the registry listing in
      `_run_tool_agent`; when `with_search_tool` is set, build the corpus tool per
      request with `build_search_routing_tool(search_url=...)` and drop the
      registry's duplicate.
      *Verify:* tests assert the agent sees exactly one `search_routing_tool`, no
      `search`, no `rag_routing_tool`, and that `web_search` plus externally
      registered tools survive.

- [x] **Task 2 — Steer with a system prompt.**
      Add `TOOL_AGENT_SYSTEM_PROMPT` and prepend it as a system message only when
      no system message is already present.
      *Verify:* tests assert the prompt is prepended for empty history and that a
      caller-supplied system message is left alone.

- [x] **Task 3 — Stop the special-token leak.**
      Give `ToolParser._decode` a `skip_special_tokens` parameter defaulting to
      `True`; `Llama3ToolParser` passes `False` because its call markers are
      special tokens.
      *Verify:* tests assert hermes/json decode with `skip_special_tokens=True`,
      and that llama3 decodes with `False` and still extracts its calls.

- [x] **Task 4 — Verify end-to-end.**
      Run the real `_run_tool_agent` against `Qwen/Qwen2.5-1.5B-Instruct` and a
      live retrieval server on the query that previously produced no tool call.
      *Verify:* `search_routing_tool` is called, documents come back, intent is
      `search`, and the answer contains no special tokens.

## Verification

| Gate | Command | Result |
| --- | --- | --- |
| Unit + regression | `python3 -m pytest` | 2782 passed |
| Lint | `ruff check . && ruff format --check .` | clean |
| End-to-end | real model + retrieval on `What is FAISS?` | `search_routing_tool` called, 3 docs, no leak |
