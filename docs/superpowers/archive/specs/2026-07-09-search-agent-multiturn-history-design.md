# Search-Agent Multi-Turn History — Design

Date: 2026-07-09
Status: Approved (brainstorming)
Branch/PR: feat/search-agent-multiturn-history
Related: [[project_chat_orchestration]]

## Problem

In the web backend, `_run_search_agent` (`src/internal/servers/web/app.py`) calls
`loop.run([{"role": "user", "content": query}], …)` with only the current query —
it has no `history` parameter. Meanwhile `_run_tool_agent` threads
`history + [query]` and the RAG path threads `chat_history`. So **search-agent
mode is effectively single-turn from the model's perspective**: prior questions
and answers in the same session are invisible to the flagship multi-turn loop.

This was verified firsthand (`app.py:564-591`). Both `_run_search_agent` call
sites (`app.py:839`, `app.py:1459`) already have `history` in scope; it is simply
not plumbed into the function.

Goal: make search-agent mode conversation-aware by threading prior turns into
`SearchAgentLoop.run`, capped tighter than the default because search mode stacks
long `<information>` observations on top of history.

## Non-goals

- No change to the base token-crop (`base.py:186` keep-tail is non-message-aware;
  that is a separate, known gap — out of scope).
- No change to `_run_tool_agent`'s existing history handling or the RAG path.
- No summarization / semantic compression of history.
- No new persistence — history already loads from `AgenticSearchStore` and is
  trimmed to `MAX_HISTORY_MESSAGES = 40` in `_run_agent_impl` (`app.py:1267`).

## Approach

Threading DB history is clean: the store persists only `role:user` (queries) and
`role:assistant` (final answers) — the ephemeral `<information>`/`<search>`
scaffolding lives only in the loop's `working_messages` and is never saved. So
prepending history injects prior Q&A pairs, not scaffolding. `SearchAgentLoop._with_system_prompt`
already handles a multi-message list (prepends the system prompt in front and
keeps the rest), so the model sees `[system, …prior Q&A…, new question]`.

### Components (all in `src/internal/servers/web/app.py`)

1. **Constant** — `SEARCH_AGENT_HISTORY_MESSAGES = 6` (last 3 user/assistant
   exchanges), beside `MAX_HISTORY_MESSAGES`. Tighter than 40 because search mode
   adds its own long observations each turn, and the base token-crop drops from
   the front (the system prompt) on overflow.

2. **Pure helper** — `_build_search_agent_messages(query: str, history: list) -> list[dict]`:
   - `capped = _trim_history(history, max_messages=SEARCH_AGENT_HISTORY_MESSAGES)`
   - map each entry to `{"role": m.role, "content": m.content}`
   - append `{"role": "user", "content": query}`
   - return the list. No model/loop dependency → unit-testable in isolation.

3. **`_run_search_agent`** — add a `history: list` keyword parameter; replace the
   inline `[{"role": "user", "content": query}]` with
   `_build_search_agent_messages(query, history)` passed to `loop.run(...)`.

4. **Call sites** — `app.py:839` and `app.py:1459` pass `history=history`
   (already in scope at both).

## Success criteria

- `_build_search_agent_messages` returns `[…≤6 capped prior turns…, {user, query}]`;
  a history longer than 6 is capped to its last 6; empty history yields just
  `[{user, query}]`; roles/content are preserved.
- `_run_search_agent` passes the built messages (not the bare query) to `loop.run`.
- Existing web tests stay green.

## Testing

Unit tests (no model load — respects the web-test model-load gotcha) on the pure
helper `_build_search_agent_messages`:
1. Empty history → `[{"role": "user", "content": query}]`.
2. Short history (< cap) → all prior turns + query, in order, query last.
3. Long history (> cap) → only the last `SEARCH_AGENT_HISTORY_MESSAGES` prior
   turns + query.
4. Role/content mapping preserved from `ChatMessage`-like entries.

The `_run_search_agent` wiring (loop construction) is covered by the existing web
integration tests; the new behavior worth asserting in isolation is the message
buffer construction, which the pure helper localizes.

## Risks

- **Token budget**: history + growing observations could still overflow and the
  base crop would drop the system prompt. Mitigated (not eliminated) by the tight
  cap of 6; the deeper fix (message-aware crop) is deliberately out of scope.
