# Search-Agent Deterministic Auto-Search Design

Status: accepted
Date: 2026-06-28

## Goal

Guarantee that `SearchAgentLoop` performs at least one retrieval round per run,
even when the policy model fails to emit a recognized `<search>` action tag. The
loop's control flow should *trigger search deterministically* on the first turn
rather than depending entirely on the model producing well-formed tagged output.

## Current Problem

`SearchAgentLoop.run` is fully LLM-gated for triggering retrieval:

1. Each turn the policy model generates text; `Planner.parse_actions` extracts
   `<search>` / `<answer>` tags from it (`src/agents/search.py:989`).
2. If the generation contains **no recognized tag**, the loop routes into
   `_handle_no_action`, which records a `format_recovery` event and re-prompts
   (`src/agents/search.py:1402-1434`).
3. After `max_consecutive_format_errors` (default 3) consecutive tag-less turns,
   the loop dead-ends with `exit_status="format_error_limit"` and emits an empty
   (or weakly forced) answer (`src/agents/search.py:1229-1251`).

Consequently a small or weakly-instruction-tuned policy model (e.g.
`Qwen/Qwen2.5-0.5B-Instruct`) that never emits a parseable `<search>` tag causes
the loop to return an **empty answer with zero retrieved documents** — retrieval
is never invoked at all.

There is already a dormant fallback: `Planner.decide()` returns a best-effort
`SearchAction` for unparseable input (`src/agents/components/planner.py:139-148`),
but the run loop uses `parse_actions`, so `decide()` is never called. The intent
("always be able to search") exists in code but is not wired into control flow.

## Chosen Approach

Add a deterministic first-turn auto-search to the control flow:

When a turn yields **no recognized action** AND **no search round has run yet**
(`state.search_rounds == 0`), synthesize a search action on the user's original
question and feed it through the existing search-execution path, instead of
entering the `format_recovery` / dead-end branch.

Concretely, in `SearchAgentLoop.run`, immediately after parsing actions:

```python
if (
    not actions
    and cfg.auto_search_on_deadend
    and state.search_rounds == 0
    and question
):
    actions = [(cfg.search_tag, question)]
    recorder.record(
        turn=num_turns,
        component="planner",
        action="auto_search",
        status="decided",
        details={"query": question, "reason": "no_action_first_turn"},
    )
```

The synthesized `(search_tag, question)` action is consumed by the unchanged
downstream machinery: `_collect_requested_queries_and_urls` →
`_parse_query_specifications` → `_parse_queries` returns `[question]` for the
search tag → normal partition / `SearchToolCall` / search round execution. After
the round, evidence is injected and the model gets a second chance to answer.

A new config flag gates the behavior:

```python
# Auto-issue a search on the user's question when the first turn produces no
# recognized action tag, so retrieval always fires at least once.
auto_search_on_deadend: bool = True
```

## Why first-turn only

The guard `state.search_rounds == 0` scopes the fallback to the *entry* of the
loop. This:

- Guarantees retrieval happens at least once (the user's intent: "clicking
  search should trigger the search tool").
- Does **not** mask later format failures. If the model keeps emitting garbage
  *after* a successful search, the existing `format_recovery` → dead-end →
  `force_answer_on_deadend` path still applies, so the loop still terminates.
- Avoids an infinite auto-search loop: once `search_rounds > 0`, the fallback no
  longer fires.

## Alternatives Rejected

### Call `Planner.decide()` instead of forcing the user question
`decide()` derives its fallback query from the *first non-empty line of the
model's (garbage) generation* (`planner.py:141-143`). On turn 1 that is usually
reasoning noise, not the user's actual question. Forcing `state.question`
produces a far more relevant retrieval. (`decide()` remains useful for other
call sites and is left intact.)

### Lower `max_consecutive_format_errors` to 0 / force-answer immediately
This makes dead-ends faster but still returns an answer with **no retrieved
evidence** — it does not satisfy "search must actually run."

### Wire the M10 routing layer (`src/internal/routing/`) into the loop
The per-query router lives in `RetrievalService`; `SearchAgentLoop` talks to
`SearchClient`/`search_url` directly and bypasses it. Routing the loop through
`RetrievalService` is a larger, separate change (tracked under the control-flow
roadmap) and is out of scope here. This spec only guarantees that *a* search
fires.

## Training Considerations

`SearchAgentLoop` is also the GRPO rollout loop. Forcing a turn-1 search removes
the implicit penalty for a model that never emits `<search>` (the rollout would
otherwise dead-end and score poorly). To preserve training dynamics, RL rollout
configs may set `auto_search_on_deadend=False`. The default is `True` because the
serving path (web `search_agent` mode) benefits from the guarantee. This keeps
backward-compatible behavior available via one flag.

## Testing

### Unit tests (`tests/unit/test_agent_loop.py`)

1. `test_search_agent_loop_auto_searches_when_model_emits_no_action`: model emits
   a tag-less first turn, then an `<answer>`. Assert the search client was called
   once with the user's question, `search_rounds == 1`, the final answer is
   produced, and an `auto_search` control-flow event was recorded.
2. `test_search_agent_loop_auto_search_disabled_preserves_format_recovery`: with
   `auto_search_on_deadend=False`, a tag-less run never calls the search client
   and `search_rounds == 0` (legacy behavior intact).

## Files Touched

- `src/agents/search.py` — add `auto_search_on_deadend` config field; inject the
  first-turn auto-search in `SearchAgentLoop.run`.
- `tests/unit/test_agent_loop.py` — two new unit tests.
