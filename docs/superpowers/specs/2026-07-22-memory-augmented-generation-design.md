# Spec: Memory-augmented generation (inject user memory into answers)

## Origin

The conversation-memory work (#456/#457/#458) built the **write + ops** side —
create, curate, consolidate, profile, search memories. But stored memories are
never fed back into conversations: the `UserMemoryContext` slot exists
(`chat_state.py:214`) but nothing populates it, and no prompt builder reads user
memories. So the assistant *stores* what it learns but doesn't *use* it.

This closes the **read/inject** gap: inject the user's durable memory into the
answer prompt so the model reasons over it — enabling retention, preference
tracking, cross-session continuity, and proactive reasoning (e.g. "allergic to
peanuts" → warn when recommending Thai food).

## Key design insight

Proactive recall requires durable facts to be **always present** in context, not
retrieved by query similarity. Asking about *Thai food* would not surface
*"allergic to peanuts"* via lexical/semantic search (no overlap). So memory is
injected **unconditionally** per turn (all active memories, capped), not fetched
by relevance to the question.

## Scope (v1)

- **Inject all active memories (capped)** as a system-prompt preamble.
- **Shared answer pipeline only**: the classic RAG path (`answer_with_retrieval`)
  and `AgenticRAGLoop` (which reuses the same prompt builders).

## Non-goals (deferred)

- Plain-chat (`/chat/send-chat-message`) and `SearchAgentLoop` surfaces.
- Query-relevant retrieval (we inject all-capped for now).
- Auto-curation after turns; temporal awareness; semantic conflict detection.

## Design

### Flow

```
dispatch site (app.py _run_agent_impl — has db + user_id + the flag)
   └─ service.memory_preamble(db, user_id)  →  instructional "User memory" block (or "")
        └─ passed as user_memory=  →  answer_with_retrieval(...)  /  AgenticRAGLoop.run(...)
             └─ AnswerGenerationRequest.user_memory
                  └─ build_answer_prompt / build_structured_answer_prompt  →  appended to system prompt
                       └─ LLM answers with the user's memory in context
```

### Components

1. **`service.memory_preamble(store, user_id, *, max_items=MEMORY_INJECTION_MAX) -> str`**
   (new, `src/internal/memory/service.py`; `MEMORY_INJECTION_MAX = 20`).
   - `mems = store.get_user_memories(user_id)` (active, display order); take the
     most-recent `max_items` (`mems[-max_items:]`).
   - Return `""` when there are none; otherwise an **instructional** block:
     ```
     \n\nWhat you know about this user (remembered from earlier conversations).
     Use these facts when relevant and apply them proactively — honor stated
     preferences, and warn about allergies or constraints:
     - <memory 1>
     - <memory 2>
     ```
   The instructional wording is what drives proactive behavior (peanut→Thai).

2. **Flag** `AGENTIC_SEARCH_MEMORY_INJECTION` — add `memory_injection: bool = False`
   to `SearchExperienceSettings`, populated via the existing `_flag(...)` idiom
   (`app.py`). Off (default) → no fetch, no injection, zero behavior change.

3. **Thread `user_memory: str | None`** (the pre-formatted preamble) through the
   pure prompt layer:
   - `AnswerGenerationRequest` (`src/context/models.py`) gains
     `user_memory: str | None = None`.
   - `build_answer_prompt(question, context, config=None, *, user_memory=None)`
     and `build_structured_answer_prompt(..., *, user_memory=None)`
     (`src/context/prompts.py`) append `user_memory` to their `system` string
     when set. (`build_corrective_answer_prompt` reuses the structured system, so
     it inherits.)
   - `generate_answer` (`src/context/pipeline.py`) passes
     `request.user_memory` into the builders.
   - `answer_with_retrieval(..., *, user_memory=None)` sets it on the request.
   - `AgenticRAGLoop.run(..., user_memory=None)` sets it on the
     `AnswerGenerationRequest` it builds.

4. **Dispatch wiring** (`src/internal/servers/web/app.py`, `_run_agent_impl`):
   `user_id` is already resolved (`user_id = request.user_id or (auth_user.id …)`)
   and `db` is in scope. When `settings.memory_injection` and `user_id`:
   `preamble = memory_preamble(db, user_id)` and pass `user_memory=preamble` into
   the classic `answer_with_retrieval(...)` call and `_run_agentic_rag(...)`
   (which forwards it to `AgenticRAGLoop.run`). Prompt builders remain pure — the
   store read happens only here, at the edge.

## Error handling

- Flag off or no `user_id` → `user_memory` stays `None`; every builder path is a
  no-op (identical output to today).
- Empty memory set → `memory_preamble` returns `""`; treat `""` the same as
  `None` (no block appended).

## Testing

Offline / deterministic (no real LLM, no network):

- **`memory_preamble`** unit: empty user → `""`; caps to `MEMORY_INJECTION_MAX`
  and takes the most-recent; formats the `- ` list with the instructional header.
- **Prompt builders** unit: `build_answer_prompt` and
  `build_structured_answer_prompt` include the memory text in `system` when
  `user_memory` is set, and are byte-identical to today when it is `None`.
- **End-to-end** (`tests/unit/servers/web/`): `/api/agent` (RAG path) with the
  flag on and a seeded memory (`db.add_user_memory(uid, "User is allergic to
  peanuts")`), using a **fake LLM that captures the messages it received**;
  assert the **system message contains "allergic to peanuts"** — proving the
  memory reaches the model. With the flag off, assert it does not.
- **Flag plumbing**: `SearchExperienceSettings.from_app_settings` reads
  `AGENTIC_SEARCH_MEMORY_INJECTION`.

Optional (not CI): an `examples/` script that runs the real LLM to show the
peanut→Thai proactive warning end to end.

## Acceptance criteria

- With `AGENTIC_SEARCH_MEMORY_INJECTION=1`, the RAG + AgenticRAG answer paths
  include the user's active memories (capped) in the system prompt; a seeded
  allergy memory appears in the messages the LLM receives.
- With the flag off, output is unchanged (all tests of existing behavior pass).
- `pytest` green; `ruff` clean.
