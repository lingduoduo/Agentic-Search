# Memory-augmented generation — Plan

Spec: `docs/superpowers/specs/2026-07-22-memory-augmented-generation-design.md`

A focused threading change across the answer pipeline + a store helper + a flag.

## Step 1 — `service.memory_preamble` (src/internal/memory/service.py)
`MEMORY_INJECTION_MAX = 20`; `memory_preamble(store, user_id, *, max_items) -> str`:
most-recent active memories (`get_user_memories`, tail `max_items`) formatted as
an instructional block; `""` when none.
→ test: `tests/unit/memory/test_injection.py` (empty, cap+newest, format).

## Step 2 — Prompt builders accept + inject `user_memory` (src/context/prompts.py)
Add keyword `user_memory: str | None = None` to `build_answer_prompt`,
`build_chat_prompt` (forwards), `build_structured_answer_prompt`,
`build_corrective_answer_prompt` (forwards). Append `(user_memory or "")` to the
`system` string in the two base builders.
→ test: builders include the memory when set, unchanged when None.

## Step 3 — Thread through the pipeline (src/context/models.py, pipeline.py)
`AnswerGenerationRequest` gains `user_memory: str | None = None`.
`generate_answer` passes `request.user_memory` into every builder call
(`build_chat_prompt`, both `build_structured_answer_prompt` sites, and the
corrective builder inside `_generate_guarded_answer`).
`answer_with_retrieval(..., user_memory=None)` sets it on the request.

## Step 4 — AgenticRAG loop (src/agents/search/agentic_rag.py)
`AgenticRAGLoop.run(..., user_memory=None)` sets it on the
`AnswerGenerationRequest` it builds (its final synthesis reuses the shared
builders, so no prompt change needed there).

## Step 5 — Flag + dispatch wiring (src/internal/servers/web/app.py)
- `SearchExperienceSettings.memory_injection: bool = False`, populated via
  `_flag("AGENTIC_SEARCH_MEMORY_INJECTION")` in `from_app_settings`.
- Import `memory_preamble`.
- In `_run_agent_impl`, right after `user_id` is resolved:
  `user_memory = memory_preamble(db, user_id) if settings.memory_injection and user_id else None`.
- Pass `user_memory=user_memory` into the three answer entry points:
  `_run_auto_routed` (which forwards to its inner `_run_agentic_rag`),
  the explicit chat_loop `_run_agentic_rag`, and the classic `answer_with_retrieval`.
→ test: `tests/unit/servers/web/test_memory_injection.py` — `/api/agent`
  (chat_once) with the flag on + a seeded allergy memory captures the preamble at
  `answer_with_retrieval`; flag off → `None`; the flag reads the env var.

## Step 6 — Fix existing test-doubles
Existing `answer_with_retrieval` fakes with strict signatures gain
`user_memory=None` (the "new optional kwarg breaks strict test-double sigs"
gotcha) in `tests/unit/servers/web/test_web_experience_app.py`.

## Step 7 — Verify
`pytest tests/unit/servers/web/ tests/unit/memory/ tests/unit -k "prompt or
pipeline or agentic_rag or ..."` green; `ruff check`/`ruff format` clean. Commit;
final review; PR.

## Deferred (noted in spec)
Plain-chat + SearchAgent surfaces; query-relevant retrieval; auto-curation;
temporal awareness; semantic conflict detection. Optional real-LLM
peanut→Thai demo script.
