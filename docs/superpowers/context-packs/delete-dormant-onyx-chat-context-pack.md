# Generated Context Pack

# Delete Dormant Onyx Chat

## Sources

- [Specification: 2026-07-03-delete-dormant-onyx-chat-design.md](../specs/2026-07-03-delete-dormant-onyx-chat-design.md)
- [Plan: 2026-07-03-delete-dormant-onyx-chat.md](../plans/2026-07-03-delete-dormant-onyx-chat.md)

## Specification Context

### Non-goals (deferred to a possible phase 2)

- Extracting the two live functions out of `llm_step` (1632) / `chat_utils`
  (733) to then delete those + the tangled `chat_state` / `citation_processor` /
  `emitter` / `tool_call_args_streaming`. That needs real extraction surgery.
- No change to `src/agents`, the live chat path, or any other subsystem.

### Testing

- **Safety proof:** grep confirms every reference to the eight modules is
  internal to the deletion set (or in the two test files handled above); zero
  live/keep-set importers; `chat/__init__.py` re-exports none of them.
- `python -c "import src"` resolves; the chat keep-set imports
  (`queue_manager`, `models`, `tool_models`, `chat_utils`, `llm_step`) still load.
- `ruff check` clean; the unit suite green (minus the removed dormant tests).

## Implementation Plan Context

### Global Constraints

- **No behavior change to any live path.** Keep-set (`queue_manager`, `models`, `tool_models`, `chat_utils`, `llm_step`) untouched.
- Only the eight dormant modules + the two test files change.

---

### Task 1: Trim test_llm_providers.py

- [x] **Step 1:** Remove the test functions that import `get_llm_for_persona`/`LLMOverride` from `process_message` (lines ~200/214/229). If a helper/fixture becomes unused, remove it too.
- [x] **Verify:** `pytest tests/unit/test_llm_providers.py` green (no import of `process_message`).

### Task 2: Delete the dormant modules + test_chat_stubs

- [x] **Step 1:** `git rm` the eight modules + `tests/unit/test_chat_stubs.py`.
- [x] **Step 2:** `python -c "import src"` resolves; keep-set imports (`from src.internal.chat.llm_step import translate_history_to_llm_format`, `chat_utils.run_functions_tuples_in_parallel`, `models`, `queue_manager`, `tool_models`) all load.
- [x] **Verify:** grep confirms zero remaining references to the deleted modules.

### Task 3: Full verification

- [x] `ruff check` clean.
- [x] Broad unit suite green (chat keep-set, agents, tools, retrieval untouched).

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
