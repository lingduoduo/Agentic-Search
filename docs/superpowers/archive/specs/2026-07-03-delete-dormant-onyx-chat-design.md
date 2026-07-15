# Delete the dormant Onyx chat pipeline (phase 1) — design

**Date:** 2026-07-03
**Status:** Approved (design).
**Scope:** Remove the eight cleanly-dormant modules of the Onyx-derived
`src/internal/chat/` streaming pipeline that no live code path reaches. Phase 1
of a possible two-phase removal — the harder extraction-required tranche
(`llm_step`/`chat_utils` + tangled modules) is explicitly deferred.

## Problem

`src/internal/chat/` (~9.2k LOC) is Onyx-derived scaffolding sampled for ideas.
Its **streaming pipeline is dormant**: `process_message.handle_stream_message_objects`
/ `gather_stream` / `build_chat_turn` are imported by **zero** non-test files;
the repo's live chat runs through `src/agents` + `/api/agent`. Only a narrow
"utility" surface of `chat/` is actually used (`queue_manager`, `models`,
`tool_models`, and one function each from `chat_utils` and `llm_step`).

Eight modules form a self-contained dormant cluster — reachable only from each
other and tests, with **zero** importers from any live file or from the live
"keep" set:

| Module | LOC |
|--------|-----|
| `process_message.py` | 2301 |
| `llm_loop.py` | 1182 |
| `save_chat.py` | 519 |
| `compression.py` | 517 |
| `prompt_utils.py` | 361 |
| `citation_utils.py` | 221 |
| `stop_signal_checker.py` | 58 |
| `chat_processing_checker.py` | 52 |

**~5,211 LOC.** All their cross-imports are internal to this set, so they delete
together cleanly. Most are not even unit-tested.

## Change

1. **Delete the eight modules.**
2. **Delete `tests/unit/test_chat_stubs.py`** — it imports only
   `process_message` / `save_chat` (dormant), nothing from the keep set.
3. **Trim `tests/unit/test_llm_providers.py`** — remove the three test cases that
   import `get_llm_for_persona` / `LLMOverride` from `process_message` (they test
   dormant code; an integration test already marks `get_llm_for_persona` as
   removed).

## What stays (live surface — untouched)

`queue_manager`, `models`, `tool_models` (used by `agents`, `evals`,
`internal/tools`), and `chat_utils` / `llm_step` (each keeps its one live
function: `run_functions_tuples_in_parallel`, `translate_history_to_llm_format`).

## Non-goals (deferred to a possible phase 2)

- Extracting the two live functions out of `llm_step` (1632) / `chat_utils`
  (733) to then delete those + the tangled `chat_state` / `citation_processor` /
  `emitter` / `tool_call_args_streaming`. That needs real extraction surgery.
- No change to `src/agents`, the live chat path, or any other subsystem.

## Testing

- **Safety proof:** grep confirms every reference to the eight modules is
  internal to the deletion set (or in the two test files handled above); zero
  live/keep-set importers; `chat/__init__.py` re-exports none of them.
- `python -c "import src"` resolves; the chat keep-set imports
  (`queue_manager`, `models`, `tool_models`, `chat_utils`, `llm_step`) still load.
- `ruff check` clean; the unit suite green (minus the removed dormant tests).

## Files touched

- **Delete:** the eight `src/internal/chat/*.py` modules + `tests/unit/test_chat_stubs.py`.
- **Modify:** `tests/unit/test_llm_providers.py`.
