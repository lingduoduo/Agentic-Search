# Generated Context Pack

# Delete Dormant Onyx Chat

## Sources

- [Specification: 2026-07-03-delete-dormant-onyx-chat-design.md](../specs/2026-07-03-delete-dormant-onyx-chat-design.md)
- [Plan: 2026-07-03-delete-dormant-onyx-chat.md](../plans/2026-07-03-delete-dormant-onyx-chat.md)

## Specification Context

### Overview

**Date:** 2026-07-03
**Status:** Approved (design).
**Scope:** Remove the eight cleanly-dormant modules of the Onyx-derived
`src/internal/chat/` streaming pipeline that no live code path reaches. Phase 1
of a possible two-phase removal — the harder extraction-required tranche
(`llm_step`/`chat_utils` + tangled modules) is explicitly deferred.

## Implementation Plan Context

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
