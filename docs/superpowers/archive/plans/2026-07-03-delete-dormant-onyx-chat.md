# Delete Dormant Onyx Chat Pipeline (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Delete the eight cleanly-dormant Onyx chat modules (~5,211 LOC) + their test cleanup, leaving the live utility surface intact.

**Architecture:** Pure deletion. The eight modules cross-import only each other; nothing live or in the keep-set imports them. Two test files handled: delete `test_chat_stubs.py`, trim three cases from `test_llm_providers.py`.

**Tech Stack:** Python 3, pytest.

**Spec:** `docs/superpowers/specs/2026-07-03-delete-dormant-onyx-chat-design.md`.

## Global Constraints

- **No behavior change to any live path.** Keep-set (`queue_manager`, `models`, `tool_models`, `chat_utils`, `llm_step`) untouched.
- Only the eight dormant modules + the two test files change.

---

## File Structure

- **Delete** `src/internal/chat/{process_message,llm_loop,save_chat,compression,prompt_utils,citation_utils,stop_signal_checker,chat_processing_checker}.py`.
- **Delete** `tests/unit/test_chat_stubs.py`.
- **Modify** `tests/unit/test_llm_providers.py` — drop the 3 `get_llm_for_persona`/`LLMOverride` test cases.

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
