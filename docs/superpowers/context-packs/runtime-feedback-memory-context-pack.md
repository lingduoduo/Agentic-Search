# Generated Context Pack

# Runtime Feedback Memory

## Sources

- [Specification: 2026-07-07-runtime-feedback-memory-design.md](../archive/specs/2026-07-07-runtime-feedback-memory-design.md)
- [Plan: 2026-07-07-runtime-feedback-memory.md](../archive/plans/2026-07-07-runtime-feedback-memory.md)

## Specification Context

### Architecture

Add a small runtime-feedback utility module for redaction and deterministic text capture. The SQLite store owns durable tables for user memories and enhanced retrieval feedback metadata. `src/internal/db/memory.py` becomes a compatibility layer backed by a default local `AgenticSearchStore`, while exposing a store injection hook for tests and callers that already manage a store.

Retrieval feedback remains append-only and summary-compatible. Optional metadata captures a redacted note, source, parent feedback id, and correlation id, so future feedback loops can reconstruct provenance without exposing secrets.

## Implementation Plan Context

### Task 1: Runtime Feedback Sanitization Helpers

**Files:**
- Create: `src/internal/feedback/__init__.py`
- Create: `src/internal/feedback/runtime.py`
- Test: `tests/unit/test_runtime_feedback.py`

**Interfaces:**
- Produces: `redact_text(text: str) -> tuple[str, int]`
- Produces: `deterministic_capture(text: str, *, head_lines: int = 5, tail_lines: int = 30, max_chars: int = 4000) -> tuple[str, dict[str, int]]`

- [ ] **Step 1: Write failing tests**

- [ ] **Step 2: Verify tests fail**

Run: `pytest tests/unit/test_runtime_feedback.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.internal.feedback'`.

- [ ] **Step 3: Implement helpers**

…

### Task 2: Durable User Memory Store

**Files:**
- Modify: `src/internal/db/models.py`
- Modify: `src/internal/db/store.py`
- Modify: `src/internal/db/memory.py`
- Test: `tests/unit/db/test_user_memory_store.py`

**Interfaces:**
- Produces: `UserMemoryRecord`
- Produces: `AgenticSearchStore.add_user_memory(user_id: str, memory_text: str, metadata: dict | None = None) -> UserMemoryRecord | None`
- Produces: `AgenticSearchStore.update_user_memory_at_index(user_id: str, index: int, new_text: str, metadata: dict | None = None) -> UserMemoryRecord | None`
- Produces: `AgenticSearchStore.get_user_memories(user_id: str) -> list[str]`
- Produces: `src.internal.db.memory.set_memory_store(store: AgenticSearchStore | None) -> None`

…

### Task 3: Feedback Metadata Persistence

**Files:**
- Modify: `src/internal/db/store.py`
- Modify: `src/internal/servers/retrieval/feedback_router.py`
- Modify: `tests/unit/db/test_retrieval_feedback.py`
- Modify: `tests/unit/servers/retrieval/test_feedback_router.py`

**Interfaces:**
- Produces: `AgenticSearchStore.save_retrieval_feedback(session_id: str | None, signal: str, *, note: str | None = None, source: str | None = None, parent_feedback_id: str | None = None, correlation_id: str | None = None, metadata: dict | None = None) -> str`
- Produces: `AgenticSearchStore.list_retrieval_feedback() -> list[dict[str, object]]`

- [ ] **Step 1: Write failing store tests**

- [ ] **Step 2: Write failing router test**

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
