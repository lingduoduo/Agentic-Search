# Runtime Feedback Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add durable, sanitized memory storage and richer feedback metadata by borrowing the useful structured-capture ideas from the sampled runtime feedback code.

**Architecture:** Create a focused runtime feedback utility module for redaction and bounded capture. Extend `AgenticSearchStore` for user memories and feedback metadata while keeping current APIs and training readers compatible. Remove the sampled demo code from `src/internal/servers/web/app.py`.

**Tech Stack:** Python 3.10+, SQLite, FastAPI/Pydantic, pytest.

## Global Constraints

- No new external dependencies.
- Preserve existing `POST /api/feedback` request compatibility.
- Preserve `load_feedback_examples` compatibility with `retrieval_feedback(session_id, signal)`.
- Redact sensitive text before persistence, not only at read time.
- Keep edits scoped to runtime feedback, memory, and sampled-code cleanup.

---

## File Structure

- Create `src/internal/feedback/__init__.py`: package exports for feedback utilities.
- Create `src/internal/feedback/runtime.py`: redaction and deterministic capture helpers.
- Modify `src/internal/db/models.py`: add `UserMemoryRecord`.
- Modify `src/internal/db/store.py`: add memory table/methods and feedback metadata columns/methods.
- Modify `src/internal/db/memory.py`: use the SQLite store via a configurable default store.
- Modify `src/internal/servers/retrieval/feedback_router.py`: accept optional metadata fields.
- Modify `src/internal/servers/web/app.py`: remove appended sampled demo code.
- Create `tests/unit/test_runtime_feedback.py`: utility tests.
- Create `tests/unit/db/test_user_memory_store.py`: memory persistence tests.
- Modify `tests/unit/db/test_retrieval_feedback.py`: feedback metadata tests.
- Modify `tests/unit/servers/retrieval/test_feedback_router.py`: router metadata tests.

### Task 1: Runtime Feedback Sanitization Helpers

**Files:**
- Create: `src/internal/feedback/__init__.py`
- Create: `src/internal/feedback/runtime.py`
- Test: `tests/unit/test_runtime_feedback.py`

**Interfaces:**
- Produces: `redact_text(text: str) -> tuple[str, int]`
- Produces: `deterministic_capture(text: str, *, head_lines: int = 5, tail_lines: int = 30, max_chars: int = 4000) -> tuple[str, dict[str, int]]`

- [ ] **Step 1: Write failing tests**

```python
from src.internal.feedback.runtime import deterministic_capture, redact_text


def test_redact_text_masks_common_secret_shapes():
    text = "Authorization: Bearer abc.def\na password=hunter2\nAKIAIOSFODNN7EXAMPLE"
    redacted, hits = redact_text(text)
    assert hits == 3
    assert "hunter2" not in redacted
    assert "abc.def" not in redacted
    assert "AKIAIOSFODNN7EXAMPLE" not in redacted


def test_deterministic_capture_keeps_head_tail_and_counts_removed_lines():
    text = "\n".join(f"line {i}" for i in range(10))
    captured, meta = deterministic_capture(text, head_lines=2, tail_lines=2, max_chars=1000)
    assert captured.splitlines() == [
        "line 0",
        "line 1",
        "...truncated 6 lines...",
        "line 8",
        "line 9",
    ]
    assert meta["truncated_lines"] == 6
    assert meta["redactions"] == 0


def test_deterministic_capture_redacts_before_returning_text():
    captured, meta = deterministic_capture("token=secret-value", max_chars=1000)
    assert "secret-value" not in captured
    assert meta["redactions"] == 1
```

- [ ] **Step 2: Verify tests fail**

Run: `pytest tests/unit/test_runtime_feedback.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.internal.feedback'`.

- [ ] **Step 3: Implement helpers**

Create `src/internal/feedback/runtime.py` with compiled redaction patterns and deterministic line/char bounds.

- [ ] **Step 4: Verify tests pass**

Run: `pytest tests/unit/test_runtime_feedback.py -v`

Expected: PASS.

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

- [ ] **Step 1: Write failing tests**

```python
from src.internal.db.memory import add_memory, get_memories, set_memory_store, update_memory_at_index
from src.internal.db.store import AgenticSearchStore


def test_memory_store_adds_and_lists_sanitized_memories():
    store = AgenticSearchStore(":memory:")
    record = store.add_user_memory("u1", "My API key is api_key=secret123")
    assert record is not None
    assert store.get_user_memories("u1") == ["My API key is api_key=[REDACTED]"]
    assert record.metadata["redactions"] == 1


def test_memory_store_updates_by_zero_based_index():
    store = AgenticSearchStore(":memory:")
    store.add_user_memory("u1", "old")
    updated = store.update_user_memory_at_index("u1", 0, "new")
    assert updated is not None
    assert store.get_user_memories("u1") == ["new"]


def test_memory_store_rejects_blank_and_out_of_range_updates():
    store = AgenticSearchStore(":memory:")
    assert store.add_user_memory("u1", "   ") is None
    assert store.update_user_memory_at_index("u1", 3, "new") is None


def test_memory_module_uses_configured_store():
    store = AgenticSearchStore(":memory:")
    set_memory_store(store)
    try:
        memory_id = add_memory("u1", "prefers concise answers")
        assert memory_id is not None
        assert get_memories("u1") == ["prefers concise answers"]
        assert update_memory_at_index("u1", 0, "prefers detailed answers") == memory_id
        assert get_memories("u1") == ["prefers detailed answers"]
    finally:
        set_memory_store(None)
```

- [ ] **Step 2: Verify tests fail**

Run: `pytest tests/unit/db/test_user_memory_store.py -v`

Expected: FAIL because store memory methods do not exist and module functions still return stubs.

- [ ] **Step 3: Implement store schema, migrations, and methods**

Add `user_memories` table with `id`, `user_id`, `memory_text`, `metadata_json`, `is_active`, `created_at`, and `updated_at`; add methods using `deterministic_capture`.

- [ ] **Step 4: Implement memory compatibility wrapper**

Back `add_memory`, `update_memory_at_index`, and `get_memories` with a configurable `AgenticSearchStore`.

- [ ] **Step 5: Verify tests pass**

Run: `pytest tests/unit/db/test_user_memory_store.py -v`

Expected: PASS.

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

```python
def test_feedback_metadata_is_redacted_and_queryable():
    db = _store()
    feedback_id = db.save_retrieval_feedback(
        "s1",
        "thumbs_down",
        note="failed with Bearer abc.def",
        source="answer_panel",
        correlation_id="turn-1",
    )
    rows = db.list_retrieval_feedback()
    assert rows[0]["id"] == feedback_id
    assert rows[0]["metadata"]["note"] == "failed with Bearer [REDACTED]"
    assert rows[0]["metadata"]["source"] == "answer_panel"
    assert rows[0]["correlation_id"] == "turn-1"
    assert db.get_feedback_summary()["rated_queries"] == 1
```

- [ ] **Step 2: Write failing router test**

```python
def test_feedback_router_accepts_optional_metadata():
    db = AgenticSearchStore(":memory:")
    client = _app(db)
    resp = client.post(
        "/api/feedback",
        json={
            "session_id": "s1",
            "signal": "thumbs_down",
            "note": "token=secret-value",
            "source": "answer_panel",
            "correlation_id": "turn-1",
        },
    )
    assert resp.status_code == 200
    row = db.list_retrieval_feedback()[0]
    assert row["metadata"]["note"] == "token=[REDACTED]"
    assert row["metadata"]["source"] == "answer_panel"
```

- [ ] **Step 3: Verify tests fail**

Run: `pytest tests/unit/db/test_retrieval_feedback.py tests/unit/servers/retrieval/test_feedback_router.py -v`

Expected: FAIL because optional metadata fields and `list_retrieval_feedback` do not exist.

- [ ] **Step 4: Implement metadata persistence**

Add migration columns to `retrieval_feedback`, sanitize note text with `deterministic_capture`, return the inserted feedback id, and preserve summary queries.

- [ ] **Step 5: Update router model and handler**

Add optional fields to `FeedbackRequest` and pass them to `save_retrieval_feedback`.

- [ ] **Step 6: Verify tests pass**

Run: `pytest tests/unit/db/test_retrieval_feedback.py tests/unit/servers/retrieval/test_feedback_router.py -v`

Expected: PASS.

### Task 4: Sample Code Cleanup and Regression Verification

**Files:**
- Modify: `src/internal/servers/web/app.py`

**Interfaces:**
- Consumes: helper/store/router changes from Tasks 1-3.
- Produces: valid importable web app module without appended demo code.

- [ ] **Step 1: Remove appended sample code**

Delete everything after the first `app = create_web_app()` assignment in `src/internal/servers/web/app.py`.

- [ ] **Step 2: Verify import syntax**

Run: `python -m py_compile src/internal/servers/web/app.py`

Expected: exit code 0.

- [ ] **Step 3: Run targeted regression tests**

Run: `pytest tests/unit/test_runtime_feedback.py tests/unit/db/test_user_memory_store.py tests/unit/db/test_retrieval_feedback.py tests/unit/servers/retrieval/test_feedback_router.py tests/unit/test_feedback_examples.py -v`

Expected: PASS.

- [ ] **Step 4: Inspect diff**

Run: `git diff --check`

Expected: no whitespace errors.
