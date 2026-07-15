# Generated Context Pack

# Runtime Feedback Memory

## Sources

- [Specification: 2026-07-07-runtime-feedback-memory-design.md](../specs/2026-07-07-runtime-feedback-memory-design.md)
- [Plan: 2026-07-07-runtime-feedback-memory.md](../plans/2026-07-07-runtime-feedback-memory.md)

## Specification Context

### Goals

- Make user memories real local data instead of no-op stubs.
- Sanitize and bound persisted feedback and memory text before storage.
- Keep existing feedback APIs and GRPO training compatibility intact.
- Remove the sampled demo code from the web app module.

### Non-Goals

- Do not add subprocess execution to the web backend.
- Do not build a general event stream or observability subsystem.
- Do not change reward math or feedback training semantics.
- Do not introduce new external dependencies.

### Architecture

Add a small runtime-feedback utility module for redaction and deterministic text capture. The SQLite store owns durable tables for user memories and enhanced retrieval feedback metadata. `src/internal/db/memory.py` becomes a compatibility layer backed by a default local `AgenticSearchStore`, while exposing a store injection hook for tests and callers that already manage a store.

Retrieval feedback remains append-only and summary-compatible. Optional metadata captures a redacted note, source, parent feedback id, and correlation id, so future feedback loops can reconstruct provenance without exposing secrets.

### Components

- `src/internal/feedback/runtime.py`
  - `redact_text(text) -> tuple[str, int]`
  - `deterministic_capture(text, head_lines=5, tail_lines=30, max_chars=4000) -> tuple[str, dict[str, int]]`
  - Redacts bearer tokens, password/API-key style assignments, AWS keys, Slack tokens, and private key blocks.
- `src/internal/db/models.py`
  - Adds `UserMemoryRecord` for store return values.
- `src/internal/db/store.py`
  - Adds `user_memories` table and migrations.
  - Adds `add_user_memory`, `update_user_memory_at_index`, and `get_user_memories`.
  - Extends `retrieval_feedback` with `metadata_json`, `parent_feedback_id`, and `correlation_id`.
  - Adds `list_retrieval_feedback` for tests and future tooling.
- `src/internal/db/memory.py`
  - Keeps the existing public functions but backs them with a configurable store.
- `src/internal/servers/retrieval/feedback_router.py`
  - Accepts optional note/source/lineage fields and passes them into the store.
- `src/internal/servers/web/app.py`
  - Removes the appended sample code.

### Testing

- Unit-test redaction and deterministic capture.
- Unit-test memory add/update/list behavior and sanitization.
- Unit-test feedback metadata persistence, redaction, and existing summary compatibility.
- Unit-test feedback router optional metadata acceptance.
- Run targeted pytest suites plus a syntax/import check for `src.internal.servers.web.app`.

## Implementation Plan Context

### Global Constraints

- No new external dependencies.
- Preserve existing `POST /api/feedback` request compatibility.
- Preserve `load_feedback_examples` compatibility with `retrieval_feedback(session_id, signal)`.
- Redact sensitive text before persistence, not only at read time.
- Keep edits scoped to runtime feedback, memory, and sampled-code cleanup.

---

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

_[Section compacted.]_

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

_[Section compacted.]_

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

_[Section compacted.]_

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

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
