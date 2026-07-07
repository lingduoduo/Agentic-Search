# Runtime Feedback Memory Design

## Summary

Borrow the durable parts of the sampled `runtime_feedback_loops` code to improve the existing memory and feedback paths. The sample should not live in `src/internal/servers/web/app.py`; instead, its useful ideas become small backend primitives: redaction before persistence, deterministic truncation, structured metadata, and lightweight lineage.

## Goals

- Make user memories real local data instead of no-op stubs.
- Sanitize and bound persisted feedback and memory text before storage.
- Keep existing feedback APIs and GRPO training compatibility intact.
- Remove the sampled demo code from the web app module.

## Non-Goals

- Do not add subprocess execution to the web backend.
- Do not build a general event stream or observability subsystem.
- Do not change reward math or feedback training semantics.
- Do not introduce new external dependencies.

## Architecture

Add a small runtime-feedback utility module for redaction and deterministic text capture. The SQLite store owns durable tables for user memories and enhanced retrieval feedback metadata. `src/internal/db/memory.py` becomes a compatibility layer backed by a default local `AgenticSearchStore`, while exposing a store injection hook for tests and callers that already manage a store.

Retrieval feedback remains append-only and summary-compatible. Optional metadata captures a redacted note, source, parent feedback id, and correlation id, so future feedback loops can reconstruct provenance without exposing secrets.

## Components

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

## Data Flow

Memory writes:

1. Caller invokes `add_memory(user_id, memory_text)` or store method directly.
2. Text is deterministically captured and redacted before persistence.
3. Store writes one active memory row with metadata describing truncation/redaction counts.
4. Reads return ordered active memory text.

Feedback writes:

1. `POST /api/feedback` receives `session_id`, `signal`, and optional metadata.
2. Store sanitizes note text, writes feedback row, and preserves existing summary counts.
3. `load_feedback_examples` continues reading `session_id` and `signal` only, so training behavior stays unchanged.

## Error Handling

- Empty or whitespace-only memory text is ignored and returns `None`.
- Updating an out-of-range memory index returns `None`.
- Feedback signal validation remains enforced by the existing SQLite check and Pydantic model.
- Sanitization runs before writes, so read-time consumers do not need to remember to redact.

## Testing

- Unit-test redaction and deterministic capture.
- Unit-test memory add/update/list behavior and sanitization.
- Unit-test feedback metadata persistence, redaction, and existing summary compatibility.
- Unit-test feedback router optional metadata acceptance.
- Run targeted pytest suites plus a syntax/import check for `src.internal.servers.web.app`.
