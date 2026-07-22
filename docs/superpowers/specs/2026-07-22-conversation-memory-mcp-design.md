# Spec: Conversation-memory MCP tools (LLM-generated user profile)

## Origin

`src/internal/mcp_server/memory.py` is a copy-pasted **Memobase** example
(save_memory / get_user_profiles / search_memories) that depends on an external
SDK, uses the wrong FastMCP import, references undefined symbols, and is not
registered. We borrow its *ideas* and implement them natively on this repo's own
store, LLM, and e5 encoder — no external service.

The LLM-generation centerpiece is a **structured user profile**: the LLM reads
the user's conversation turns and saved notes and consolidates them into
`{topic, subtopic, content}` entries, which are persisted.

## Goals

Native MCP tools:

- `save_memory(text)` — store an explicit note.
- `generate_user_profile()` — LLM reads the user's conversations + saved notes
  and (re)generates a structured profile; persists it.
- `get_user_profile()` — return the persisted structured profile.
- `search_memories(query, max_results)` — semantic recall over saved notes.

## Non-goals

- Auto-generating the profile on every chat turn (kept explicit/on-demand).
- Wiring the profile back into the chat pipeline's `UserMemoryContext` (a later,
  separate step — the injection point already exists).
- A separate memory microservice (Memobase style). Everything runs in-process.
- Per-memory embedding storage/index (embeddings are computed on-the-fly).

## Architecture

Approach A — thin MCP tools over a testable service module:

```
src/internal/mcp_server/tools/memory.py   (thin @mcp_server.tool wrappers)
        │  resolves user_id from the MCP bearer identity
        ▼
src/internal/memory/service.py            (all logic; no MCP dependency)
        ├─ save_memory  → store.add_user_memory(user_id, text)
        ├─ generate_profile → gather() → LLM.complete() → parse → store.replace_user_profile()
        ├─ get_profile  → store.get_user_profile(user_id)
        └─ search       → e5 cosine over notes (lexical fallback)
```

Reuses: `AgenticSearchStore` (SQLite), `OpenAICompatibleLLM.complete`
(built via the existing `_build_llm()` env pattern), and
`build_e5_encoder` from `servers/retrieval/hybrid.py`.

### Data flow — `generate_user_profile()`

1. `gather(user_id)`:
   - conversations: `store.list_chat_sessions(user_id)` →
     `store.list_chat_messages(session_id)` → `role: content` lines.
   - notes: `store.get_user_memories(user_id)`.
   - Concatenate most-recent-first, truncated to a token/char budget.
2. One `LLM.complete()` call with a consolidation prompt → JSON array of
   `{topic, subtopic, content}`. Parsed defensively (bad/empty JSON → empty
   list, tool returns a clear message; no crash).
3. `store.replace_user_profile(user_id, entries)` — atomic delete-then-insert so
   regeneration fully replaces the prior profile.
4. Return the persisted entries.

### Identity

`user_id` is a stable id derived from the authenticated bearer identity — the
subject/email the existing `AgenticSearchTokenVerifier` (`/me` delegation)
already resolves for a request. When unauthenticated (local/tests), fall back to
a `DEFAULT_MEMORY_USER_ID` constant. The exact accessor for that identity is
pinned in the plan against what the token verifier exposes.

## Storage

- Reuse `user_memories` (existing) for saved notes.
- **New `user_profiles` table**, added to `_init_schema` `executescript` and
  (for older DBs) `_migrate_schema`, following the existing pattern:

  ```sql
  CREATE TABLE IF NOT EXISTS user_profiles (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      topic TEXT NOT NULL,
      subtopic TEXT NOT NULL DEFAULT '',
      content TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_user_profiles_user
      ON user_profiles(user_id, topic, id);
  ```

- New `UserProfileEntryRecord` dataclass in `db/models.py` (+ export).
- New accessors on `AgenticSearchStore`:
  - `replace_user_profile(user_id, entries: list[dict|record]) -> list[UserProfileEntryRecord]`
    (delete existing rows for the user, insert the new set, in one transaction).
  - `get_user_profile(user_id) -> list[UserProfileEntryRecord]` (topic order).
  - `list_chat_sessions(user_id) -> list[ChatSessionRecord]` (uses the existing
    `idx_chat_sessions_user_updated` index, most-recent-first).

## Semantic search

`search_memories` embeds the user's notes with `build_e5_encoder` on demand
(small N per user — no persisted embeddings, no staleness) and ranks by cosine
(`query_vec @ note_matrix.T`, e5-normalized).

**Graceful fallback:** if `sentence_transformers` / the e5 model is unavailable,
fall back to lexical token-overlap ranking (mirrors `hybrid.py`'s "dense leg
unavailable" path). Returns the top `max_results` notes with scores.

## Error handling

- No LLM configured → `generate_user_profile` returns a clear message, no crash
  (mirrors `chat.py`'s `_build_llm() -> … | None`).
- Malformed LLM JSON → parsed to empty; tool reports "could not parse".
- Empty sources → profile generation returns "no conversations or notes yet".
- Every tool wraps its body in try/except returning an error string (the MCP
  tool convention already used in the stub and existing tools).

## Testing

Unit tests (no model download, no network):

- Store: `user_profiles` replace/get round-trip; `replace` overwrites prior;
  `list_chat_sessions` returns a user's sessions newest-first.
- `service.generate_profile` with a monkeypatched LLM returning JSON → asserts
  persisted entries; malformed JSON → empty + message.
- `service.search` lexical fallback → deterministic ranking by token overlap.
- `service.get_profile` returns persisted entries.
- Registration: importing `tools.memory` registers the tools and `api.py`
  imports it without error.

## Config

No new required config. Reuse the existing LLM env (`GEN_AI_*` /
`AGENTIC_SEARCH_LLM_*`) via `_build_llm()`, and the e5 default
(`intfloat/e5-base-v2`). A token/char budget for gathering is a module constant.

## Cleanup

Remove `src/internal/mcp_server/memory.py` (the non-importing Memobase stub);
its ideas live on in `tools/memory.py` + `src/internal/memory/service.py`.

## Acceptance criteria

- The four tools are registered in `api.py` and callable.
- `generate_user_profile` persists `{topic, subtopic, content}` entries derived
  by the LLM from conversations + notes; `get_user_profile` returns them.
- `search_memories` returns semantically/lexically ranked notes and never
  requires an external service or a forced model download in tests.
- `pytest` green; `ruff` clean.
