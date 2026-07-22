# Spec: Conversation-memory MCP tools (agentic curation + LLM profile)

## Origin

`src/internal/mcp_server/memory.py` holds two copy-pasted examples used as
design references only:

1. A **Memobase** MCP server (save_memory / get_user_profiles / search_memories)
   — external SDK, wrong FastMCP import, undefined symbols, unregistered.
2. A **UserMemoryAgent** — a hand-rolled ReAct loop where the LLM curates
   memories with `add_memory` / `update_memory` / `delete_memory` tool calls,
   plus a provider zoo, streaming, trajectory files, and four memory "modes".

We borrow the *ideas* and implement them natively on this repo's own store, LLM,
`ToolRegistry`, and e5 encoder — no external service, no provider/streaming code.

**Curation engine note:** `ToolAgentLoop` is bound to the local model stack
(needs `tokenizer` + `server_manager`); the MCP server uses the remote
`OpenAICompatibleLLM`. So curation is a **thin bounded loop over the LLM's native
tool-calling** that dispatches through the same `ToolRegistry` `ToolAgentLoop`
uses internally — not `ToolAgentLoop` itself.

## What we adopt vs. drop

**Adopt:**
- **Agentic memory curation** — the LLM reads a conversation + existing memories
  and calls add/update/delete to *reconcile* the memory set (handles
  contradictions, not just appends). This is the "use LLM to generate
  conversation memory" centerpiece.
- **update / delete** memory operations (the store has add + update-by-index +
  get today; we add id-based update/delete).
- **Enhanced-notes** style: memories are full contextual sentences, not bare
  key-values.
- A persisted **structured user profile** (`topic / subtopic / content`),
  consolidated from the curated memories.
- A persisted, inspectable **curation trajectory** (what the LLM saw, its tool
  calls, before/after memory state) — implemented repo-native (a store table +
  returned from the tool), echoing the existing request-capture pattern, not a
  loose `memory_trajectory.json` in cwd.
- A **secret guardrail**: the curation prompt instructs the LLM not to persist
  obvious secrets (passwords, PINs, full SSNs, full card/account numbers).
- A deterministic **consolidate** op (no LLM): exact-content dedup + attribute
  conflict resolution (keep the newest per attribute, supersede older),
  returning a report. Complements the LLM curation. The attribute key is an
  optional `tags`/`attribute` stored in the existing `user_memories.metadata_json`
  (no schema change); memories with no attribute are only exact-deduped.

**Drop (YAGNI / this repo has equivalents):**
- Provider zoo (siliconflow/doubao/kimi/moonshot/openrouter) + reasoning-temp
  hacks → use `OpenAICompatibleLLM`.
- Console streaming, verbose prints, and the loose `memory_trajectory.json`
  file (we persist the trajectory to the store instead).
- The four memory modes → **one** representation (enhanced notes for raw
  memories; structured topic/subtopic/content for the profile).
- Hand-rolled provider/streaming/ReAct plumbing → reuse `OpenAICompatibleLLM`
  (native tool-calling) + `ToolRegistry` (dispatch) under a thin bounded loop.

## Goals — MCP tools

- `save_memory(text)` — store one explicit note.
- `update_memory_from_conversation(session_id=None)` — **agentic curation.** A
  thin bounded loop drives `OpenAICompatibleLLM`'s tool-calling with internal
  add/update/delete tools over the user's conversation turns + current memories;
  the LLM reconciles the memory set. Persists a trajectory and returns its
  summary.
- `generate_user_profile()` — LLM consolidates the curated memories into
  structured `{topic, subtopic, content}` entries; persists them.
- `get_user_profile()` — return the persisted structured profile.
- `search_memories(query, max_results)` — semantic recall over the memories.
- `consolidate_memories(resolve_conflicts=True)` — deterministic dedup +
  attribute conflict resolution; returns a `{initial, duplicates_removed,
  conflicts_resolved[], final}` report. No LLM.

`add/update/delete` are **internal** `Tool`s the curation loop calls — not part
of the MCP surface (which stays: save / update-from-conversation / generate /
get / search / consolidate).

## Non-goals

- Auto-curating on every chat turn (kept explicit/on-demand).
- Wiring the profile back into the chat pipeline's `UserMemoryContext` (later,
  separate — the injection point already exists).
- A memory microservice; per-memory embedding storage; selectable memory modes.

## Architecture

Approach A — thin MCP tools over a testable service:

```
src/internal/mcp_server/tools/memory.py   (thin @mcp_server.tool wrappers)
        │  resolves user_id from the MCP bearer identity
        ▼
src/internal/memory/service.py            (logic; no MCP dependency)
        ├─ save            → store.add_user_memory(user_id, text)
        ├─ curate          → thin loop: OpenAICompatibleLLM tool-calling +
        │                     ToolRegistry(add/update/delete) over
        │                     conversation + current memories
        ├─ generate_profile→ LLM.complete() over curated memories → parse
        │                     → store.replace_user_profile()
        ├─ get_profile     → store.get_user_profile(user_id)
        ├─ search          → e5 cosine over memories (lexical fallback)
        └─ consolidate     → dedup + attribute conflict resolution (no LLM)

src/internal/memory/tools.py              (add/update/delete Tool factory,
                                           closing over user_id + store)
```

Reuses: `AgenticSearchStore`, `OpenAICompatibleLLM` (built via the existing
`_build_llm()` env pattern; native `tools=[...]` function-calling),
`ToolRegistry` (`src/tools/registry.py`, the same dispatch path `ToolAgentLoop`
uses), and `build_e5_encoder` (`servers/retrieval/hybrid.py`).

### Data flow — `update_memory_from_conversation(session_id=None)`

1. Gather source text:
   - conversation turns: `store.list_chat_sessions(user_id)` (or the one
     `session_id`) → `store.list_chat_messages` → `role: content` lines.
   - current memories: `store.get_user_memory_records(user_id)` → `[id] text`.
   - Truncated to a char/token budget, most-recent-first.
2. Build a `ToolRegistry` with three tools bound to `(user_id, store)`:
   `add_memory(content)`, `update_memory(memory_id, content)`,
   `delete_memory(memory_id)`.
3. Thin bounded loop (≤ `MAX_CURATION_TURNS`): call
   `llm` with the enhanced-notes system prompt (which includes the **secret
   guardrail** — do not store passwords, PINs, full SSNs, full card/account
   numbers) + gathered text + `tools`; for each returned tool call,
   `registry.invoke(...)`; append results; repeat until the model stops emitting
   tool calls (or the cap is hit).
4. Build a trajectory record — `{model, session_id, memory_before, tool_calls
   (name/args/result), memory_after, counts}` — and persist it via
   `store.add_memory_trajectory(...)`.
5. Return the trajectory summary (counts + before/after sizes).

### Data flow — `generate_user_profile()`

`LLM.complete()` over the curated memories with a consolidation prompt → JSON
array of `{topic, subtopic, content}`, parsed defensively (bad/empty → empty
list + clear message) → `store.replace_user_profile(user_id, entries)` (atomic
delete-then-insert). Returns the entries.

### Data flow — `consolidate_memories(resolve_conflicts=True)` (no LLM)

Ported from the Go sample's `Consolidate`, operating on the shared store:

1. Load `store.get_user_memory_records(user_id)`.
2. **Dedup:** drop memories with identical trimmed content, keeping the newest
   (`updated_at`). Count removals.
3. **Conflict resolution** (when `resolve_conflicts`): group by `attribute`
   (a `tags[0]`-style key read from each memory's `metadata_json`); within each
   group keep the newest and mark the rest superseded, recording
   `{attribute, kept, superseded[]}`. Memories without an attribute pass through.
4. Apply by soft-deleting the removed/superseded memories via
   `delete_user_memory`; return the `ConsolidationReport`
   (`initial / duplicates_removed / conflicts_resolved[] / final`).

### Identity

`user_id` is a stable id derived from the authenticated bearer identity — the
subject/email the existing `AgenticSearchTokenVerifier` (`/me` delegation)
resolves. Unauthenticated (local/tests) → a `DEFAULT_MEMORY_USER_ID` constant.
Exact accessor pinned in the plan against what the verifier exposes.

## Storage

- Reuse `user_memories` for notes. **Add id-based accessors** so the agent can
  target memories reliably (not by fragile offset):
  - `update_user_memory(user_id, memory_id, new_text) -> bool`
  - `delete_user_memory(user_id, memory_id) -> bool` (soft delete via the
    existing `is_active` column)
  - `get_user_memory_records(user_id) -> list[UserMemoryRecord]` (id + text)
- **New `user_profiles` table** via the established `_init_schema` +
  `_migrate_schema` pattern:

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

- New `UserProfileEntryRecord` in `db/models.py` (+ export).
- New accessors: `replace_user_profile`, `get_user_profile`,
  `list_chat_sessions(user_id)` (uses `idx_chat_sessions_user_updated`).
- **New `memory_trajectories` table** (curation audit trail):

  ```sql
  CREATE TABLE IF NOT EXISTS memory_trajectories (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      session_id TEXT,
      model TEXT NOT NULL DEFAULT '',
      trajectory_json TEXT NOT NULL DEFAULT '{}',
      created_at TEXT NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_memory_trajectories_user
      ON memory_trajectories(user_id, created_at, id);
  ```

  `trajectory_json` holds `{memory_before, tool_calls, memory_after, counts}`.
  New `MemoryTrajectoryRecord` in `db/models.py` (+ export) and accessors
  `add_memory_trajectory(...)` / `list_memory_trajectories(user_id, limit)`.
  (A Dev Console panel over this is a later, separate step — not in scope here.)

## Semantic search

`search_memories` embeds the user's memories with `build_e5_encoder` on demand
(small N — no persisted embeddings) and ranks by cosine. **Graceful fallback:**
if `sentence_transformers`/e5 is unavailable, fall back to lexical token-overlap
ranking (mirrors `hybrid.py`'s "dense leg unavailable"). Returns the top
`max_results` memories with scores.

## Error handling

- No LLM configured → `curate` / `generate_user_profile` return a clear message,
  no crash (mirrors `chat.py`'s `_build_llm() -> … | None`).
- Malformed tool-call args in the loop → answered with an error tool result and
  the loop continues (never leaves a tool call unanswered).
- Malformed profile JSON → empty + "could not parse".
- Empty sources → "no conversations or notes yet".
- Every MCP tool wraps its body in try/except returning an error string.

## Testing

Unit tests (no model download, no network):

- Store: id-based `update_user_memory` / `delete_user_memory` /
  `get_user_memory_records`; `user_profiles` replace/get round-trip + overwrite;
  `list_chat_sessions` newest-first.
- Memory tools (`add/update/delete` `Tool`s) invoked directly → assert the store
  mutates correctly (no LLM needed).
- `service.curate` with a **fake LLM** that emits scripted tool calls then stops
  → assert add/update/delete land in the store, the turn cap is honored, and a
  `memory_trajectories` row is persisted with before/after + tool calls.
- Store: `add_memory_trajectory` / `list_memory_trajectories` round-trip.
- `service.generate_profile` with a fake LLM returning JSON → persisted entries;
  malformed JSON → empty + message.
- `service.search` lexical fallback → deterministic ranking.
- `service.consolidate` → exact duplicates removed (newest kept); tagged
  conflicts resolve to newest-per-attribute with a correct report; untagged
  memories survive.
- Registration: importing `tools.memory` registers the tools; `api.py` imports
  it without error.

## Config

No new required config. Reuse the LLM env (`GEN_AI_*` / `AGENTIC_SEARCH_LLM_*`)
via `_build_llm()` and the e5 default (`intfloat/e5-base-v2`). Gather budget and
`MAX_CURATION_TURNS` are module constants.

## Follow-on deliverable (separate spec) — memory CLI

Out of scope here, recorded so this spec builds toward it: a CLI to manage
memory (add / query / update / show / consolidate). It will be a **subcommand of
the existing Go `cli/`** (`cli/api/client.go` HTTP client), calling **backend
memory HTTP endpoints** on the web app — one source of truth (SQLite), reusing
`memory/service.py`. We are **not** adding the sampled standalone Go tool with
its own `data/memories/*.json` store (it would fork user-memory into a second
silo disconnected from the MCP tools + web app). Building it requires (a) memory
REST endpoints on the web backend and (b) Go subcommands — its own spec/plan.

## Cleanup

Remove `src/internal/mcp_server/memory.py` (both non-importing examples); their
ideas live on in `tools/memory.py`, `src/internal/memory/service.py`, and
`src/internal/memory/tools.py`.

## Acceptance criteria

- MCP tools registered in `api.py` and callable.
- `update_memory_from_conversation` runs the thin tool-calling loop, reconciles
  memories (add/update/delete) from conversation + existing notes, and persists a
  trajectory (before/after + tool calls) it returns a summary of.
- `generate_user_profile` persists LLM-consolidated `{topic, subtopic, content}`;
  `get_user_profile` returns them.
- `consolidate_memories` deterministically dedups + resolves tagged conflicts and
  returns a report (no LLM).
- `search_memories` ranks memories with no external service / forced model
  download in tests.
- `pytest` green; `ruff` clean.
