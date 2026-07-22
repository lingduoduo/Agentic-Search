# Conversation-Memory MCP Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native MCP memory tools — save a note, agentically curate memories from a conversation (add/update/delete via LLM tool-calling), consolidate deterministically, generate/read a structured user profile, and semantically search memories — all on the repo's own SQLite store, LLM, `ToolRegistry`, and e5 encoder.

**Architecture:** Thin MCP tool wrappers (`src/internal/mcp_server/tools/memory.py`) delegate to a testable, MCP-free service (`src/internal/memory/`) over `AgenticSearchStore`. Curation drives `OpenAICompatibleLLM.stream(tools=...)` (there is **no** `.invoke`) and dispatches internal add/update/delete `Tool`s through `ToolRegistry`. Profile generation uses `.complete()`. Search uses an injected e5 encoder with a lexical fallback.

**Tech Stack:** Python 3.12, SQLite (`sqlite3`), FastMCP (`fastmcp`), `OpenAICompatibleLLM`, `src/tools` `ToolRegistry`/`Tool`, sentence-transformers e5 (optional), pytest, ruff.

## Global Constraints

- Work on branch `feat/conversation-memory-mcp` (already checked out). Never commit to `main`.
- Spec: `docs/superpowers/specs/2026-07-22-conversation-memory-mcp-design.md`.
- Representation is **enhanced notes** (plain contextual text). No memory "modes".
- Tests must not download models or hit the network: curation/profile tests use a fake LLM; search tests use the lexical fallback or an injected fake encoder.
- Run tests with `pytest`; lint with `ruff check . --fix && ruff format .`.
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Constants: `DEFAULT_MEMORY_USER_ID = "default_user"`, `MAX_CURATION_TURNS = 6`, `MEMORY_GATHER_CHAR_BUDGET = 12000`.
- Deviation from spec wording: reuse the existing `store.list_sessions_for_user(user_id)` instead of adding `list_chat_sessions`.

---

### Task 1: Store — id-based memory accessors

**Files:**
- Modify: `src/internal/db/store.py` (add three methods near `get_user_memories`, ~line 2458)
- Test: `tests/unit/db/test_memory_accessors.py` (create)

**Interfaces:**
- Consumes: existing `_now`, `_new_id`, `_json_dumps`, `_json_loads`, `deterministic_capture`, `_row_to_user_memory`, `UserMemoryRecord`.
- Produces:
  - `update_user_memory(self, user_id: str, memory_id: str, new_text: str, metadata: dict | None = None) -> UserMemoryRecord | None`
  - `delete_user_memory(self, user_id: str, memory_id: str) -> bool`
  - `get_user_memory_records(self, user_id: str) -> list[UserMemoryRecord]`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/db/test_memory_accessors.py
from src.internal.db.store import AgenticSearchStore


def test_update_and_delete_memory_by_id():
    store = AgenticSearchStore(":memory:")
    a = store.add_user_memory("u1", "likes window seats")
    b = store.add_user_memory("u1", "lives in Beijing")

    # get_user_memory_records returns records with ids, active only
    ids = {r.id for r in store.get_user_memory_records("u1")}
    assert ids == {a.id, b.id}

    # update by id
    updated = store.update_user_memory("u1", a.id, "prefers aisle seats")
    assert updated is not None and updated.memory_text == "prefers aisle seats"

    # wrong user cannot update
    assert store.update_user_memory("other", a.id, "hacked") is None

    # delete by id is a soft delete (drops from active list)
    assert store.delete_user_memory("u1", b.id) is True
    remaining = [r.memory_text for r in store.get_user_memory_records("u1")]
    assert remaining == ["prefers aisle seats"]

    # deleting again returns False (already inactive)
    assert store.delete_user_memory("u1", b.id) is False
    store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/db/test_memory_accessors.py -v`
Expected: FAIL — `AttributeError: 'AgenticSearchStore' object has no attribute 'get_user_memory_records'`.

- [ ] **Step 3: Write minimal implementation**

Add these methods to `AgenticSearchStore` in `src/internal/db/store.py` immediately after `get_user_memories` (~line 2458):

```python
    def get_user_memory_records(self, user_id: str) -> list[UserMemoryRecord]:
        """Return active memory records (with ids) for a user in display order."""
        rows = self._conn.execute(
            """
            SELECT * FROM user_memories
            WHERE user_id = ? AND is_active = 1
            ORDER BY created_at, id
            """,
            (user_id,),
        ).fetchall()
        return [self._row_to_user_memory(row) for row in rows]

    def update_user_memory(
        self,
        user_id: str,
        memory_id: str,
        new_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> UserMemoryRecord | None:
        """Replace one active memory identified by id (scoped to user_id)."""
        if not new_text.strip():
            return None
        row = self._conn.execute(
            "SELECT * FROM user_memories WHERE id = ? AND user_id = ? AND is_active = 1",
            (memory_id, user_id),
        ).fetchone()
        if row is None:
            return None
        captured, capture_meta = deterministic_capture(new_text.strip())
        merged_meta = _json_loads(row["metadata_json"])
        merged_meta.update(metadata or {})
        merged_meta.update(capture_meta)
        self._conn.execute(
            "UPDATE user_memories SET memory_text = ?, metadata_json = ?, updated_at = ? WHERE id = ?",
            (captured, _json_dumps(merged_meta), _now(), memory_id),
        )
        self._conn.commit()
        updated = self._conn.execute(
            "SELECT * FROM user_memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return self._row_to_user_memory(updated)

    def delete_user_memory(self, user_id: str, memory_id: str) -> bool:
        """Soft-delete one active memory by id (scoped to user_id)."""
        cur = self._conn.execute(
            "UPDATE user_memories SET is_active = 0, updated_at = ? "
            "WHERE id = ? AND user_id = ? AND is_active = 1",
            (_now(), memory_id, user_id),
        )
        self._conn.commit()
        return cur.rowcount > 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/db/test_memory_accessors.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/internal/db/store.py tests/unit/db/test_memory_accessors.py
git commit -m "feat: id-based user-memory update/delete/list accessors

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Store — `user_profiles` table + accessors

**Files:**
- Modify: `src/internal/db/models.py` (add `UserProfileEntryRecord`)
- Modify: `src/internal/db/__init__.py` (export it)
- Modify: `src/internal/db/store.py` (`_init_schema` executescript + `_migrate_schema` executescript + two accessors)
- Test: `tests/unit/db/test_user_profiles.py` (create)

**Interfaces:**
- Produces:
  - `UserProfileEntryRecord(id, user_id, topic, subtopic, content, created_at=None, updated_at=None)`
  - `store.replace_user_profile(self, user_id: str, entries: list[dict]) -> list[UserProfileEntryRecord]` (each entry dict has `topic`, `subtopic`, `content`)
  - `store.get_user_profile(self, user_id: str) -> list[UserProfileEntryRecord]`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/db/test_user_profiles.py
from src.internal.db.store import AgenticSearchStore


def test_replace_and_get_profile_overwrites():
    store = AgenticSearchStore(":memory:")
    first = store.replace_user_profile(
        "u1",
        [{"topic": "work", "subtopic": "role", "content": "software engineer"}],
    )
    assert len(first) == 1 and first[0].topic == "work"

    # regeneration fully replaces the prior profile
    store.replace_user_profile(
        "u1",
        [
            {"topic": "home", "subtopic": "city", "content": "Shanghai"},
            {"topic": "food", "subtopic": "", "content": "likes Sichuan"},
        ],
    )
    got = store.get_user_profile("u1")
    assert [e.topic for e in got] == ["food", "home"]  # ordered by topic
    # entries with neither topic nor content are dropped
    store.replace_user_profile("u1", [{"topic": "", "subtopic": "", "content": ""}])
    assert store.get_user_profile("u1") == []
    store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/db/test_user_profiles.py -v`
Expected: FAIL — `AttributeError: ... has no attribute 'replace_user_profile'`.

- [ ] **Step 3a: Add the record dataclass**

In `src/internal/db/models.py`, after `UserMemoryRecord` (~line 63):

```python
@dataclass(slots=True)
class UserProfileEntryRecord:
    """One entry of an LLM-consolidated user profile."""

    id: str
    user_id: str
    topic: str
    subtopic: str
    content: str
    created_at: str | None = None
    updated_at: str | None = None
```

In `src/internal/db/__init__.py`, add `UserProfileEntryRecord` to both the `from .models import (...)` block and `__all__` (keep alphabetical among the memory/profile records).

- [ ] **Step 3b: Add the table to schema**

In `src/internal/db/store.py` `_init_schema`, inside the `executescript` string, after the `user_memories` index (~line 330) add:

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

In `_migrate_schema`, append the same `CREATE TABLE IF NOT EXISTS user_profiles (...)` + index inside the existing `self._conn.executescript("""...""")` block (the one that re-creates `user_memories`, ~line 367) so old DBs gain the table.

- [ ] **Step 3c: Add accessors**

Add to `AgenticSearchStore` (near the memory accessors). Note `UserProfileEntryRecord` must be importable — it is imported at the top of `store.py` from `.models`; add it to that import list.

```python
    def replace_user_profile(
        self, user_id: str, entries: list[dict[str, Any]]
    ) -> list[UserProfileEntryRecord]:
        """Atomically replace a user's profile with *entries* ({topic, subtopic, content})."""
        now = _now()
        self._conn.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
        out: list[UserProfileEntryRecord] = []
        for entry in entries:
            topic = str(entry.get("topic", "")).strip()
            subtopic = str(entry.get("subtopic", "")).strip()
            content = str(entry.get("content", "")).strip()
            if not topic and not content:
                continue
            rid = _new_id("prof")
            self._conn.execute(
                """
                INSERT INTO user_profiles
                    (id, user_id, topic, subtopic, content, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (rid, user_id, topic, subtopic, content, now, now),
            )
            out.append(
                UserProfileEntryRecord(
                    id=rid, user_id=user_id, topic=topic, subtopic=subtopic,
                    content=content, created_at=now, updated_at=now,
                )
            )
        self._conn.commit()
        return out

    def get_user_profile(self, user_id: str) -> list[UserProfileEntryRecord]:
        rows = self._conn.execute(
            "SELECT * FROM user_profiles WHERE user_id = ? ORDER BY topic, id",
            (user_id,),
        ).fetchall()
        return [
            UserProfileEntryRecord(
                id=r["id"], user_id=r["user_id"], topic=r["topic"],
                subtopic=r["subtopic"], content=r["content"],
                created_at=r["created_at"], updated_at=r["updated_at"],
            )
            for r in rows
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/db/test_user_profiles.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/internal/db/models.py src/internal/db/__init__.py src/internal/db/store.py tests/unit/db/test_user_profiles.py
git commit -m "feat: user_profiles table + replace/get accessors

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Store — `memory_trajectories` table + accessors

**Files:**
- Modify: `src/internal/db/models.py` (add `MemoryTrajectoryRecord`)
- Modify: `src/internal/db/__init__.py` (export it)
- Modify: `src/internal/db/store.py` (schema in `_init_schema` + `_migrate_schema` + two accessors + import)
- Test: `tests/unit/db/test_memory_trajectories.py` (create)

**Interfaces:**
- Produces:
  - `MemoryTrajectoryRecord(id, user_id, session_id, model, trajectory: dict, created_at=None)`
  - `store.add_memory_trajectory(self, user_id: str, *, session_id: str | None, model: str, trajectory: dict) -> MemoryTrajectoryRecord`
  - `store.list_memory_trajectories(self, user_id: str, limit: int = 20) -> list[MemoryTrajectoryRecord]`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/db/test_memory_trajectories.py
from src.internal.db.store import AgenticSearchStore


def test_trajectory_round_trip_newest_first():
    store = AgenticSearchStore(":memory:")
    store.add_memory_trajectory(
        "u1", session_id="s1", model="m",
        trajectory={"memory_before": [], "tool_calls": [{"name": "add_memory"}],
                    "memory_after": ["x"], "counts": {"add": 1}},
    )
    store.add_memory_trajectory("u1", session_id=None, model="m", trajectory={"counts": {}})
    got = store.list_memory_trajectories("u1")
    assert len(got) == 2
    assert got[0].session_id is None  # newest first
    assert got[1].trajectory["counts"] == {"add": 1}
    store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/db/test_memory_trajectories.py -v`
Expected: FAIL — `AttributeError: ... 'add_memory_trajectory'`.

- [ ] **Step 3a: Record dataclass + export**

In `src/internal/db/models.py`, after `UserProfileEntryRecord`:

```python
@dataclass(slots=True)
class MemoryTrajectoryRecord:
    """Audit record of one memory-curation run."""

    id: str
    user_id: str
    session_id: str | None
    model: str
    trajectory: JsonObject = field(default_factory=dict)
    created_at: str | None = None
```

Add `MemoryTrajectoryRecord` to the imports + `__all__` in `src/internal/db/__init__.py`.

- [ ] **Step 3b: Schema**

Add to `_init_schema` executescript (after the `user_profiles` block) and to the `_migrate_schema` executescript block:

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

- [ ] **Step 3c: Accessors**

Add `MemoryTrajectoryRecord` to the `.models` import at the top of `store.py`, then add:

```python
    def add_memory_trajectory(
        self, user_id: str, *, session_id: str | None, model: str, trajectory: dict[str, Any]
    ) -> MemoryTrajectoryRecord:
        rid = _new_id("mtraj")
        now = _now()
        self._conn.execute(
            """
            INSERT INTO memory_trajectories
                (id, user_id, session_id, model, trajectory_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (rid, user_id, session_id, model, json.dumps(trajectory, sort_keys=True), now),
        )
        self._conn.commit()
        return MemoryTrajectoryRecord(
            id=rid, user_id=user_id, session_id=session_id, model=model,
            trajectory=trajectory, created_at=now,
        )

    def list_memory_trajectories(
        self, user_id: str, limit: int = 20
    ) -> list[MemoryTrajectoryRecord]:
        rows = self._conn.execute(
            """
            SELECT * FROM memory_trajectories
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        return [
            MemoryTrajectoryRecord(
                id=r["id"], user_id=r["user_id"], session_id=r["session_id"],
                model=r["model"], trajectory=json.loads(r["trajectory_json"] or "{}"),
                created_at=r["created_at"],
            )
            for r in rows
        ]
```

(`json` is already imported at the top of `store.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/db/test_memory_trajectories.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/internal/db/models.py src/internal/db/__init__.py src/internal/db/store.py tests/unit/db/test_memory_trajectories.py
git commit -m "feat: memory_trajectories table + add/list accessors

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Memory service — save, search, consolidate (no LLM)

**Files:**
- Create: `src/internal/memory/__init__.py`
- Create: `src/internal/memory/service.py`
- Test: `tests/unit/memory/__init__.py` (empty), `tests/unit/memory/test_service_basic.py`

**Interfaces:**
- Consumes: `store.add_user_memory`, `store.get_user_memory_records`, `store.delete_user_memory`.
- Produces (module functions in `service.py`):
  - `save_memory(store, user_id: str, text: str) -> str | None` (returns new memory id)
  - `search_memories(store, user_id: str, query: str, max_results: int = 5, encoder=None) -> list[tuple[UserMemoryRecord, float]]`
  - `consolidate_memories(store, user_id: str, resolve_conflicts: bool = True) -> dict`
  - constants `DEFAULT_MEMORY_USER_ID`, `MAX_CURATION_TURNS`, `MEMORY_GATHER_CHAR_BUDGET`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/memory/test_service_basic.py
from src.internal.db.store import AgenticSearchStore
from src.internal.memory import service


def _store():
    return AgenticSearchStore(":memory:")


def test_save_and_lexical_search():
    store = _store()
    service.save_memory(store, "u1", "User enjoys hiking in the mountains")
    service.save_memory(store, "u1", "User is allergic to peanuts")
    hits = service.search_memories(store, "u1", "mountain hiking trip")
    assert hits and "hiking" in hits[0][0].memory_text
    store.close()


def test_search_with_injected_encoder():
    import numpy as np

    store = _store()
    service.save_memory(store, "u1", "alpha")
    service.save_memory(store, "u1", "omega")

    def fake_encoder(texts):
        # "query: alpha" ~ "passage: alpha": map by substring to orthogonal vectors
        vecs = []
        for t in texts:
            vecs.append([1.0, 0.0] if "alpha" in t else [0.0, 1.0])
        return np.array(vecs, dtype=np.float32)

    hits = service.search_memories(store, "u1", "alpha", encoder=fake_encoder)
    assert hits[0][0].memory_text == "alpha"
    store.close()


def test_consolidate_dedups_and_resolves_conflicts():
    store = _store()
    store.add_user_memory("u1", "lives in Beijing", metadata={"tags": ["home"]})
    store.add_user_memory("u1", "likes window seats", metadata={"tags": ["seat"]})
    store.add_user_memory("u1", "likes window seats", metadata={"tags": ["seat"]})  # dup
    store.add_user_memory("u1", "lives in Shanghai", metadata={"tags": ["home"]})  # conflict

    report = service.consolidate_memories(store, "u1")
    assert report["initial"] == 4
    assert report["duplicates_removed"] == 1
    assert report["conflicts_resolved"][0]["attribute"] == "home"
    assert report["conflicts_resolved"][0]["kept"] == "lives in Shanghai"
    assert report["final"] == 2
    texts = {r.memory_text for r in store.get_user_memory_records("u1")}
    assert texts == {"lives in Shanghai", "likes window seats"}
    store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/memory/test_service_basic.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.internal.memory'`.

- [ ] **Step 3: Implement**

Create `src/internal/memory/__init__.py` (empty). Create `src/internal/memory/service.py`:

```python
"""Conversation-memory service: MCP-free logic over AgenticSearchStore."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from src.internal.db.models import UserMemoryRecord

DEFAULT_MEMORY_USER_ID = "default_user"
MAX_CURATION_TURNS = 6
MEMORY_GATHER_CHAR_BUDGET = 12000

Encoder = Callable[[list[str]], Any]  # list[str] -> np.ndarray


def save_memory(store, user_id: str, text: str) -> str | None:
    record = store.add_user_memory(user_id, text)
    return record.id if record is not None else None


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^0-9a-z]+", text.lower()) if t]


def search_memories(
    store, user_id: str, query: str, max_results: int = 5, encoder: Encoder | None = None
) -> list[tuple[UserMemoryRecord, float]]:
    records = store.get_user_memory_records(user_id)
    if not records or not query.strip():
        return []
    if encoder is not None:
        import numpy as np

        matrix = encoder([f"passage: {r.memory_text}" for r in records])
        qvec = encoder([f"query: {query}"])
        sims = (np.asarray(qvec) @ np.asarray(matrix).T)[0]
        ranked = sorted(zip(records, sims), key=lambda x: float(x[1]), reverse=True)
        return [(r, float(s)) for r, s in ranked[:max_results]]
    # Lexical fallback: token-overlap, normalized by query length.
    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return []
    scored: list[tuple[UserMemoryRecord, float]] = []
    for r in records:
        overlap = len(q_tokens & set(_tokenize(r.memory_text)))
        if overlap:
            scored.append((r, overlap / len(q_tokens)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:max_results]


def _attribute(record: UserMemoryRecord) -> str | None:
    tags = record.metadata.get("tags") if isinstance(record.metadata, dict) else None
    if isinstance(tags, list) and tags and isinstance(tags[0], str) and tags[0].strip():
        return tags[0].strip()
    return None


def consolidate_memories(store, user_id: str, resolve_conflicts: bool = True) -> dict[str, Any]:
    records = store.get_user_memory_records(user_id)
    report: dict[str, Any] = {
        "initial": len(records),
        "duplicates_removed": 0,
        "conflicts_resolved": [],
        "final": 0,
    }

    def newest_first(rs: list[UserMemoryRecord]) -> list[UserMemoryRecord]:
        return sorted(rs, key=lambda r: (r.updated_at or "", r.id), reverse=True)

    # Exact-content dedup, keeping the newest.
    seen: set[str] = set()
    keep: list[UserMemoryRecord] = []
    remove: list[UserMemoryRecord] = []
    for r in newest_first(records):
        key = r.memory_text.strip().lower()
        if key in seen:
            remove.append(r)
            report["duplicates_removed"] += 1
            continue
        seen.add(key)
        keep.append(r)

    if resolve_conflicts:
        by_attr: dict[str, list[UserMemoryRecord]] = {}
        untagged: list[UserMemoryRecord] = []
        for r in keep:
            attr = _attribute(r)
            (untagged if attr is None else by_attr.setdefault(attr, [])).append(r)
        resolved = list(untagged)
        for attr, group in by_attr.items():
            ordered = newest_first(group)
            resolved.append(ordered[0])
            if len(ordered) > 1:
                remove.extend(ordered[1:])
                report["conflicts_resolved"].append(
                    {
                        "attribute": attr,
                        "kept": ordered[0].memory_text,
                        "superseded": [r.memory_text for r in ordered[1:]],
                    }
                )
        keep = resolved

    for r in remove:
        store.delete_user_memory(user_id, r.id)
    report["final"] = len(keep)
    return report
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/memory/test_service_basic.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/internal/memory/__init__.py src/internal/memory/service.py tests/unit/memory/
git commit -m "feat: memory service — save, search (lexical + encoder), consolidate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Internal memory Tools + registry factory

**Files:**
- Create: `src/internal/memory/tools.py`
- Test: `tests/unit/memory/test_memory_tools.py`

**Interfaces:**
- Consumes: `src.tools.base.Tool`/`ToolSchema`, `src.tools.registry.ToolRegistry`, store id-based accessors.
- Produces: `build_memory_registry(store, user_id: str) -> tuple[ToolRegistry, dict, list[dict]]` — returns `(registry, counts, tool_schemas)` where `counts` is `{"add": int, "update": int, "delete": int}` mutated as tools run, and `tool_schemas` is the list of `schema.to_dict()` for the LLM `tools=` param.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/memory/test_memory_tools.py
import asyncio

from src.internal.db.store import AgenticSearchStore
from src.internal.memory.tools import build_memory_registry


def test_memory_tools_mutate_store_and_count():
    store = AgenticSearchStore(":memory:")
    registry, counts, schemas = build_memory_registry(store, "u1")
    assert {s["function"]["name"] for s in schemas} == {"add_memory", "update_memory", "delete_memory"}

    async def run():
        resp, _raw, errs = await registry.invoke("add_memory", {"content": "likes tea"})
        assert not errs
        mem_id = store.get_user_memory_records("u1")[0].id
        await registry.invoke("update_memory", {"memory_id": mem_id, "content": "likes green tea"})
        await registry.invoke("delete_memory", {"memory_id": mem_id})

    asyncio.run(run())
    assert counts == {"add": 1, "update": 1, "delete": 1}
    assert store.get_user_memory_records("u1") == []
    store.close()


def test_add_memory_schema_validation_rejects_missing_content():
    store = AgenticSearchStore(":memory:")
    registry, counts, _ = build_memory_registry(store, "u1")

    async def run():
        _resp, _raw, errs = await registry.invoke("add_memory", {})
        return errs

    errs = asyncio.run(run())
    assert errs  # schema requires "content"
    assert counts["add"] == 0
    store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/memory/test_memory_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.internal.memory.tools'`.

- [ ] **Step 3: Implement**

Create `src/internal/memory/tools.py`:

```python
"""Internal add/update/delete memory Tools dispatched by the curation loop."""

from __future__ import annotations

from typing import Any

from src.tools.base import Tool, ToolEffect, ToolSchema
from src.tools.registry import ToolRegistry


class _AddMemoryTool(Tool):
    def __init__(self, store, user_id: str, counts: dict[str, int]) -> None:
        self._store, self._user_id, self._counts = store, user_id, counts

    @property
    def name(self) -> str:
        return "add_memory"

    @property
    def effect(self) -> ToolEffect:
        return ToolEffect.SIDE_EFFECTING

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="add_memory",
            description="Store a new durable memory (one contextual sentence) about the user.",
            parameters={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The memory to store."}
                },
                "required": ["content"],
            },
        )

    async def execute(self, instance_id: str, arguments: dict[str, Any]):
        del instance_id
        record = self._store.add_user_memory(self._user_id, str(arguments.get("content", "")))
        if record is None:
            return "empty content; nothing added", None, {}
        self._counts["add"] += 1
        return f"added memory {record.id}", record, {}


class _UpdateMemoryTool(Tool):
    def __init__(self, store, user_id: str, counts: dict[str, int]) -> None:
        self._store, self._user_id, self._counts = store, user_id, counts

    @property
    def name(self) -> str:
        return "update_memory"

    @property
    def effect(self) -> ToolEffect:
        return ToolEffect.SIDE_EFFECTING

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="update_memory",
            description="Replace the content of an existing memory identified by memory_id.",
            parameters={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["memory_id", "content"],
            },
        )

    async def execute(self, instance_id: str, arguments: dict[str, Any]):
        del instance_id
        updated = self._store.update_user_memory(
            self._user_id, str(arguments.get("memory_id", "")), str(arguments.get("content", ""))
        )
        if updated is None:
            return "memory not found", None, {}
        self._counts["update"] += 1
        return f"updated memory {updated.id}", updated, {}


class _DeleteMemoryTool(Tool):
    def __init__(self, store, user_id: str, counts: dict[str, int]) -> None:
        self._store, self._user_id, self._counts = store, user_id, counts

    @property
    def name(self) -> str:
        return "delete_memory"

    @property
    def effect(self) -> ToolEffect:
        return ToolEffect.SIDE_EFFECTING

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="delete_memory",
            description="Delete an outdated or incorrect memory identified by memory_id.",
            parameters={
                "type": "object",
                "properties": {"memory_id": {"type": "string"}},
                "required": ["memory_id"],
            },
        )

    async def execute(self, instance_id: str, arguments: dict[str, Any]):
        del instance_id
        ok = self._store.delete_user_memory(self._user_id, str(arguments.get("memory_id", "")))
        if not ok:
            return "memory not found", None, {}
        self._counts["delete"] += 1
        return "deleted memory", None, {}


def build_memory_registry(store, user_id: str) -> tuple[ToolRegistry, dict[str, int], list[dict]]:
    counts = {"add": 0, "update": 0, "delete": 0}
    tools = [
        _AddMemoryTool(store, user_id, counts),
        _UpdateMemoryTool(store, user_id, counts),
        _DeleteMemoryTool(store, user_id, counts),
    ]
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    schemas = [tool.schema.to_dict() for tool in tools]
    return registry, counts, schemas
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/memory/test_memory_tools.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/internal/memory/tools.py tests/unit/memory/test_memory_tools.py
git commit -m "feat: internal add/update/delete memory tools + registry factory

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Memory service — agentic curation (stream loop + trajectory)

**Files:**
- Modify: `src/internal/memory/service.py` (add curation + prompts + stream helper)
- Test: `tests/unit/memory/test_curation.py`

**Interfaces:**
- Consumes: `build_memory_registry`, `store.list_sessions_for_user`, `store.list_chat_messages`, `store.get_user_memory_records`, `store.add_memory_trajectory`; an `llm` object exposing `llm.stream(prompt, tools=..., max_tokens=...)` yielding chunks with `chunk.choice.delta.content` and `chunk.choice.delta.tool_calls[i]` (`.id`, `.index`, `.function.name`, `.function.arguments`) and `llm.config.model_name`.
- Produces:
  - `async curate_from_conversation(store, user_id: str, llm, session_id: str | None = None, max_turns: int = MAX_CURATION_TURNS) -> dict`
  - `_stream_turn(llm, messages, schemas) -> tuple[str, list[dict]]` (each tool call dict: `{"id", "name", "arguments"}`)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/memory/test_curation.py
import asyncio

from src.internal.db.store import AgenticSearchStore
from src.internal.llm.model_response import (
    ChatCompletionDeltaToolCall,
    Delta,
    FunctionCall,
    ModelResponseStream,
    StreamingChoice,
)
from src.internal.memory import service


class _FakeConfig:
    model_name = "fake-model"


class _FakeLLM:
    """Yields scripted stream turns: a list of chunk-lists, one per stream() call."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.config = _FakeConfig()

    def stream(self, prompt, tools=None, tool_choice=None, max_tokens=None, **kwargs):
        return iter(self._turns.pop(0))


def _tool_chunk(index, call_id, name, arguments):
    return ModelResponseStream(
        id="x", created="0",
        choice=StreamingChoice(
            delta=Delta(
                tool_calls=[
                    ChatCompletionDeltaToolCall(
                        id=call_id, index=index,
                        function=FunctionCall(name=name, arguments=arguments),
                    )
                ]
            )
        ),
    )


def _text_chunk(text):
    return ModelResponseStream(
        id="x", created="0", choice=StreamingChoice(delta=Delta(content=text))
    )


def test_curation_applies_tool_calls_and_persists_trajectory():
    store = AgenticSearchStore(":memory:")
    session = store.create_chat_session(user_id="u1")
    store.add_chat_message(session.id, role="user", content="I just moved to Shanghai.")
    store.add_chat_message(session.id, role="assistant", content="Noted!")

    llm = _FakeLLM(
        turns=[
            [_tool_chunk(0, "c1", "add_memory", '{"content": "User moved to Shanghai"}')],
            [_text_chunk("STOP")],  # second turn: no tool calls -> loop ends
        ]
    )

    summary = asyncio.run(service.curate_from_conversation(store, "u1", llm))
    assert summary["status"] == "ok"
    assert summary["counts"]["add"] == 1
    texts = [r.memory_text for r in store.get_user_memory_records("u1")]
    assert texts == ["User moved to Shanghai"]

    traj = store.list_memory_trajectories("u1")
    assert len(traj) == 1
    assert traj[0].trajectory["counts"]["add"] == 1
    assert traj[0].trajectory["memory_after"] == ["User moved to Shanghai"]
    store.close()


def test_curation_empty_sources_returns_message():
    store = AgenticSearchStore(":memory:")
    llm = _FakeLLM(turns=[])
    summary = asyncio.run(service.curate_from_conversation(store, "nobody", llm))
    assert summary["status"] == "empty"
    store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/memory/test_curation.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'curate_from_conversation'`.

- [ ] **Step 3: Implement**

Append to `src/internal/memory/service.py`:

```python
import asyncio  # add to the existing imports at the top of the file

from src.internal.memory.tools import build_memory_registry  # add near top imports

_CURATION_SYSTEM = (
    "You maintain a user's long-term memory. Given a recent conversation and the "
    "user's current memories, reconcile them by calling add_memory, update_memory, "
    "and delete_memory. Add new durable facts/preferences, update changed ones, and "
    "delete outdated or contradicted ones. Keep each memory a single contextual "
    "sentence. Do NOT store secrets (passwords, PINs, full SSNs, or full card/account "
    "numbers). When there is nothing left to change, reply with STOP and no tool calls."
)

_CURATION_USER = (
    "Recent conversation:\n{conversation}\n\n"
    "Current memories (id: text):\n{memories}\n\n"
    "Update the memory set now using the tools."
)


def _gather_sources(store, user_id: str, session_id: str | None) -> str:
    sessions = (
        [store.get_chat_session(session_id)]
        if session_id
        else store.list_sessions_for_user(user_id)
    )
    lines: list[str] = []
    for sess in sessions:
        if sess is None:
            continue
        for msg in store.list_chat_messages(sess.id):
            lines.append(f"{msg.role.upper()}: {msg.content}")
    text = "\n".join(lines)
    return text[-MEMORY_GATHER_CHAR_BUDGET:]


def _format_memories(store, user_id: str) -> str:
    records = store.get_user_memory_records(user_id)
    if not records:
        return "(none)"
    return "\n".join(f"{r.id}: {r.memory_text}" for r in records)


def _stream_turn(llm, messages: list[dict], schemas: list[dict]) -> tuple[str, list[dict]]:
    content_parts: list[str] = []
    acc: dict[int, dict[str, str]] = {}
    for chunk in llm.stream(messages, tools=schemas, max_tokens=1024):
        delta = chunk.choice.delta
        if delta.content:
            content_parts.append(delta.content)
        for tcd in delta.tool_calls:
            slot = acc.setdefault(tcd.index, {"id": "", "name": "", "arguments": ""})
            if tcd.id:
                slot["id"] = tcd.id
            if tcd.function is not None:
                if tcd.function.name:
                    slot["name"] = tcd.function.name
                if tcd.function.arguments:
                    slot["arguments"] += tcd.function.arguments
    tool_calls = [acc[i] for i in sorted(acc)]
    return "".join(content_parts), tool_calls


async def curate_from_conversation(
    store, user_id: str, llm, session_id: str | None = None, max_turns: int = MAX_CURATION_TURNS
) -> dict[str, Any]:
    sources = _gather_sources(store, user_id, session_id)
    if not sources.strip():
        return {"status": "empty", "message": "no conversations or notes yet", "counts": {}}

    before = [r.memory_text for r in store.get_user_memory_records(user_id)]
    registry, counts, schemas = build_memory_registry(store, user_id)
    messages: list[dict] = [
        {"role": "system", "content": _CURATION_SYSTEM},
        {
            "role": "user",
            "content": _CURATION_USER.format(
                conversation=sources, memories=_format_memories(store, user_id)
            ),
        },
    ]
    tool_call_log: list[dict] = []
    for _ in range(max_turns):
        content, tool_calls = await asyncio.to_thread(_stream_turn, llm, messages, schemas)
        assistant: dict[str, Any] = {"role": "assistant", "content": content or None}
        if tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in tool_calls
            ]
        messages.append(assistant)
        if not tool_calls:
            break
        for tc in tool_calls:
            try:
                args = json.loads(tc["arguments"] or "{}")
            except json.JSONDecodeError as exc:
                result = f"error: invalid JSON arguments: {exc}"
            else:
                response, _raw, errors = await registry.invoke(tc["name"], args)
                result = response or ("; ".join(errors) if errors else "ok")
                tool_call_log.append({"name": tc["name"], "arguments": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    after = [r.memory_text for r in store.get_user_memory_records(user_id)]
    trajectory = {
        "memory_before": before,
        "tool_calls": tool_call_log,
        "memory_after": after,
        "counts": dict(counts),
    }
    record = store.add_memory_trajectory(
        user_id,
        session_id=session_id,
        model=getattr(getattr(llm, "config", None), "model_name", ""),
        trajectory=trajectory,
    )
    return {
        "status": "ok",
        "trajectory_id": record.id,
        "counts": dict(counts),
        "memory_count": len(after),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/memory/test_curation.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/internal/memory/service.py tests/unit/memory/test_curation.py
git commit -m "feat: agentic memory curation via LLM stream tool-calling + trajectory

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Memory service — profile generation + read

**Files:**
- Modify: `src/internal/memory/service.py`
- Test: `tests/unit/memory/test_profile.py`

**Interfaces:**
- Consumes: `store.get_user_memory_records`, `store.replace_user_profile`, `store.get_user_profile`; an `llm` with `.complete(prompt, max_tokens=..., temperature=...) -> str`.
- Produces:
  - `generate_user_profile(store, user_id: str, llm) -> list[UserProfileEntryRecord]`
  - `get_user_profile(store, user_id: str) -> list[UserProfileEntryRecord]`
  - `_parse_profile_json(text: str) -> list[dict]`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/memory/test_profile.py
from src.internal.db.store import AgenticSearchStore
from src.internal.memory import service


class _FakeLLM:
    def __init__(self, text):
        self._text = text

    def complete(self, prompt, **kwargs):
        return self._text


def test_generate_profile_persists_parsed_entries():
    store = AgenticSearchStore(":memory:")
    store.add_user_memory("u1", "User is a software engineer at TechCorp")
    llm = _FakeLLM(
        'Here is the profile: [{"topic": "work", "subtopic": "role", '
        '"content": "Software engineer at TechCorp"}] done'
    )
    entries = service.generate_user_profile(store, "u1", llm)
    assert len(entries) == 1 and entries[0].topic == "work"
    assert service.get_user_profile(store, "u1")[0].content == "Software engineer at TechCorp"
    store.close()


def test_generate_profile_malformed_json_yields_empty():
    store = AgenticSearchStore(":memory:")
    store.add_user_memory("u1", "x")
    entries = service.generate_user_profile(store, "u1", _FakeLLM("not json at all"))
    assert entries == []
    store.close()


def test_generate_profile_no_memories_clears_profile():
    store = AgenticSearchStore(":memory:")
    store.replace_user_profile("u1", [{"topic": "old", "subtopic": "", "content": "stale"}])
    entries = service.generate_user_profile(store, "u1", _FakeLLM("[]"))
    assert entries == []
    assert service.get_user_profile(store, "u1") == []
    store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/memory/test_profile.py -v`
Expected: FAIL — `AttributeError: ... 'generate_user_profile'`.

- [ ] **Step 3: Implement**

Append to `src/internal/memory/service.py`:

```python
from src.internal.db.models import UserProfileEntryRecord  # add near top imports

_PROFILE_SYSTEM = (
    "You build a concise structured profile of a user from their memories. "
    "Return ONLY a JSON array of objects with keys 'topic', 'subtopic', and "
    "'content'. Group related facts under a shared topic. No prose outside the array."
)

_PROFILE_USER = "User memories:\n{memories}\n\nReturn the JSON profile array now."


def _parse_profile_json(text: str) -> list[dict[str, str]]:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    out: list[dict[str, str]] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and (item.get("topic") or item.get("content")):
                out.append(
                    {
                        "topic": str(item.get("topic", "")),
                        "subtopic": str(item.get("subtopic", "")),
                        "content": str(item.get("content", "")),
                    }
                )
    return out


def generate_user_profile(store, user_id: str, llm) -> list[UserProfileEntryRecord]:
    memories = [r.memory_text for r in store.get_user_memory_records(user_id)]
    if not memories:
        return store.replace_user_profile(user_id, [])
    prompt = [
        {"role": "system", "content": _PROFILE_SYSTEM},
        {"role": "user", "content": _PROFILE_USER.format(
            memories="\n".join(f"- {m}" for m in memories))},
    ]
    raw = llm.complete(prompt, max_tokens=800, temperature=0.0)
    text = raw if isinstance(raw, str) else getattr(raw, "text", "")
    return store.replace_user_profile(user_id, _parse_profile_json(text))


def get_user_profile(store, user_id: str) -> list[UserProfileEntryRecord]:
    return store.get_user_profile(user_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/memory/test_profile.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/internal/memory/service.py tests/unit/memory/test_profile.py
git commit -m "feat: LLM user-profile generation + read over curated memories

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: MCP tools + registration + stub removal

**Files:**
- Create: `src/internal/mcp_server/tools/memory.py`
- Modify: `src/internal/mcp_server/api.py` (add `from .tools import memory`)
- Delete: `src/internal/mcp_server/memory.py` (the non-importing examples)
- Test: `tests/unit/test_mcp_memory_tools.py`

**Interfaces:**
- Consumes: `service.*`, `require_access_token`, `load_app_settings`, `OpenAICompatibleLLM`/`LLMConfig`, `build_e5_encoder`, `AgenticSearchStore`, `mcp_server`.
- Produces: six `@mcp_server.tool()` async functions returning `dict`: `save_memory`, `update_memory_from_conversation`, `generate_user_profile`, `get_user_profile`, `search_memories`, `consolidate_memories`; plus module helpers `_resolve_user_id() -> str`, `_get_store() -> AgenticSearchStore`, `_build_llm()`, `_maybe_encoder()`.

- [ ] **Step 1: Write the failing test**

The tools call `require_access_token()` (needs a request context) and env-configured deps, so the unit test exercises the module's pure helpers + import/registration, not the decorated tools over HTTP.

```python
# tests/unit/test_mcp_memory_tools.py
def test_memory_module_imports_and_registers_tools():
    # Importing the tools module must register the tools on the shared server
    from src.internal.mcp_server import api  # noqa: F401
    from src.internal.mcp_server.tools import memory  # noqa: F401

    assert hasattr(memory, "save_memory")
    assert hasattr(memory, "update_memory_from_conversation")
    assert hasattr(memory, "search_memories")


def test_resolve_user_id_defaults_without_token(monkeypatch):
    from src.internal.mcp_server.tools import memory
    from src.internal.memory.service import DEFAULT_MEMORY_USER_ID

    def _raise():
        raise ValueError("no token")

    monkeypatch.setattr(memory, "require_access_token", _raise)
    assert memory._resolve_user_id() == DEFAULT_MEMORY_USER_ID


def test_old_stub_removed():
    import importlib.util

    assert importlib.util.find_spec("src.internal.mcp_server.memory") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_mcp_memory_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.internal.mcp_server.tools.memory'`.

- [ ] **Step 3a: Implement the tools module**

Create `src/internal/mcp_server/tools/memory.py`:

```python
"""MCP tools for user-memory management (thin wrappers over memory.service)."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from typing import Any

from src.internal.configs.app_configs import load_app_settings
from src.internal.db.store import AgenticSearchStore
from src.internal.llm.interfaces import LLMConfig
from src.internal.llm.providers import OpenAICompatibleLLM
from src.internal.memory import service
from src.internal.memory.service import DEFAULT_MEMORY_USER_ID

from ..api import mcp_server
from ..utils import require_access_token

logger = logging.getLogger(__name__)

_STORE: AgenticSearchStore | None = None


def _get_store() -> AgenticSearchStore:
    global _STORE
    if _STORE is None:
        db_path = load_app_settings().services.web_db_path
        _STORE = AgenticSearchStore(db_path)
    return _STORE


def _resolve_user_id() -> str:
    try:
        token = require_access_token()
        sub = (getattr(token, "claims", None) or {}).get("sub")
        if isinstance(sub, str) and sub.strip():
            return sub.strip()
    except Exception:  # noqa: BLE001 — unauthenticated/local falls back
        pass
    return DEFAULT_MEMORY_USER_ID


def _build_llm() -> OpenAICompatibleLLM | None:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AGENTIC_SEARCH_LLM_API_KEY")
    if not api_key:
        return None
    return OpenAICompatibleLLM(
        LLMConfig(
            model_provider=os.getenv("AGENTIC_SEARCH_LLM_PROVIDER", "openai"),
            model_name=os.getenv("AGENTIC_SEARCH_LLM_MODEL", "gpt-4o-mini"),
            api_key=api_key,
            api_base=os.getenv("AGENTIC_SEARCH_LLM_API_BASE"),
        )
    )


def _maybe_encoder():
    if os.getenv("AGENTIC_SEARCH_MEMORY_SEMANTIC", "").lower() not in {"1", "true", "yes"}:
        return None
    try:
        from src.internal.servers.retrieval.hybrid import build_e5_encoder

        return build_e5_encoder(device=os.getenv("AGENTIC_SEARCH_MEMORY_EMBED_DEVICE", "cpu"))
    except Exception as exc:  # noqa: BLE001 — fall back to lexical
        logger.warning("Memory e5 encoder unavailable, using lexical search: %s", exc)
        return None


@mcp_server.tool()
async def save_memory(text: str) -> dict[str, Any]:
    """Save one explicit long-term memory (a contextual sentence) for the user."""
    try:
        memory_id = service.save_memory(_get_store(), _resolve_user_id(), text)
        if memory_id is None:
            return {"status": "empty", "message": "content was empty; nothing saved"}
        return {"status": "ok", "memory_id": memory_id}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": str(exc)}


@mcp_server.tool()
async def update_memory_from_conversation(session_id: str | None = None) -> dict[str, Any]:
    """Read the user's conversation(s) + memories and reconcile memories via the LLM."""
    llm = _build_llm()
    if llm is None:
        return {"status": "error", "message": "no LLM configured (set OPENAI_API_KEY)"}
    try:
        return await service.curate_from_conversation(
            _get_store(), _resolve_user_id(), llm, session_id=session_id
        )
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": str(exc)}


@mcp_server.tool()
async def generate_user_profile() -> dict[str, Any]:
    """Consolidate the user's memories into a structured {topic, subtopic, content} profile."""
    llm = _build_llm()
    if llm is None:
        return {"status": "error", "message": "no LLM configured (set OPENAI_API_KEY)"}
    try:
        entries = service.generate_user_profile(_get_store(), _resolve_user_id(), llm)
        return {"status": "ok", "profile": [asdict(e) for e in entries]}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": str(exc)}


@mcp_server.tool()
async def get_user_profile() -> dict[str, Any]:
    """Return the persisted structured user profile."""
    try:
        entries = service.get_user_profile(_get_store(), _resolve_user_id())
        return {"status": "ok", "profile": [asdict(e) for e in entries]}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": str(exc)}


@mcp_server.tool()
async def search_memories(query: str, max_results: int = 5) -> dict[str, Any]:
    """Semantically (or lexically) search the user's memories."""
    try:
        hits = service.search_memories(
            _get_store(), _resolve_user_id(), query, max_results=max_results,
            encoder=_maybe_encoder(),
        )
        return {
            "status": "ok",
            "results": [{"id": r.id, "text": r.memory_text, "score": s} for r, s in hits],
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": str(exc)}


@mcp_server.tool()
async def consolidate_memories(resolve_conflicts: bool = True) -> dict[str, Any]:
    """Deterministically dedup + resolve tagged conflicts in the user's memories."""
    try:
        report = service.consolidate_memories(
            _get_store(), _resolve_user_id(), resolve_conflicts=resolve_conflicts
        )
        return {"status": "ok", "report": report}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": str(exc)}
```

- [ ] **Step 3b: Register + remove the stub**

In `src/internal/mcp_server/api.py`, in the "Import tools ... AFTER mcp_server is created" block, add:

```python
from .tools import memory  # noqa: E402, F401
```

Delete the old example stub:

```bash
git rm src/internal/mcp_server/memory.py
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_mcp_memory_tools.py -v`
Expected: PASS (all three tests).

- [ ] **Step 5: Commit**

```bash
git add src/internal/mcp_server/tools/memory.py src/internal/mcp_server/api.py tests/unit/test_mcp_memory_tools.py
git commit -m "feat: MCP memory tools (save/curate/profile/search/consolidate); drop stub

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Full-suite verification + lint

**Files:** none (verification only)

- [ ] **Step 1: Run the memory + db + mcp suites**

Run: `pytest tests/unit/memory/ tests/unit/db/ tests/unit/test_mcp_memory_tools.py tests/unit/test_mcp_server.py -v`
Expected: all PASS.

- [ ] **Step 2: Run the broader suite to catch regressions**

Run: `pytest tests/unit/db/ tests/unit/ -q -k "memory or mcp or store or profile"`
Expected: PASS.

- [ ] **Step 3: Lint + format**

Run: `ruff check . --fix && ruff format .`
Expected: "All checks passed!" and files formatted. Re-run the Step 1 suite if ruff changed anything.

- [ ] **Step 4: Commit any lint fixups**

```bash
git add -A
git commit -m "chore: ruff for conversation-memory MCP tools

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Post-implementation

After all tasks pass, push the branch and open a PR against `main` (per project workflow), titled e.g. *feat: native conversation-memory MCP tools (agentic curation + profile + consolidate)*, linking the spec and this plan. The CLI (deliverable 2) is a separate follow-on spec — do not start it here.
