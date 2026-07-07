"""Tests for persisted user memory storage."""

from __future__ import annotations

from src.internal.db.memory import (
    add_memory,
    get_memories,
    set_memory_store,
    update_memory_at_index,
)
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
