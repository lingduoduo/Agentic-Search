"""User memory context for personalised chat responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.internal.db.store import AgenticSearchStore


@dataclass
class UserMemoryContext:
    """Holds memories fetched for the current user at the start of a chat turn."""

    user_id: str | None = None
    memories: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def without_memories(self) -> "UserMemoryContext":
        """Return a copy with the memories list cleared (used when memory injection is disabled)."""
        return UserMemoryContext(
            user_id=self.user_id, memories=[], extra=dict(self.extra)
        )

    def has_memories(self) -> bool:
        return bool(self.memories)


_memory_store: AgenticSearchStore | None = None


def _get_memory_store() -> AgenticSearchStore:
    global _memory_store
    if _memory_store is None:
        _memory_store = AgenticSearchStore(":memory:")
    return _memory_store


def set_memory_store(store: AgenticSearchStore | None) -> None:
    """Set the store used by module-level memory helpers."""
    global _memory_store
    _memory_store = store


def add_memory(user_id: str, memory_text: str) -> str | None:
    """Persist a new memory for *user_id*. Returns the new memory ID or None."""
    record = _get_memory_store().add_user_memory(user_id, memory_text)
    return record.id if record else None


def update_memory_at_index(user_id: str, index: int, new_text: str) -> str | None:
    """Replace the memory at *index* for *user_id*. Returns the memory ID or None."""
    record = _get_memory_store().update_user_memory_at_index(user_id, index, new_text)
    return record.id if record else None


def get_memories(user_id: str) -> list[str]:
    """Return the stored memories for *user_id*."""
    return _get_memory_store().get_user_memories(user_id)
