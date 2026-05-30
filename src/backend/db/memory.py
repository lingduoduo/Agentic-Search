"""User memory context for personalised chat responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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


def add_memory(user_id: str, memory_text: str) -> int | None:
    """Persist a new memory for *user_id*. Returns the new memory ID or None."""
    return None


def update_memory_at_index(user_id: str, index: int, new_text: str) -> int | None:
    """Replace the memory at *index* for *user_id*. Returns the memory ID or None."""
    return None


def get_memories(user_id: str) -> list[str]:
    """Return the stored memories for *user_id*."""
    return []
