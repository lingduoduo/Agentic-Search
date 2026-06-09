"""Shared utility functions used across services."""

from __future__ import annotations

from typing import TypeVar

_T = TypeVar("_T")


def batch_list(items: list[_T], batch_size: int) -> list[list[_T]]:
    """Split a list into consecutive non-overlapping batches of at most batch_size."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
