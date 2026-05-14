"""Top-level package for Agentic-Search."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:
    try:
        return getattr(import_module(".agent_loop", __name__), name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
