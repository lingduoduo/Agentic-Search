"""Public API — delegates to the parent src package.

All symbols live in src.__init__. This module exists for backward compatibility
so that ``from src.agent_loop import X`` continues to work without changes.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_src = import_module("..", __name__)  # the src package
__all__ = _src.__all__


def __getattr__(name: str) -> Any:
    try:
        return getattr(_src, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
