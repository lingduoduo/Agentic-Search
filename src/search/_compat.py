"""Helpers for legacy :mod:`src.search` import paths."""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType


def alias_module(alias: str, target: str) -> ModuleType:
    """Make a legacy module name resolve to its current implementation."""
    module = import_module(target)
    sys.modules[alias] = module
    return module
