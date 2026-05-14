"""Compatibility alias for :mod:`src.tools.search`."""

from __future__ import annotations

import sys

from ..tools import search as _impl

sys.modules[__name__] = _impl
