"""Compatibility alias for :mod:`src.retrieval.index_builder`."""

from __future__ import annotations

import sys

from ..retrieval import index_builder as _impl

sys.modules[__name__] = _impl
