"""Compatibility alias for :mod:`src.retrieval.rerank`."""

from __future__ import annotations

import sys

from ..retrieval import rerank as _impl

sys.modules[__name__] = _impl
