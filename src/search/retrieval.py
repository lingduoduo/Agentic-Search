"""Compatibility alias for :mod:`src.retrieval.dense_retriever`."""

from __future__ import annotations

import sys

from ..retrieval import dense_retriever as _impl

sys.modules[__name__] = _impl
