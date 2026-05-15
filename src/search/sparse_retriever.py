"""Compatibility alias for :mod:`src.retrieval.sparse_retriever`."""

from __future__ import annotations

import sys

from ..retrieval import sparse_retriever as _impl

sys.modules[__name__] = _impl
