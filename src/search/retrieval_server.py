"""Compatibility alias for :mod:`src.retrieval.servers.retrieval`."""

from __future__ import annotations

import sys

from ..retrieval.servers import retrieval as _impl

sys.modules[__name__] = _impl
