"""Compatibility alias for :mod:`src.model.generation`."""

from __future__ import annotations

import sys

from ..model import generation as _impl

sys.modules[__name__] = _impl
