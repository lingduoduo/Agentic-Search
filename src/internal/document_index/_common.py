"""Shared plumbing for the indexing pipeline: optional-dependency guards and
the long-running-job heartbeat interface.

Kept as a leaf module (no internal imports) so the chunking/embedding/faiss_io
modules can share it without creating import cycles. Heavy dependencies
(torch/faiss/transformers/tqdm) are imported lazily inside the ``_require_*``
helpers so importing this module never pulls them in.
"""

from __future__ import annotations

import os
from typing import Protocol

# Must be set before torch/faiss are imported to prevent an OpenMP conflict on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _require_torch():
    import torch

    return torch


def _require_faiss():
    import faiss

    return faiss


def _require_transformers():
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    return AutoConfig, AutoModel, AutoTokenizer


def _require_tqdm():
    from tqdm import tqdm

    return tqdm


class IndexingHeartbeatInterface(Protocol):
    """Reports progress and cancellation for long-running indexing jobs."""

    def should_stop(self) -> bool: ...

    def progress(self, tag: str, amount: int) -> None: ...


def _raise_if_indexing_stopped(
    callback: IndexingHeartbeatInterface | None,
    tag: str,
) -> None:
    if callback and callback.should_stop():
        raise RuntimeError(f"{tag}: stop signal detected")


def _report_indexing_progress(
    callback: IndexingHeartbeatInterface | None,
    tag: str,
    amount: int,
) -> None:
    if callback:
        callback.progress(tag, amount)
