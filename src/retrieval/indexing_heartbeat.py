"""Heartbeat callback interface for long-running indexing jobs."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class IndexingHeartbeatInterface(ABC):
    """Callback interface for indexing entrypoints and pipeline stages."""

    @abstractmethod
    def should_stop(self) -> bool:
        """Signal to stop the in-flight indexing function."""

    @abstractmethod
    def progress(self, tag: str, amount: int) -> None:
        """Send progress updates or keep-alives to the caller.

        ``amount`` can be positive to indicate progress, or zero/negative to
        act as a keep-alive.
        """
