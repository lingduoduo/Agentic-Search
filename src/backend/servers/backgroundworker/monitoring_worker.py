"""Monitoring worker: single-threaded system health and metrics collection.

Responsibilities
----------------
- Collect process memory (RSS via stdlib ``resource``)
- Query the store for queue depth (pending / in-progress index attempts)
- Count active connectors and indexed documents
- Optionally POST a JSON snapshot to a cloud data-plane URL (cloud health reporting)

Architecture: single thread by design — monitoring never needs parallelism.
``run_once()`` collects a ``SystemMetrics`` snapshot and optionally ships it.
``start()`` drives a blocking loop at the configured interval; ``stop()``
signals it to exit cleanly.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import urllib.request
from dataclasses import asdict, dataclass

from src.backend.db.store import AgenticSearchStore

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Metrics snapshot                                                    #
# ------------------------------------------------------------------ #


@dataclass
class SystemMetrics:
    """Point-in-time snapshot of process and system health."""

    process_memory_mb: float
    pending_index_attempts: int
    in_progress_index_attempts: int
    active_connectors: int
    total_documents: int
    timestamp: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ #
# Config                                                              #
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class MonitoringConfig:
    """Runtime configuration for the monitoring worker."""

    interval_secs: float = 300.0
    cloud_report_url: str | None = None
    cloud_report_timeout_secs: float = 5.0

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "MonitoringConfig":
        source: dict[str, str] = env if env is not None else dict(os.environ)
        return cls(
            interval_secs=float(
                source.get("AGENTIC_SEARCH_MONITORING_INTERVAL_SECS") or "300"
            ),
            cloud_report_url=source.get("AGENTIC_SEARCH_CLOUD_DATA_PLANE_URL") or None,
            cloud_report_timeout_secs=float(
                source.get("AGENTIC_SEARCH_MONITORING_CLOUD_TIMEOUT_SECS") or "5"
            ),
        )


# ------------------------------------------------------------------ #
# Worker                                                              #
# ------------------------------------------------------------------ #


class MonitoringWorker:
    """Single-threaded health monitor and metrics collector.

    Usage::

        worker = MonitoringWorker(store, config=MonitoringConfig.from_env())

        # One-shot
        metrics = worker.run_once()

        # Blocking loop (call stop() from another thread to exit)
        worker.start()
    """

    def __init__(
        self,
        store: AgenticSearchStore,
        *,
        config: MonitoringConfig | None = None,
    ) -> None:
        self._store = store
        self._config = config or MonitoringConfig.from_env()
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------ #
    # Public entry points                                                  #
    # ------------------------------------------------------------------ #

    def run_once(self) -> SystemMetrics:
        """Collect a metrics snapshot and ship it if a cloud URL is configured."""
        metrics = self._collect()
        if self._config.cloud_report_url:
            self._report_to_cloud(metrics)
        return metrics

    def start(self) -> None:
        """Run the monitoring loop until ``stop()`` is called.

        Blocks the calling thread.  Designed to be launched from a dedicated
        daemon thread or as the main loop of a monitoring process.
        """
        logger.info(
            "MonitoringWorker starting (interval=%.0fs)", self._config.interval_secs
        )
        self._stop_event.clear()
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception:
                logger.exception("MonitoringWorker.run_once failed")
            self._stop_event.wait(timeout=self._config.interval_secs)
        logger.info("MonitoringWorker stopped")

    def stop(self) -> None:
        """Signal the loop started by ``start()`` to exit after its current sleep."""
        self._stop_event.set()

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _collect(self) -> SystemMetrics:
        from datetime import datetime, timezone

        attempts = self._store.list_index_attempts()
        pending = sum(1 for a in attempts if a.status == "not_started")
        in_progress = sum(1 for a in attempts if a.status == "in_progress")

        connectors = self._store.list_connectors(enabled=True)
        documents = self._store.list_documents()

        metrics = SystemMetrics(
            process_memory_mb=_process_memory_mb(),
            pending_index_attempts=pending,
            in_progress_index_attempts=in_progress,
            active_connectors=len(connectors),
            total_documents=len(documents),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        logger.debug(
            "metrics: memory=%.1fMB pending=%d in_progress=%d "
            "connectors=%d documents=%d",
            metrics.process_memory_mb,
            metrics.pending_index_attempts,
            metrics.in_progress_index_attempts,
            metrics.active_connectors,
            metrics.total_documents,
        )
        return metrics

    def _report_to_cloud(self, metrics: SystemMetrics) -> None:
        url = self._config.cloud_report_url
        if not url:
            return
        payload = json.dumps(metrics.as_dict()).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                req, timeout=self._config.cloud_report_timeout_secs
            ):
                pass
            logger.debug("MonitoringWorker: metrics posted to %s", url)
        except Exception:
            logger.warning(
                "MonitoringWorker: failed to post metrics to %s", url, exc_info=True
            )


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #


def _process_memory_mb() -> float:
    """Return current RSS memory in MB using only stdlib."""
    try:
        import resource as _resource

        usage = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        # macOS returns bytes; Linux returns kilobytes
        if sys.platform == "darwin":
            return usage / (1024 * 1024)
        return usage / 1024
    except Exception:
        return 0.0


__all__ = [
    "MonitoringConfig",
    "MonitoringWorker",
    "SystemMetrics",
]
