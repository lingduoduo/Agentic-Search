"""Beat worker: pure-Python periodic task scheduler replacing Celery Beat.

This repo has no Celery / Redis dependency.  ``BeatWorker`` implements the
same scheduling contract with a simple event-loop:

  1. Maintain a list of ``ScheduledTask`` entries (name, interval, callable).
  2. On each ``tick()``, find every task whose next-due time has passed and
     call it in order.
  3. ``start()`` drives the loop, sleeping up to ``tick_secs`` between
     iterations.  ``stop()`` signals a clean exit.

Default schedule (mirrors the Celery Beat configuration in the spec):

  Task                    Interval
  -----------------------------------------------
  check_indexing          15 s  — pick up not_started index attempts
  check_connector_delete  20 s  — delete disabled connectors
  check_vdb_sync          20 s  — push ACL changes to vector DB
  check_pruning           20 s  — prune stale documents per connector
  run_monitoring          300 s — collect system health metrics
  run_cleanup             3600 s— purge old index-attempt rows

``DynamicTenantScheduler`` wraps ``BeatWorker`` with per-tenant store
injection, supporting multi-tenant deployments without modifying the core
scheduling logic.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from src.internal.db.store import AgenticSearchStore
from src.internal.servers.backgroundworker.heavy_worker import HeavyWorker
from src.internal.servers.backgroundworker.light_worker import (
    CleanupIndexAttemptsTask,
    DeleteConnectorTask,
    LightWorker,
    SyncVdbMetadataTask,
)
from src.internal.servers.backgroundworker.monitoring_worker import MonitoringWorker

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Scheduled task descriptor                                           #
# ------------------------------------------------------------------ #


@dataclass
class ScheduledTask:
    """One periodic task entry in the beat schedule."""

    name: str
    interval_secs: float
    fn: Callable[[], None]
    _next_due: float = field(default=0.0, init=False, repr=False)

    def is_due(self, now: float) -> bool:
        return now >= self._next_due

    def mark_ran(self, now: float) -> None:
        self._next_due = now + self.interval_secs


# ------------------------------------------------------------------ #
# Config                                                              #
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class BeatWorkerConfig:
    """Timing configuration for the beat worker."""

    tick_secs: float = 1.0
    indexing_check_interval_secs: float = 15.0
    connector_delete_interval_secs: float = 20.0
    vdb_sync_interval_secs: float = 20.0
    pruning_interval_secs: float = 20.0
    monitoring_interval_secs: float = 300.0
    cleanup_interval_secs: float = 3600.0
    cleanup_older_than_days: int = 7

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "BeatWorkerConfig":
        source: dict[str, str] = env if env is not None else dict(os.environ)
        return cls(
            tick_secs=float(source.get("AGENTIC_SEARCH_BEAT_TICK_SECS") or "1"),
            indexing_check_interval_secs=float(
                source.get("AGENTIC_SEARCH_BEAT_INDEXING_INTERVAL_SECS") or "15"
            ),
            connector_delete_interval_secs=float(
                source.get("AGENTIC_SEARCH_BEAT_CONNECTOR_DELETE_INTERVAL_SECS") or "20"
            ),
            vdb_sync_interval_secs=float(
                source.get("AGENTIC_SEARCH_BEAT_VDB_SYNC_INTERVAL_SECS") or "20"
            ),
            pruning_interval_secs=float(
                source.get("AGENTIC_SEARCH_BEAT_PRUNING_INTERVAL_SECS") or "20"
            ),
            monitoring_interval_secs=float(
                source.get("AGENTIC_SEARCH_BEAT_MONITORING_INTERVAL_SECS") or "300"
            ),
            cleanup_interval_secs=float(
                source.get("AGENTIC_SEARCH_BEAT_CLEANUP_INTERVAL_SECS") or "3600"
            ),
            cleanup_older_than_days=int(
                source.get("AGENTIC_SEARCH_BEAT_CLEANUP_OLDER_THAN_DAYS") or "7"
            ),
        )


# ------------------------------------------------------------------ #
# Worker                                                              #
# ------------------------------------------------------------------ #


class BeatWorker:
    """Pure-Python periodic task scheduler.

    Usage::

        worker = BeatWorker(
            store=store,
            light=LightWorker(store),
            heavy=HeavyWorker(store),
            monitoring=MonitoringWorker(store),
            indexing_check_fn=my_indexing_check,  # optional
            config=BeatWorkerConfig.from_env(),
        )
        worker.start()   # blocking; call stop() from another thread

    ``indexing_check_fn`` and ``pruning_check_fn`` are optional callbacks
    for operations that require connector-specific logic outside this module
    (e.g. instantiating the right connector class for a given source type).
    Omitting them silently skips those schedule slots.
    """

    def __init__(
        self,
        store: AgenticSearchStore,
        light: LightWorker,
        heavy: HeavyWorker,
        monitoring: MonitoringWorker,
        *,
        indexing_check_fn: Callable[[], None] | None = None,
        pruning_check_fn: Callable[[], None] | None = None,
        config: BeatWorkerConfig | None = None,
    ) -> None:
        self._store = store
        self._light = light
        self._heavy = heavy
        self._monitoring = monitoring
        self._indexing_check_fn = indexing_check_fn
        self._pruning_check_fn = pruning_check_fn
        self._config = config or BeatWorkerConfig.from_env()
        self._stop_event = threading.Event()
        self._schedule = self._build_schedule()

    # ------------------------------------------------------------------ #
    # Public entry points                                                  #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Run the scheduling loop until ``stop()`` is called.

        Blocks the calling thread.
        """
        logger.info(
            "BeatWorker starting (tick=%.1fs, tasks=%d)",
            self._config.tick_secs,
            len(self._schedule),
        )
        self._stop_event.clear()
        while not self._stop_event.is_set():
            self.tick()
            self._stop_event.wait(timeout=self._config.tick_secs)
        logger.info("BeatWorker stopped")

    def stop(self) -> None:
        """Signal the loop to exit after its current sleep."""
        self._stop_event.set()

    def tick(self) -> list[str]:
        """Run all tasks that are currently due. Returns names of tasks that ran."""
        now = time.monotonic()
        ran: list[str] = []
        for task in self._schedule:
            if task.is_due(now):
                try:
                    task.fn()
                except Exception:
                    logger.exception(
                        "BeatWorker task %r raised an exception", task.name
                    )
                task.mark_ran(now)
                ran.append(task.name)
        return ran

    # ------------------------------------------------------------------ #
    # Default task implementations                                         #
    # ------------------------------------------------------------------ #

    def _check_indexing(self) -> None:
        """Dispatch the external indexing-check callback if provided."""
        if self._indexing_check_fn is not None:
            self._indexing_check_fn()

    def _check_connector_delete(self) -> None:
        """Delete connectors that have been disabled in the store."""
        disabled = self._store.list_connectors(enabled=False)
        if not disabled:
            return
        tasks = [DeleteConnectorTask(c.id) for c in disabled]
        result = self._light.run(tasks)
        if result.tasks_failed:
            logger.warning("_check_connector_delete: %d failed", result.tasks_failed)

    def _check_vdb_sync(self) -> None:
        """Push ACL changes for recently-updated documents to the vector DB."""
        documents = self._store.list_documents()
        if not documents:
            return
        doc_ids = [d.id for d in documents]
        self._light.run([SyncVdbMetadataTask(doc_ids)])

    def _check_pruning(self) -> None:
        """Dispatch the external pruning callback if provided."""
        if self._pruning_check_fn is not None:
            self._pruning_check_fn()

    def _run_monitoring(self) -> None:
        """Collect system health metrics (and ship to cloud if configured)."""
        self._monitoring.run_once()

    def _run_cleanup(self) -> None:
        """Purge old finished index attempts."""
        self._light.run(
            [
                CleanupIndexAttemptsTask(
                    older_than_days=self._config.cleanup_older_than_days
                )
            ]
        )

    # ------------------------------------------------------------------ #
    # Schedule construction                                                #
    # ------------------------------------------------------------------ #

    def _build_schedule(self) -> list[ScheduledTask]:
        cfg = self._config
        return [
            ScheduledTask(
                "check_indexing", cfg.indexing_check_interval_secs, self._check_indexing
            ),
            ScheduledTask(
                "check_connector_delete",
                cfg.connector_delete_interval_secs,
                self._check_connector_delete,
            ),
            ScheduledTask(
                "check_vdb_sync", cfg.vdb_sync_interval_secs, self._check_vdb_sync
            ),
            ScheduledTask(
                "check_pruning", cfg.pruning_interval_secs, self._check_pruning
            ),
            ScheduledTask(
                "run_monitoring", cfg.monitoring_interval_secs, self._run_monitoring
            ),
            ScheduledTask("run_cleanup", cfg.cleanup_interval_secs, self._run_cleanup),
        ]


# ------------------------------------------------------------------ #
# DynamicTenantScheduler                                              #
# ------------------------------------------------------------------ #


class DynamicTenantScheduler:
    """Wraps ``BeatWorker`` with per-tenant store injection.

    Each tenant gets its own ``BeatWorker`` instance backed by an independent
    ``AgenticSearchStore``.  ``add_tenant`` / ``remove_tenant`` manage the
    registry at runtime.  A single background thread drives all per-tenant
    beat loops on a shared tick cadence.
    """

    def __init__(self, *, config: BeatWorkerConfig | None = None) -> None:
        self._config = config or BeatWorkerConfig.from_env()
        self._tenants: dict[str, BeatWorker] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def add_tenant(
        self,
        tenant_id: str,
        store: AgenticSearchStore,
        *,
        indexing_check_fn: Callable[[], None] | None = None,
        pruning_check_fn: Callable[[], None] | None = None,
    ) -> None:
        """Register *tenant_id* with its own store and optional callbacks."""
        light = LightWorker(store)
        heavy = HeavyWorker(store)
        monitoring = MonitoringWorker(store)
        worker = BeatWorker(
            store,
            light,
            heavy,
            monitoring,
            indexing_check_fn=indexing_check_fn,
            pruning_check_fn=pruning_check_fn,
            config=self._config,
        )
        with self._lock:
            self._tenants[tenant_id] = worker
        logger.info("DynamicTenantScheduler: added tenant %r", tenant_id)

    def remove_tenant(self, tenant_id: str) -> None:
        with self._lock:
            self._tenants.pop(tenant_id, None)
        logger.info("DynamicTenantScheduler: removed tenant %r", tenant_id)

    def start(self) -> None:
        """Drive all tenant beat workers in a single blocking loop."""
        logger.info("DynamicTenantScheduler starting")
        self._stop_event.clear()
        while not self._stop_event.is_set():
            with self._lock:
                workers = list(self._tenants.values())
            for worker in workers:
                try:
                    worker.tick()
                except Exception:
                    logger.exception("DynamicTenantScheduler: tick error")
            self._stop_event.wait(timeout=self._config.tick_secs)
        logger.info("DynamicTenantScheduler stopped")

    def stop(self) -> None:
        """Signal the loop to exit."""
        self._stop_event.set()


__all__ = [
    "BeatWorker",
    "BeatWorkerConfig",
    "DynamicTenantScheduler",
    "ScheduledTask",
]
