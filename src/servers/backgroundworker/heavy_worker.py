"""Heavy worker: thread-pool execution of resource-intensive background tasks.

Responsibilities
----------------
- connector_pruning       — delete indexed documents no longer present in the source
- doc_permissions_sync    — overwrite ACL for every document owned by a connector
- external_group_sync     — replace group membership from an external source
- csv_report_generation   — build and persist a usage-report ZIP via the store

All task types are plain dataclasses.  ``HeavyWorker.run()`` fans them out over
a ``ThreadPoolExecutor`` sized for resource-heavy work (default 2 threads).
Each task method is also callable directly for single-task use or testing.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Union

from src.db.models import DocumentPermission, GroupRecord
from src.db.store import AgenticSearchStore
from src.servers.reporting.generation import create_new_usage_report
from src.servers.reporting.models import UsageReportMetadata

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Task dataclasses                                                    #
# ------------------------------------------------------------------ #


@dataclass
class PruneConnectorTask:
    """Delete documents for *connector_id* that are absent from *current_doc_ids*.

    ``current_doc_ids`` is the authoritative set returned by the connector's
    latest slim-doc scan.  Any stored document whose id is not in this set is
    considered stale and removed.
    """

    connector_id: str
    current_doc_ids: set[str]


@dataclass
class SyncDocPermissionsTask:
    """Overwrite ACL entries for all documents owned by *connector_id*.

    ``permissions_by_doc`` maps document ID → list of permissions.  Documents
    already in the store but absent from the map are left untouched.
    """

    connector_id: str
    permissions_by_doc: dict[str, list[DocumentPermission]]


@dataclass
class SyncExternalGroupsTask:
    """Upsert group records and their membership from an external source.

    ``groups`` is the full authoritative list; each record replaces the stored
    membership for that group (create-or-replace semantics).
    """

    groups: list[GroupRecord]


@dataclass
class GenerateCsvReportTask:
    """Build a usage-report ZIP and persist it via the store."""

    requestor_id: str | None = None
    period_from: str | None = None
    period_to: str | None = None


HeavyTask = Union[
    PruneConnectorTask,
    SyncDocPermissionsTask,
    SyncExternalGroupsTask,
    GenerateCsvReportTask,
]


# ------------------------------------------------------------------ #
# Config / Result                                                     #
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class HeavyWorkerConfig:
    """Runtime configuration for the heavy worker."""

    max_workers: int = 2

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "HeavyWorkerConfig":
        source: dict[str, str] = env if env is not None else dict(os.environ)
        return cls(
            max_workers=_env_int(source, "AGENTIC_SEARCH_NUM_HEAVY_WORKERS", 2),
        )


@dataclass
class HeavyWorkerResult:
    tasks_completed: int = 0
    tasks_failed: int = 0
    errors: list[str] = field(default_factory=list)
    reports: list[UsageReportMetadata] = field(default_factory=list)


# ------------------------------------------------------------------ #
# Worker                                                              #
# ------------------------------------------------------------------ #


class HeavyWorker:
    """Thread-pool worker for resource-intensive background tasks.

    Usage::

        worker = HeavyWorker(store=store, config=HeavyWorkerConfig.from_env())
        result = worker.run([
            PruneConnectorTask("conn-1", current_ids),
            GenerateCsvReportTask(requestor_id="admin"),
        ])
        # result.reports contains UsageReportMetadata for each CSV task
    """

    def __init__(
        self,
        store: AgenticSearchStore,
        *,
        config: HeavyWorkerConfig | None = None,
    ) -> None:
        self._store = store
        self._config = config or HeavyWorkerConfig.from_env()

    # ------------------------------------------------------------------ #
    # Public entry points                                                  #
    # ------------------------------------------------------------------ #

    def run(self, tasks: Iterable[HeavyTask]) -> HeavyWorkerResult:
        """Execute *tasks* concurrently and return aggregated results."""
        result = HeavyWorkerResult()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._config.max_workers,
            thread_name_prefix="heavy-worker",
        ) as pool:
            futures = {pool.submit(self._execute, task): task for task in tasks}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    report = fut.result()
                    result.tasks_completed += 1
                    if report is not None:
                        result.reports.append(report)
                except Exception as exc:
                    result.tasks_failed += 1
                    result.errors.append(str(exc))
                    logger.exception(
                        "Heavy task failed: %s", type(futures[fut]).__name__
                    )
        return result

    # ------------------------------------------------------------------ #
    # Individual task handlers (also usable directly)                     #
    # ------------------------------------------------------------------ #

    def prune_connector(self, task: PruneConnectorTask) -> None:
        """Delete stored documents that are absent from *current_doc_ids*."""
        stored = self._store.list_documents(connector_id=task.connector_id)
        stale = [doc.id for doc in stored if doc.id not in task.current_doc_ids]
        for doc_id in stale:
            self._store.delete_document(doc_id)
        logger.info(
            "prune_connector: connector=%r stale=%d total=%d",
            task.connector_id,
            len(stale),
            len(stored),
        )

    def sync_doc_permissions(self, task: SyncDocPermissionsTask) -> None:
        """Overwrite ACL entries for documents in *permissions_by_doc*."""
        total = 0
        for doc_id, permissions in task.permissions_by_doc.items():
            for permission in permissions:
                self._store.grant_document_access(permission)
                total += 1
        logger.info(
            "sync_doc_permissions: connector=%r docs=%d entries=%d",
            task.connector_id,
            len(task.permissions_by_doc),
            total,
        )

    def sync_external_groups(self, task: SyncExternalGroupsTask) -> None:
        """Upsert all groups and their membership lists."""
        for group in task.groups:
            self._store.upsert_group(group)
        logger.info("sync_external_groups: upserted %d groups", len(task.groups))

    def generate_csv_report(self, task: GenerateCsvReportTask) -> UsageReportMetadata:
        """Generate a usage-report ZIP and persist it in the store."""
        report = create_new_usage_report(
            self._store,
            requestor_id=task.requestor_id,
            period_from=task.period_from,
            period_to=task.period_to,
        )
        logger.info("generate_csv_report: created %r", report.report_name)
        return report

    # ------------------------------------------------------------------ #
    # Dispatch                                                             #
    # ------------------------------------------------------------------ #

    def _execute(self, task: HeavyTask) -> UsageReportMetadata | None:
        if isinstance(task, PruneConnectorTask):
            self.prune_connector(task)
        elif isinstance(task, SyncDocPermissionsTask):
            self.sync_doc_permissions(task)
        elif isinstance(task, SyncExternalGroupsTask):
            self.sync_external_groups(task)
        elif isinstance(task, GenerateCsvReportTask):
            return self.generate_csv_report(task)
        else:
            raise TypeError(f"Unknown heavy task type: {type(task)}")
        return None


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #


def _env_int(env: dict[str, str], name: str, default: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


__all__ = [
    "GenerateCsvReportTask",
    "HeavyTask",
    "HeavyWorker",
    "HeavyWorkerConfig",
    "HeavyWorkerResult",
    "PruneConnectorTask",
    "SyncDocPermissionsTask",
    "SyncExternalGroupsTask",
]
