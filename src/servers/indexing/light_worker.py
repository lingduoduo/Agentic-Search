"""Light worker: high-concurrency execution of fast, metadata-only tasks.

Responsibilities
----------------
- vector_db_metadata_sync  — push updated ACL/metadata to the vector DB
- connector_deletion        — remove a connector and all its owned documents
- doc_permissions_upsert    — bulk-write document ACL entries
- checkpoint_cleanup        — discard old index-attempt records for a connector
- index_attempt_cleanup     — purge completed attempt rows older than N days

All task types are plain dataclasses.  ``LightWorker.run()`` fans them out over
a ``ThreadPoolExecutor`` and returns an aggregated ``LightWorkerResult``.
Each task method is also callable directly for single-task use or testing.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Union

from src.db.models import DocumentPermission
from src.db.store import AgenticSearchStore

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Task dataclasses                                                    #
# ------------------------------------------------------------------ #


@dataclass
class SyncVdbMetadataTask:
    """Push updated ACL/metadata for *document_ids* to the vector DB.

    ``metadata_sync_fn`` is called with the document IDs whose permissions
    changed.  Pass ``None`` to skip the vector DB push (SQL-only update).
    """

    document_ids: list[str]
    metadata_sync_fn: Callable[[list[str]], None] | None = None


@dataclass
class DeleteConnectorTask:
    """Delete *connector_id* together with all documents it owns."""

    connector_id: str


@dataclass
class UpsertDocPermissionsTask:
    """Bulk-write *permissions* to the document ACL table."""

    permissions: list[DocumentPermission]


@dataclass
class CleanupCheckpointsTask:
    """Keep only the *keep_last_n* most recent index attempts for *connector_id*."""

    connector_id: str
    keep_last_n: int = 5


@dataclass
class CleanupIndexAttemptsTask:
    """Delete finished index attempts older than *older_than_days* days."""

    older_than_days: int = 30


LightTask = Union[
    SyncVdbMetadataTask,
    DeleteConnectorTask,
    UpsertDocPermissionsTask,
    CleanupCheckpointsTask,
    CleanupIndexAttemptsTask,
]


# ------------------------------------------------------------------ #
# Config / Result                                                     #
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class LightWorkerConfig:
    """Runtime configuration for the light worker."""

    max_workers: int = 8

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "LightWorkerConfig":
        source: dict[str, str] = env if env is not None else dict(os.environ)
        return cls(
            max_workers=_env_int(source, "AGENTIC_SEARCH_NUM_LIGHT_WORKERS", 8),
        )


@dataclass
class LightWorkerResult:
    tasks_completed: int = 0
    tasks_failed: int = 0
    errors: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ #
# Worker                                                              #
# ------------------------------------------------------------------ #


class LightWorker:
    """High-concurrency worker for lightweight metadata operations.

    Usage::

        worker = LightWorker(store=store, config=LightWorkerConfig.from_env())
        result = worker.run([
            DeleteConnectorTask("old-connector"),
            CleanupIndexAttemptsTask(older_than_days=7),
            UpsertDocPermissionsTask(new_permissions),
        ])
    """

    def __init__(
        self,
        store: AgenticSearchStore,
        *,
        config: LightWorkerConfig | None = None,
    ) -> None:
        self._store = store
        self._config = config or LightWorkerConfig.from_env()

    # ------------------------------------------------------------------ #
    # Public entry points                                                  #
    # ------------------------------------------------------------------ #

    def run(self, tasks: Iterable[LightTask]) -> LightWorkerResult:
        """Execute *tasks* concurrently and return aggregated results."""
        result = LightWorkerResult()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._config.max_workers,
            thread_name_prefix="light-worker",
        ) as pool:
            futures = {pool.submit(self._execute, task): task for task in tasks}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    fut.result()
                    result.tasks_completed += 1
                except Exception as exc:
                    result.tasks_failed += 1
                    result.errors.append(str(exc))
                    logger.exception(
                        "Light task failed: %s", type(futures[fut]).__name__
                    )
        return result

    # ------------------------------------------------------------------ #
    # Individual task handlers (also usable directly)                     #
    # ------------------------------------------------------------------ #

    def sync_vdb_metadata(self, task: SyncVdbMetadataTask) -> None:
        """Re-read ACLs from SQL and push updated metadata to the vector DB."""
        if not task.document_ids:
            return
        changed: list[str] = []
        for doc_id in task.document_ids:
            permissions = self._store.get_document_permissions(doc_id)
            if permissions:
                changed.append(doc_id)
        if changed and task.metadata_sync_fn is not None:
            task.metadata_sync_fn(changed)
        logger.debug(
            "sync_vdb_metadata: %d/%d docs pushed",
            len(changed),
            len(task.document_ids),
        )

    def delete_connector(self, task: DeleteConnectorTask) -> None:
        """Delete connector and all its owned documents from the store."""
        deleted = self._store.delete_connector(task.connector_id)
        if deleted:
            logger.info("delete_connector: removed connector %r", task.connector_id)
        else:
            logger.warning(
                "delete_connector: connector %r not found", task.connector_id
            )

    def upsert_doc_permissions(self, task: UpsertDocPermissionsTask) -> None:
        """Bulk-write document permissions to the ACL table."""
        for permission in task.permissions:
            self._store.grant_document_access(permission)
        logger.debug("upsert_doc_permissions: wrote %d entries", len(task.permissions))

    def cleanup_checkpoints(self, task: CleanupCheckpointsTask) -> None:
        """Delete excess index-attempt rows for *connector_id*."""
        deleted = self._store.delete_old_index_attempts(
            task.connector_id, task.keep_last_n
        )
        logger.debug(
            "cleanup_checkpoints: connector=%r deleted=%d kept=%d",
            task.connector_id,
            deleted,
            task.keep_last_n,
        )

    def cleanup_index_attempts(self, task: CleanupIndexAttemptsTask) -> None:
        """Remove finished index attempts older than *older_than_days* days."""
        deleted = self._store.delete_stale_index_attempts(
            older_than_days=task.older_than_days
        )
        logger.debug(
            "cleanup_index_attempts: deleted=%d older_than=%d days",
            deleted,
            task.older_than_days,
        )

    # ------------------------------------------------------------------ #
    # Dispatch                                                             #
    # ------------------------------------------------------------------ #

    def _execute(self, task: LightTask) -> None:
        if isinstance(task, SyncVdbMetadataTask):
            self.sync_vdb_metadata(task)
        elif isinstance(task, DeleteConnectorTask):
            self.delete_connector(task)
        elif isinstance(task, UpsertDocPermissionsTask):
            self.upsert_doc_permissions(task)
        elif isinstance(task, CleanupCheckpointsTask):
            self.cleanup_checkpoints(task)
        elif isinstance(task, CleanupIndexAttemptsTask):
            self.cleanup_index_attempts(task)
        else:
            raise TypeError(f"Unknown light task type: {type(task)}")


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
    "CleanupCheckpointsTask",
    "CleanupIndexAttemptsTask",
    "DeleteConnectorTask",
    "LightTask",
    "LightWorker",
    "LightWorkerConfig",
    "LightWorkerResult",
    "SyncVdbMetadataTask",
    "UpsertDocPermissionsTask",
]
