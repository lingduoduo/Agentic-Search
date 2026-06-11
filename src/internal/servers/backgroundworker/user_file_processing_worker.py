"""User file processing worker: indexes user-uploaded files and syncs projects.

Responsibilities
----------------
- IndexUserFileTask    — run a single uploaded file through the indexing pipeline
- SyncProjectTask      — walk a directory connector and (re-)index changed files

Architecture: a ``ThreadPoolExecutor`` sized for I/O-bound file work (default 4).
Each task reads files and calls ``DocprocessingWorker.process_batch()``, reusing
all chunking / embedding / vector-DB machinery already in that worker.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union
import concurrent.futures

from src.internal.connectors.basic import LocalFileConnector
from src.internal.connectors.models import Document
from src.internal.db.models import ConnectorConfig
from src.internal.db.store import AgenticSearchStore
from src.internal.servers.backgroundworker.docprocessing import (
    DocprocessingConfig,
    DocprocessingWorker,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Task dataclasses                                                    #
# ------------------------------------------------------------------ #


@dataclass
class IndexUserFileTask:
    """Index a single user-uploaded file.

    ``file_path`` must be readable.  ``document_id`` overrides the default
    (the resolved path string).  Extra ``metadata`` is merged into the
    document record.
    """

    file_path: str | Path
    document_id: str | None = None
    connector_id: str | None = None
    metadata: dict | None = None


@dataclass
class SyncProjectTask:
    """Walk *project_dir* and (re-)index every supported file.

    A connector record keyed by *connector_id* is upserted in the store so
    downstream workers can reference indexed documents by their source.
    """

    connector_id: str
    project_dir: str | Path


UserFileTask = Union[IndexUserFileTask, SyncProjectTask]


# ------------------------------------------------------------------ #
# Config / Result                                                     #
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class UserFileProcessingConfig:
    """Runtime configuration for the user file processing worker."""

    max_workers: int = 4
    processing: DocprocessingConfig = field(default_factory=DocprocessingConfig)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "UserFileProcessingConfig":
        source: dict[str, str] = env if env is not None else dict(os.environ)
        return cls(
            max_workers=_env_int(source, "AGENTIC_SEARCH_NUM_USER_FILE_WORKERS", 4),
        )


@dataclass
class UserFileProcessingResult:
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_documents: int = 0
    total_chunks: int = 0
    errors: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ #
# Worker                                                              #
# ------------------------------------------------------------------ #


class UserFileProcessingWorker:
    """Processes user-uploaded files and directory-based project syncs.

    Usage::

        worker = UserFileProcessingWorker(
            store=store,
            config=UserFileProcessingConfig.from_env(),
        )
        result = worker.run([
            IndexUserFileTask("/uploads/report.md", connector_id="user-uploads"),
            SyncProjectTask("project-1", "/workspace/project-1"),
        ])
    """

    def __init__(
        self,
        store: AgenticSearchStore,
        *,
        config: UserFileProcessingConfig | None = None,
    ) -> None:
        self._store = store
        self._config = config or UserFileProcessingConfig.from_env()
        self._processor = DocprocessingWorker(
            store=store,
            config=self._config.processing,
        )

    # ------------------------------------------------------------------ #
    # Public entry points                                                  #
    # ------------------------------------------------------------------ #

    def run(self, tasks: Iterable[UserFileTask]) -> UserFileProcessingResult:
        """Execute *tasks* concurrently and return aggregated results."""
        overall = UserFileProcessingResult()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._config.max_workers,
            thread_name_prefix="user-file",
        ) as pool:
            futures = {pool.submit(self._execute, task): task for task in tasks}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    doc_count, chunk_count = fut.result()
                    overall.tasks_completed += 1
                    overall.total_documents += doc_count
                    overall.total_chunks += chunk_count
                except Exception as exc:
                    overall.tasks_failed += 1
                    overall.errors.append(str(exc))
                    logger.exception(
                        "User file task failed: %s", type(futures[fut]).__name__
                    )
        return overall

    # ------------------------------------------------------------------ #
    # Individual task handlers (also usable directly)                     #
    # ------------------------------------------------------------------ #

    def index_file(self, task: IndexUserFileTask) -> tuple[int, int]:
        """Read and index one file. Returns (documents_processed, chunks_written)."""
        path = Path(task.file_path).expanduser().resolve()
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            logger.warning("index_file: empty file skipped: %s", path)
            return 0, 0

        doc_id = task.document_id or str(path)
        doc = Document(
            id=doc_id,
            title=path.name,
            url=path.as_uri(),
            contents=text,
            metadata={
                "source": "user_upload",
                "path": str(path),
                **(task.metadata or {}),
            },
        )
        proc_result = self._processor.process_batch([doc])
        logger.debug(
            "index_file: %s chunks=%d failures=%d",
            path.name,
            proc_result.total_chunks,
            len(proc_result.failures),
        )
        return proc_result.total_documents, proc_result.total_chunks

    def sync_project(self, task: SyncProjectTask) -> tuple[int, int]:
        """Walk *project_dir* and index all supported files.

        Upserts a connector record so documents are associated with
        *connector_id* in the store.
        """
        project_dir = Path(task.project_dir).expanduser().resolve()
        if not project_dir.is_dir():
            raise ValueError(f"project_dir does not exist: {project_dir}")

        # Ensure connector record exists
        self._store.upsert_connector(
            ConnectorConfig(
                id=task.connector_id,
                name=f"project:{project_dir.name}",
                source="local_file",
                config={"path": str(project_dir)},
                enabled=True,
                metadata={},
            )
        )

        connector = LocalFileConnector(
            paths=[project_dir],
            batch_size=32,
        )
        proc_cfg = DocprocessingConfig(
            max_workers=1,
            connector_id=task.connector_id,
        )
        proc = DocprocessingWorker(store=self._store, config=proc_cfg)

        total_docs = total_chunks = 0
        for batch in connector.load_from_state():
            documents = [item for item in batch if isinstance(item, Document)]
            if not documents:
                continue
            result = proc.process_batch(documents)
            total_docs += result.total_documents
            total_chunks += result.total_chunks

        logger.info(
            "sync_project: connector=%r dir=%s docs=%d chunks=%d",
            task.connector_id,
            project_dir,
            total_docs,
            total_chunks,
        )
        return total_docs, total_chunks

    # ------------------------------------------------------------------ #
    # Dispatch                                                             #
    # ------------------------------------------------------------------ #

    def _execute(self, task: UserFileTask) -> tuple[int, int]:
        if isinstance(task, IndexUserFileTask):
            return self.index_file(task)
        if isinstance(task, SyncProjectTask):
            return self.sync_project(task)
        raise TypeError(f"Unknown user file task type: {type(task)}")


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
    "IndexUserFileTask",
    "SyncProjectTask",
    "UserFileProcessingConfig",
    "UserFileProcessingResult",
    "UserFileProcessingWorker",
    "UserFileTask",
]
