"""Docfetching worker: streams documents from connectors and dispatches batches
to docprocessing tasks in a thread pool.

Architecture:
- A daemon *feeder* thread drives the connector generator and puts items on a
  bounded queue.
- The main thread drains the queue, accumulates batches, and submits each full
  batch to a ThreadPoolExecutor for concurrent processing.
- The watchdog is implemented naturally: ``queue.get(timeout=watchdog_timeout_secs)``
  raises ``queue.Empty`` if the connector stalls.  The feeder thread is left as a
  daemon and abandoned — it will be collected when the process exits.  This avoids
  un-interruptible ``next()`` calls on stuck network I/O.
- For ``CheckpointedConnector``, the generator return value (the new checkpoint) is
  captured from ``StopIteration.value`` inside the feeder thread and forwarded via
  a sentinel queue message.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import queue
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from src.internal.connectors.interface import (
    CheckpointedConnector,
    LoadConnector,
    PollConnector,
    SecondsSinceUnixEpoch,
)
from src.internal.connectors.models import (
    ConnectorCheckpoint,
    ConnectorFailure,
    Document,
    HierarchyNode,
)
from src.internal.document_index.index_builder import IndexingHeartbeatInterface

logger = logging.getLogger(__name__)

DocProcessingFn = Callable[[list[Document]], Any]

_KIND_ITEM = "item"
_KIND_DONE = "done"
_KIND_ERROR = "error"


@dataclass(frozen=True)
class DocfetchingConfig:
    """Runtime configuration for the docfetching worker."""

    max_workers: int = 4
    batch_size: int = 100
    watchdog_timeout_secs: float = 300.0

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "DocfetchingConfig":
        source: dict[str, str] = env if env is not None else dict(os.environ)
        return cls(
            max_workers=_env_int(source, "AGENTIC_SEARCH_NUM_DOCFETCHING_WORKERS", 4),
            batch_size=_env_int(source, "AGENTIC_SEARCH_DOCFETCHING_BATCH_SIZE", 100),
            watchdog_timeout_secs=float(
                source.get("AGENTIC_SEARCH_DOCFETCHING_WATCHDOG_TIMEOUT_SECS") or "300"
            ),
        )


@dataclass
class DocfetchingResult:
    """Outcome of a single docfetching run."""

    batches_dispatched: int = 0
    total_documents: int = 0
    failures: list[ConnectorFailure] = field(default_factory=list)
    watchdog_triggered: bool = False
    final_checkpoint: ConnectorCheckpoint | None = None


class DocfetchingWorker:
    """Fetches and dispatches document batches from any connector type.

    The ``processing_fn`` is called once per complete batch inside a thread pool.
    Exceptions raised by ``processing_fn`` are logged but do not abort the fetch
    loop — partial failures are surfaced in the returned ``DocfetchingResult``.

    Example::

        worker = DocfetchingWorker(index_document_batch, config=DocfetchingConfig.from_env())
        result = worker.run_checkpointed(connector, start, end, empty_checkpoint)
    """

    def __init__(
        self,
        processing_fn: DocProcessingFn,
        *,
        config: DocfetchingConfig | None = None,
    ) -> None:
        self._fn = processing_fn
        self._config = config or DocfetchingConfig.from_env()

    # ------------------------------------------------------------------ #
    # Public entry points                                                  #
    # ------------------------------------------------------------------ #

    def run_checkpointed(
        self,
        connector: CheckpointedConnector,
        start: SecondsSinceUnixEpoch,
        end: SecondsSinceUnixEpoch,
        checkpoint: ConnectorCheckpoint,
        *,
        callback: IndexingHeartbeatInterface | None = None,
    ) -> DocfetchingResult:
        """Drive a CheckpointedConnector and capture the returned checkpoint."""
        gen = connector.load_from_checkpoint(start, end, checkpoint)
        return self._run(gen, callback=callback)

    def run_load(
        self,
        connector: LoadConnector,
        *,
        callback: IndexingHeartbeatInterface | None = None,
    ) -> DocfetchingResult:
        """Drive a LoadConnector full-state load."""
        return self._run(
            _flatten_batches(connector.load_from_state()), callback=callback
        )

    def run_poll(
        self,
        connector: PollConnector,
        start: SecondsSinceUnixEpoch,
        end: SecondsSinceUnixEpoch,
        *,
        callback: IndexingHeartbeatInterface | None = None,
    ) -> DocfetchingResult:
        """Drive a PollConnector for the given time window."""
        return self._run(
            _flatten_batches(connector.poll_source(start, end)), callback=callback
        )

    # ------------------------------------------------------------------ #
    # Core                                                                 #
    # ------------------------------------------------------------------ #

    def _run(
        self,
        items: Iterator[Document | HierarchyNode | ConnectorFailure],
        *,
        callback: IndexingHeartbeatInterface | None,
    ) -> DocfetchingResult:
        item_q: queue.Queue[tuple[str, Any]] = queue.Queue(
            maxsize=self._config.batch_size * 2
        )

        feeder = threading.Thread(
            target=_feed,
            args=(items, item_q),
            daemon=True,
            name="docfetching-feeder",
        )
        feeder.start()

        result = DocfetchingResult()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._config.max_workers,
            thread_name_prefix="docprocessing",
        ) as pool:
            result = self._drain(item_q, pool, callback, result)

        return result

    def _drain(
        self,
        item_q: queue.Queue,
        pool: concurrent.futures.Executor,
        callback: IndexingHeartbeatInterface | None,
        result: DocfetchingResult,
    ) -> DocfetchingResult:
        futures: list[concurrent.futures.Future] = []
        pending: list[Document] = []

        while True:
            if callback and callback.should_stop():
                break

            try:
                kind, value = item_q.get(timeout=self._config.watchdog_timeout_secs)
            except queue.Empty:
                logger.warning(
                    "Docfetching watchdog: connector stalled for %.0fs, aborting.",
                    self._config.watchdog_timeout_secs,
                )
                result.watchdog_triggered = True
                break

            if kind == _KIND_DONE:
                result.final_checkpoint = value
                if pending:
                    futures.append(pool.submit(self._fn, pending))
                    result.batches_dispatched += 1
                break

            if kind == _KIND_ERROR:
                raise value

            # kind == _KIND_ITEM
            item = value
            if isinstance(item, ConnectorFailure):
                result.failures.append(item)
            elif isinstance(item, Document):
                pending.append(item)
                result.total_documents += 1
                if len(pending) >= self._config.batch_size:
                    futures.append(pool.submit(self._fn, list(pending)))
                    result.batches_dispatched += 1
                    pending = []
                    if callback:
                        callback.progress(
                            "docfetching.batch_dispatched", self._config.batch_size
                        )
            # HierarchyNode: not routed through document indexing here

        for fut in concurrent.futures.as_completed(futures):
            try:
                fut.result()
            except Exception:
                logger.exception("Docprocessing task raised an exception")

        return result


# ------------------------------------------------------------------ #
# Module-level helpers                                                #
# ------------------------------------------------------------------ #


def _feed(
    items: Iterator[Document | HierarchyNode | ConnectorFailure],
    out: queue.Queue,
) -> None:
    """Drive *items* and forward results to *out*.  Runs in a daemon thread.

    ``StopIteration.value`` is forwarded as the checkpoint on the ``done`` message
    so that CheckpointedConnector return values are not silently discarded.
    """
    try:
        while True:
            try:
                item = next(items)
            except StopIteration as exc:
                out.put((_KIND_DONE, exc.value))
                return
            out.put((_KIND_ITEM, item))
    except Exception as exc:
        out.put((_KIND_ERROR, exc))


def _flatten_batches(
    batches: Iterator[list[Document | HierarchyNode]],
) -> Iterator[Document | HierarchyNode]:
    for batch in batches:
        yield from batch


def _env_int(env: dict[str, str], name: str, default: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


__all__ = [
    "DocfetchingConfig",
    "DocfetchingResult",
    "DocfetchingWorker",
    "DocProcessingFn",
]
