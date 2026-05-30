"""Generic insertion retry helpers for embedded chunks."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from itertools import groupby
from typing import Protocol, TypeVar

from src.backend.connectors.models import ConnectorFailure
from .models import EmbeddedChunk

T = TypeVar("T")


class ChunkSink(Protocol[T]):
    """Minimal sink protocol for writing embedded chunks."""

    def index(self, chunks: Iterable[EmbeddedChunk]) -> list[T]: ...


def write_chunks_with_backoff(
    sink: ChunkSink[T],
    make_chunks: Callable[[], Iterable[EmbeddedChunk]],
    *,
    retry_sleep_seconds: float = 0.0,
) -> tuple[list[T], list[ConnectorFailure]]:
    """Write all chunks, then isolate per-document failures on retry."""

    try:
        return sink.index(make_chunks()), []
    except Exception:
        if retry_sleep_seconds:
            time.sleep(retry_sleep_seconds)

    records: list[T] = []
    failures: list[ConnectorFailure] = []
    seen_doc_ids: set[str] = set()

    def doc_id(chunk: EmbeddedChunk) -> str:
        return chunk.chunk.document_id

    for document_id, doc_chunks in groupby(make_chunks(), key=doc_id):
        if document_id in seen_doc_ids:
            raise RuntimeError(
                "Chunks must be grouped by document before per-document retry."
            )
        seen_doc_ids.add(document_id)
        doc_chunk_list = list(doc_chunks)
        try:
            records.extend(sink.index(doc_chunk_list))
        except Exception as exc:
            failures.append(
                ConnectorFailure(
                    document_id=document_id,
                    message=str(exc),
                    exception_type=type(exc).__name__,
                    metadata={
                        "chunk_ids": [chunk.chunk.id for chunk in doc_chunk_list]
                    },
                )
            )

    return records, failures


write_chunks_to_vector_db_with_backoff = write_chunks_with_backoff

__all__ = [
    "ChunkSink",
    "write_chunks_to_vector_db_with_backoff",
    "write_chunks_with_backoff",
]
