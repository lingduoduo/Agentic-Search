"""Temporary storage for embedded indexing batches."""

from __future__ import annotations

import pickle
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

from .models import EmbeddedChunk


class ChunkBatchStore:
    """Serialize embedded chunk batches to a temporary directory.

    This is useful for large indexing jobs where embedding and downstream index
    insertion should be decoupled without holding every embedded chunk in memory.
    """

    _EXT = ".pkl"

    def __init__(self) -> None:
        self._tmpdir: Path | None = None

    def __enter__(self) -> "ChunkBatchStore":
        self._tmpdir = Path(tempfile.mkdtemp(prefix="agentic_search_embeddings_"))
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    @property
    def _dir(self) -> Path:
        if self._tmpdir is None:
            raise RuntimeError("ChunkBatchStore used outside context manager.")
        return self._tmpdir

    def save(self, chunks: list[EmbeddedChunk], batch_idx: int) -> None:
        with (self._dir / f"batch_{batch_idx}{self._EXT}").open("wb") as fh:
            pickle.dump(chunks, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def stream(self) -> Iterator[EmbeddedChunk]:
        for batch_file in self._batch_files():
            yield from self._load(batch_file)

    def scrub_failed_docs(self, failed_doc_ids: set[str]) -> None:
        for batch_file in self._batch_files():
            batch_chunks = self._load(batch_file)
            cleaned = [
                chunk
                for chunk in batch_chunks
                if chunk.chunk.document_id not in failed_doc_ids
            ]
            if len(cleaned) != len(batch_chunks):
                with batch_file.open("wb") as fh:
                    pickle.dump(cleaned, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def _load(self, batch_file: Path) -> list[EmbeddedChunk]:
        with batch_file.open("rb") as fh:
            return pickle.load(fh)  # noqa: S301 - reads files written by save().

    def _batch_files(self) -> list[Path]:
        return sorted(
            self._dir.glob(f"batch_*{self._EXT}"),
            key=lambda path: int(path.stem.removeprefix("batch_")),
        )


__all__ = ["ChunkBatchStore"]
