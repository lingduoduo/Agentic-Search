"""FAISS/artifact writers and JSONL corpus I/O for index building.

Split out of ``index_builder`` so the on-disk artifact concerns (FAISS index
files, embedding memmaps, corpus JSONL) live apart from chunking and embedding.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.internal.document_index._common import _require_faiss
from src.internal.document_index.models import EmbeddedChunk, IndexChunk

try:
    import orjson as _orjson

    _json_loads = _orjson.loads
except ImportError:
    _json_loads = json.loads


class _Corpus:
    """Minimal JSONL corpus reader — avoids HuggingFace Hub network calls."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, key: int | slice | str) -> Any:
        if isinstance(key, int):
            return self._rows[key]
        if isinstance(key, slice):
            return _Corpus(self._rows[key])
        if isinstance(key, str):
            return [row.get(key) for row in self._rows]
        raise TypeError(f"Unsupported corpus key type: {type(key)}")


def load_corpus(corpus_path: str) -> _Corpus:
    with open(corpus_path, "rb") as fh:
        rows = [_json_loads(line) for line in fh if line.strip()]
    return _Corpus(rows)


def load_corpus_from_connector(connector: Any) -> _Corpus:
    """Build a _Corpus from any LoadConnector, skipping HierarchyNodes."""
    from src.internal.connectors.models import Document

    rows: list[dict] = []
    for batch in connector.load_from_state():
        for item in batch:
            if isinstance(item, Document):
                rows.append(
                    {"id": item.id, "title": item.title, "contents": item.contents}
                )
    return _Corpus(rows)


def dump_connector_to_jsonl(connector: Any, path: "Path | str") -> None:
    """Write connector documents to a JSONL file suitable for BM25/pyserini indexing."""
    from src.internal.connectors.models import Document

    dest = Path(path)
    with dest.open("w", encoding="utf-8") as fh:
        for batch in connector.load_from_state():
            for item in batch:
                if isinstance(item, Document):
                    fh.write(
                        json.dumps(
                            {
                                "id": item.id,
                                "title": item.title,
                                "contents": item.contents,
                            },
                            ensure_ascii=False,
                        )
                    )
                    fh.write("\n")


def set_hnsw_ef_construction(faiss_index: Any, value: int) -> None:
    hnsw = getattr(faiss_index, "hnsw", None)
    if hnsw is None:
        raise ValueError(
            "hnsw_ef_construction can only be used with HNSW FAISS indexes."
        )
    hnsw.efConstruction = value


def set_hnsw_ef_search(faiss_index: Any, value: int) -> None:
    """Set the HNSW efSearch parameter on a FAISS HNSW index.

    efSearch controls the recall/latency trade-off at query time.
    Higher values improve recall at the cost of slower searches.
    Typical range: 16–512.
    """
    hnsw = getattr(faiss_index, "hnsw", None)
    if hnsw is None:
        raise ValueError("hnsw_ef_search can only be used with HNSW FAISS indexes.")
    hnsw.efSearch = value


def write_dense_faiss_index(
    embeddings: np.ndarray,
    index_path: str | Path,
    *,
    faiss_type: str = "Flat",
    hnsw_ef_construction: int | None = None,
    hnsw_ef_search: int | None = None,
    faiss_gpu: bool = False,
    gpu_num: int = 0,
) -> None:
    """Build and write a FAISS index from float32 embeddings."""

    if embeddings.size == 0:
        raise ValueError("Cannot build a dense index for empty embeddings.")

    faiss = _require_faiss()
    is_hnsw = "HNSW" in faiss_type.upper()
    if faiss_gpu and is_hnsw:
        raise ValueError(
            "faiss_gpu=True is not compatible with HNSW indexes. "
            "FAISS cannot GPU-shard HNSW indexes. "
            "Use faiss_type='Flat' or an IVF variant for GPU indexing."
        )

    embeddings = np.asarray(embeddings, dtype=np.float32)
    faiss_index = faiss.index_factory(
        embeddings.shape[-1], faiss_type, faiss.METRIC_INNER_PRODUCT
    )
    if hnsw_ef_construction is not None:
        set_hnsw_ef_construction(faiss_index, hnsw_ef_construction)

    if faiss_gpu:
        if not hasattr(faiss, "GpuMultipleClonerOptions") or gpu_num == 0:
            raise RuntimeError(
                "faiss_gpu was requested, but GPU FAISS support is not available."
            )
        clone_options = faiss.GpuMultipleClonerOptions()
        clone_options.useFloat16 = True
        clone_options.shard = True
        faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, clone_options)

    if not faiss_index.is_trained:
        faiss_index.train(embeddings)
    faiss_index.add(embeddings)

    if faiss_gpu:
        faiss_index = faiss.index_gpu_to_cpu(faiss_index)

    if hnsw_ef_search is not None:
        set_hnsw_ef_search(faiss_index, hnsw_ef_search)

    faiss.write_index(faiss_index, str(index_path))


def write_corpus_jsonl(chunks: Sequence[IndexChunk], path: str | Path) -> Path:
    """Write chunk corpus JSONL compatible with IndexBuilder/retrievers."""

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(
                json.dumps(
                    {
                        "id": chunk.id,
                        "document_id": chunk.document_id,
                        "chunk_id": chunk.chunk_id,
                        "title": chunk.title,
                        "url": chunk.url,
                        "contents": chunk.text,
                        "metadata": chunk.metadata or {},
                        "mini_chunk_texts": chunk.mini_chunk_texts or [],
                    },
                    ensure_ascii=False,
                )
            )
            fh.write("\n")
    return dest


def write_embeddings_memmap(
    embedded_chunks: Sequence[EmbeddedChunk],
    path: str | Path,
) -> Path:
    """Write chunk embeddings as float32 memmap rows."""

    if not embedded_chunks:
        raise ValueError("Cannot write embeddings for an empty chunk list.")
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    embeddings = np.vstack([item.embedding for item in embedded_chunks]).astype(
        np.float32,
        copy=False,
    )
    memmap = np.memmap(dest, mode="w+", dtype=np.float32, shape=embeddings.shape)
    memmap[:] = embeddings
    memmap.flush()
    return dest


def write_faiss_index(
    embedded_chunks: Sequence[EmbeddedChunk],
    path: str | Path,
    *,
    faiss_type: str = "Flat",
    hnsw_ef_construction: int | None = None,
    hnsw_ef_search: int | None = None,
) -> Path:
    """Write a FAISS index from embedded chunks."""

    if not embedded_chunks:
        raise ValueError("Cannot write a FAISS index for an empty chunk list.")
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    embeddings = np.vstack([item.embedding for item in embedded_chunks]).astype(
        np.float32,
        copy=False,
    )
    write_dense_faiss_index(
        embeddings,
        dest,
        faiss_type=faiss_type,
        hnsw_ef_construction=hnsw_ef_construction,
        hnsw_ef_search=hnsw_ef_search,
    )
    return dest
