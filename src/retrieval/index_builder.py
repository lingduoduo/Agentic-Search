"""Utilities for building dense or BM25 retrieval indexes."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from collections import Counter
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence

try:
    import orjson as _orjson

    _json_loads = _orjson.loads
except ImportError:
    _json_loads = json.loads
import shutil
import subprocess
import sys
import warnings
from multiprocessing import cpu_count
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import numpy as np

from src.backend.connectors.models import ConnectorFailure
from src.backend.connectors.models import Document
from src.backend.document_index.text import (
    MAX_LENGTH as DEFAULT_VOCAB_MAX_LENGTH,
    Vocabulary,
    normalize_document,
    tokenize_text,
)
from .indexing_heartbeat import IndexingHeartbeatInterface
from .models import ChunkingConfig
from .models import EmbeddedChunk
from .models import EmbeddingConfig
from .models import IndexChunk
from .models import IndexingPipelineConfig
from .models import IndexingPipelineResult
from .models import IndexWriterConfig

# Must be set before torch/faiss are imported to prevent an OpenMP conflict on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "ChunkingConfig",
    "EmbeddedChunk",
    "EmbeddingConfig",
    "IndexChunk",
    "IndexingPipelineConfig",
    "IndexingPipelineResult",
    "IndexWriterConfig",
    "_split_paragraphs",
    "_split_sentences_in_paragraph",
    "chunk_document",
    "chunk_documents",
    "deterministic_embedding_fn",
    "embed_chunks",
    "embed_chunks_with_failure_handling",
    "filter_indexable_documents",
    "generate_large_chunks",
    "prepare_texts",
    "run_indexing_pipeline",
    "write_faiss_index",
]

MODEL2POOLING = {
    "e5": "mean",
    "bge": "cls",
    "contriever": "mean",
    "jina": "mean",
}

EmbeddingFn = Callable[[list[str]], np.ndarray]

RETURN_SEPARATOR = "\n\n"
SECTION_SEPARATOR = "\n\n---\n\n"
MAX_METADATA_PERCENTAGE = 0.25
CHUNK_MIN_CONTENT = 16


def _require_torch():
    import torch

    return torch


def _require_faiss():
    import faiss

    return faiss


def _require_transformers():
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    return AutoConfig, AutoModel, AutoTokenizer


def _require_tqdm():
    from tqdm import tqdm

    return tqdm


def prepare_texts(
    texts: list[str],
    retrieval_method: str,
    *,
    is_query: bool,
    query_prefix: str | None = None,
    passage_prefix: str | None = None,
) -> list[str]:
    normalized_method = retrieval_method.lower()
    prepared = list(texts)

    explicit_prefix = query_prefix if is_query else passage_prefix
    if explicit_prefix:
        prepared = [_apply_text_prefix(text, explicit_prefix) for text in prepared]
    elif "e5" in normalized_method:
        prefix = "query" if is_query else "passage"
        prepared = [_apply_text_prefix(text, f"{prefix}:") for text in prepared]

    if "bge" in normalized_method and is_query and not explicit_prefix:
        prepared = [
            f"Represent this sentence for searching relevant passages: {text}"
            for text in prepared
        ]

    return prepared


def _apply_text_prefix(text: str, prefix: str) -> str:
    normalized_prefix = prefix.strip()
    if not normalized_prefix:
        return text
    if text.startswith(normalized_prefix):
        return text
    separator = "" if normalized_prefix.endswith(("\n", " ")) else " "
    return f"{normalized_prefix}{separator}{text}"


def load_model(
    model_path: str, use_fp16: bool = False, device: str = "cpu"
) -> tuple[Any, Any]:
    _, auto_model, auto_tokenizer = _require_transformers()

    # Try local cache first to skip the network version-check that can hang.
    # Falls back to a full download when the model is not cached yet.
    def _load(cls, **extra):
        try:
            return cls.from_pretrained(
                model_path, local_files_only=True, trust_remote_code=True, **extra
            )
        except OSError:
            return cls.from_pretrained(model_path, trust_remote_code=True, **extra)

    model = _load(auto_model)
    model.eval()
    model.to(device)
    if use_fp16 and device.startswith("cuda"):
        model = model.half()
    tokenizer = _load(auto_tokenizer, use_fast=True)
    return model, tokenizer


def pooling(
    pooler_output: Any,
    last_hidden_state: Any,
    attention_mask: Any | None = None,
    pooling_method: str = "mean",
) -> Any:
    if pooling_method == "mean":
        if attention_mask is None:
            raise ValueError("attention_mask is required for mean pooling.")
        last_hidden = last_hidden_state.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    if pooling_method == "cls":
        return last_hidden_state[:, 0]
    if pooling_method == "pooler":
        return pooler_output
    raise NotImplementedError("Pooling method not implemented!")


def _encode_batch(
    encoder: Any,
    tokenizer: Any,
    texts: list[str],
    retrieval_method: str,
    max_length: int,
    pooling_method: str,
    device: str,
) -> "np.ndarray":
    """Tokenize *texts*, run the encoder, and return a float32 numpy array.

    Shared by IndexBuilder.encode_all and DenseRetriever.encode_queries — the
    two callers are responsible for wrapping in torch.no_grad() if needed.
    """
    torch = _require_torch()
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    if "T5" in type(encoder).__name__:
        decoder_input_ids = torch.zeros(
            (inputs["input_ids"].shape[0], 1),
            dtype=torch.long,
            device=inputs["input_ids"].device,
        )
        output = encoder(
            **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
        )
        embeddings = output.last_hidden_state[:, 0, :]
    else:
        output = encoder(**inputs, return_dict=True)
        embeddings = pooling(
            output.pooler_output,
            output.last_hidden_state,
            inputs["attention_mask"],
            pooling_method,
        )
        if "dpr" not in retrieval_method:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    return embeddings.detach().cpu().numpy().astype(np.float32)


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
    from src.backend.connectors.models import Document

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
    from src.backend.connectors.models import Document

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


def chunk_document(document: Document, config: ChunkingConfig) -> list[IndexChunk]:
    """Split one document into token-budgeted chunks ready for indexing."""

    config.validate()
    text = document.contents.strip()
    if not text:
        return []

    title_prefix = ""
    title_tokens = 0
    if config.include_title and document.title:
        title = _extract_blurb(document.title, config.blurb_size)
        title_prefix = f"{title}{RETURN_SEPARATOR}" if title else ""
        title_tokens = _token_count(title_prefix)

    metadata_suffix_semantic = ""
    metadata_suffix_keyword = ""
    metadata_tokens = 0
    if config.include_metadata:
        metadata_suffix_semantic, metadata_suffix_keyword = _metadata_suffix_for_index(
            document.metadata,
            include_separator=True,
        )
        metadata_tokens = _token_count(metadata_suffix_semantic)
        if metadata_tokens >= config.chunk_size * config.max_metadata_percentage:
            metadata_suffix_semantic = ""
            metadata_tokens = 0

    content_token_limit = config.chunk_size - title_tokens - metadata_tokens
    if content_token_limit <= config.min_content_tokens:
        metadata_suffix_semantic = ""
        metadata_tokens = 0
        content_token_limit = config.chunk_size - title_tokens

    if content_token_limit <= config.min_content_tokens:
        title_prefix = ""
        content_token_limit = config.chunk_size

    chunk_texts = _split_text(text, content_token_limit, config.chunk_overlap)
    chunks: list[IndexChunk] = []
    for chunk_id, chunk_text in enumerate(chunk_texts):
        index_text = (
            f"{title_prefix}{chunk_text}{metadata_suffix_semantic}"
            if title_prefix or metadata_suffix_semantic
            else chunk_text
        )
        mini_chunk_texts = _make_mini_chunk_texts(chunk_text, config)
        metadata = {
            **document.metadata,
            "permissions": document.permissions,
        }
        if metadata_suffix_keyword:
            metadata["metadata_keyword"] = metadata_suffix_keyword
        chunks.append(
            IndexChunk(
                id=f"{document.id}::chunk-{chunk_id}",
                document_id=document.id,
                chunk_id=chunk_id,
                text=index_text,
                title=document.title,
                url=document.url,
                metadata=metadata,
                blurb=_extract_blurb(chunk_text, config.blurb_size),
                metadata_suffix_semantic=metadata_suffix_semantic,
                metadata_suffix_keyword=metadata_suffix_keyword,
                mini_chunk_texts=mini_chunk_texts,
                section_continuation=chunk_id > 0,
            )
        )

    if config.enable_large_chunks:
        chunks.extend(generate_large_chunks(chunks, config.large_chunk_ratio))
    return chunks


def chunk_documents(
    documents: Iterable[Document],
    config: ChunkingConfig,
    *,
    callback: IndexingHeartbeatInterface | None = None,
) -> list[IndexChunk]:
    """Chunk all non-empty documents."""

    chunks: list[IndexChunk] = []
    for document in documents:
        _raise_if_indexing_stopped(callback, "chunk_documents")
        document_chunks = chunk_document(document, config)
        chunks.extend(document_chunks)
        _report_indexing_progress(callback, "chunk_documents", len(document_chunks))
    return chunks


def filter_indexable_documents(
    documents: Iterable[Document],
    *,
    max_document_chars: int | None = None,
) -> tuple[list[Document], list[ConnectorFailure]]:
    """Drop documents that cannot produce useful chunks and report why."""

    filtered: list[Document] = []
    failures: list[ConnectorFailure] = []
    seen_document_ids: set[str] = set()
    for document in documents:
        if document.id in seen_document_ids:
            failures.append(
                ConnectorFailure(
                    document_id=document.id,
                    message="Duplicate document id skipped before indexing.",
                    exception_type=None,
                    metadata={"url": document.url} if document.url else {},
                )
            )
            continue
        seen_document_ids.add(document.id)

        title = document.title or ""
        contents = document.contents or ""
        if not title.strip() and not contents.strip():
            failures.append(
                ConnectorFailure(
                    document_id=document.id,
                    message="Document has neither title nor contents.",
                    exception_type=None,
                    metadata={"url": document.url} if document.url else {},
                )
            )
            continue

        char_count = len(title) + len(contents)
        if max_document_chars is not None and char_count > max_document_chars:
            failures.append(
                ConnectorFailure(
                    document_id=document.id,
                    message=(
                        f"Document is too large to index "
                        f"({char_count:,} chars, max={max_document_chars:,})."
                    ),
                    exception_type=None,
                    metadata={
                        "url": document.url,
                        "char_count": char_count,
                        "max_document_chars": max_document_chars,
                    },
                )
            )
            continue

        filtered.append(document)

    return filtered, failures


def embed_chunks(
    chunks: Sequence[IndexChunk],
    *,
    embedding_fn: EmbeddingFn,
    config: EmbeddingConfig,
    callback: IndexingHeartbeatInterface | None = None,
) -> list[EmbeddedChunk]:
    """Embed chunks in batches using model-specific text preparation."""

    config.validate()
    if config.isolate_failures:
        embedded, failures = embed_chunks_with_failure_handling(
            chunks,
            embedding_fn=embedding_fn,
            config=config,
            callback=callback,
        )
        if failures:
            failed_docs = ", ".join(
                failure.document_id or "<unknown>" for failure in failures
            )
            raise RuntimeError(f"Failed to embed chunks for documents: {failed_docs}")
        return embedded

    embedded: list[EmbeddedChunk] = []
    title_embedding_cache: dict[str, np.ndarray] = {}
    for batch in _batched(list(chunks), config.batch_size):
        _raise_if_indexing_stopped(callback, "embed_chunks")
        embedded.extend(
            _embed_chunk_batch(
                batch,
                embedding_fn=embedding_fn,
                config=config,
                title_embedding_cache=title_embedding_cache,
            )
        )
        _report_indexing_progress(callback, "embed_chunks", len(batch))
    return embedded


def embed_chunks_with_failure_handling(
    chunks: Sequence[IndexChunk],
    *,
    embedding_fn: EmbeddingFn,
    config: EmbeddingConfig,
    callback: IndexingHeartbeatInterface | None = None,
) -> tuple[list[EmbeddedChunk], list[ConnectorFailure]]:
    """Embed a batch, then isolate failures document-by-document if needed."""

    try:
        fallback_config = replace(config, isolate_failures=False)
        return (
            embed_chunks(
                chunks,
                embedding_fn=embedding_fn,
                config=fallback_config,
                callback=callback,
            ),
            [],
        )
    except Exception:
        logger.exception("Failed to embed chunk batch. Trying individual documents.")
        if config.failure_retry_seconds:
            time.sleep(config.failure_retry_seconds)

    embedded_chunks: list[EmbeddedChunk] = []
    failures: list[ConnectorFailure] = []
    chunks_by_doc: dict[str, list[IndexChunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_doc[chunk.document_id].append(chunk)

    fallback_config = replace(config, isolate_failures=False)
    for document_id, document_chunks in chunks_by_doc.items():
        _raise_if_indexing_stopped(callback, "embed_chunks_with_failure_handling")
        try:
            embedded_chunks.extend(
                embed_chunks(
                    document_chunks,
                    embedding_fn=embedding_fn,
                    config=fallback_config,
                    callback=callback,
                )
            )
        except Exception as exc:
            logger.exception("Failed to embed chunks for document '%s'", document_id)
            failures.append(
                ConnectorFailure(
                    document_id=document_id,
                    message=str(exc),
                    exception_type=type(exc).__name__,
                    metadata={
                        "chunk_ids": [chunk.id for chunk in document_chunks],
                    },
                )
            )

    return embedded_chunks, failures


def _embed_texts(
    texts: Sequence[str],
    *,
    embedding_fn: EmbeddingFn,
    config: EmbeddingConfig,
    is_query: bool,
) -> np.ndarray:
    prepared_texts = prepare_texts(
        list(texts),
        config.retrieval_method,
        is_query=is_query,
        query_prefix=config.query_prefix,
        passage_prefix=config.passage_prefix,
    )
    vectors = embedding_fn(prepared_texts)
    return _coerce_embedding_matrix(
        vectors,
        expected_rows=len(prepared_texts),
        normalize=config.normalize_embeddings,
    )


def _get_title_embedding(
    chunk: IndexChunk,
    cache: dict[str, np.ndarray],
    *,
    embedding_fn: EmbeddingFn,
    config: EmbeddingConfig,
) -> np.ndarray | None:
    if not config.embed_titles or not chunk.title:
        return None
    title = chunk.title.strip()
    if not title:
        return None
    if title not in cache:
        cache[title] = _embed_texts(
            [title],
            embedding_fn=embedding_fn,
            config=config,
            is_query=False,
        )[0]
    return cache[title]


def _embed_chunk_batch(
    chunks: Sequence[IndexChunk],
    *,
    embedding_fn: EmbeddingFn,
    config: EmbeddingConfig,
    title_embedding_cache: dict[str, np.ndarray],
) -> list[EmbeddedChunk]:
    flat_texts: list[str] = []
    text_counts: list[int] = []
    for chunk in chunks:
        texts = [chunk.text, *(chunk.mini_chunk_texts or [])]
        flat_texts.extend(texts)
        text_counts.append(len(texts))

    vectors = _embed_texts(
        flat_texts,
        embedding_fn=embedding_fn,
        config=config,
        is_query=False,
    )

    embedded: list[EmbeddedChunk] = []
    vector_index = 0
    for chunk, text_count in zip(chunks, text_counts):
        chunk_vectors = vectors[vector_index : vector_index + text_count]
        embedded.append(
            EmbeddedChunk(
                chunk=chunk,
                embedding=chunk_vectors[0],
                title_embedding=_get_title_embedding(
                    chunk,
                    title_embedding_cache,
                    embedding_fn=embedding_fn,
                    config=config,
                ),
                mini_chunk_embeddings=[vector.copy() for vector in chunk_vectors[1:]],
            )
        )
        vector_index += text_count

    return embedded


def _coerce_embedding_matrix(
    vectors: Any,
    *,
    expected_rows: int,
    normalize: bool,
) -> np.ndarray:
    if not isinstance(vectors, np.ndarray):
        vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != expected_rows:
        raise ValueError(
            "embedding_fn must return a 2D array with one row per input text."
        )
    vectors = vectors.astype(np.float32, copy=False)
    if normalize:
        vectors = _normalize_embedding_rows(vectors)
    return vectors


def _normalize_embedding_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(
        vectors,
        np.maximum(norms, np.finfo(np.float32).eps),
        out=np.zeros_like(vectors, dtype=np.float32),
    )


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


def run_indexing_pipeline(
    documents: Iterable[Document],
    *,
    config: IndexingPipelineConfig,
    embedding_fn: EmbeddingFn,
    callback: IndexingHeartbeatInterface | None = None,
) -> IndexingPipelineResult:
    """Chunk, embed, and write index artifacts for connector documents."""

    config.validate()
    docs = list(documents)
    indexable_docs, failures = filter_indexable_documents(
        docs,
        max_document_chars=config.chunking.max_document_chars,
    )
    _raise_if_indexing_stopped(callback, "run_indexing_pipeline")
    chunks = chunk_documents(indexable_docs, config.chunking, callback=callback)
    if not chunks:
        raise ValueError("No non-empty chunks were produced.")

    save_dir = Path(config.writer.save_dir)
    corpus_path = save_dir / config.writer.corpus_filename
    embedding_path = save_dir / config.writer.embedding_filename
    index_path = save_dir / config.writer.index_filename

    _raise_if_indexing_stopped(callback, "write_corpus_jsonl")
    write_corpus_jsonl(chunks, corpus_path)
    _report_indexing_progress(callback, "write_corpus_jsonl", len(chunks))
    embedded_chunks = embed_chunks(
        chunks,
        embedding_fn=embedding_fn,
        config=config.embedding,
        callback=callback,
    )
    _raise_if_indexing_stopped(callback, "write_embeddings_memmap")
    write_embeddings_memmap(embedded_chunks, embedding_path)
    _report_indexing_progress(callback, "write_embeddings_memmap", len(embedded_chunks))

    written_index_path = None
    if config.writer.write_faiss:
        _raise_if_indexing_stopped(callback, "write_faiss_index")
        written_index_path = write_faiss_index(
            embedded_chunks,
            index_path,
            faiss_type=config.writer.faiss_type,
            hnsw_ef_construction=config.writer.hnsw_ef_construction,
            hnsw_ef_search=config.writer.hnsw_ef_search,
        )
        _report_indexing_progress(callback, "write_faiss_index", len(embedded_chunks))

    return IndexingPipelineResult(
        total_documents=len(docs),
        total_chunks=len(chunks),
        corpus_path=corpus_path,
        embedding_path=embedding_path,
        index_path=written_index_path,
        failures=failures,
    )


def _raise_if_indexing_stopped(
    callback: IndexingHeartbeatInterface | None,
    tag: str,
) -> None:
    if callback and callback.should_stop():
        raise RuntimeError(f"{tag}: stop signal detected")


def _report_indexing_progress(
    callback: IndexingHeartbeatInterface | None,
    tag: str,
    amount: int,
) -> None:
    if callback:
        callback.progress(tag, amount)


def deterministic_embedding_fn(dim: int = 8) -> EmbeddingFn:
    """Return a deterministic embedding function useful for demos and tests."""

    if dim < 1:
        raise ValueError("dim must be at least 1.")

    def embed(texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), dim), dtype=np.float32)
        for row, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            for col in range(dim):
                vectors[row, col] = digest[col] / 255.0
            norm = np.linalg.norm(vectors[row])
            if norm > 0:
                vectors[row] /= norm
        return vectors

    return embed


def _metadata_suffix_for_index(
    metadata: dict[str, Any],
    *,
    include_separator: bool = False,
) -> tuple[str, str]:
    """Render metadata for semantic and keyword indexing."""

    if not metadata:
        return "", ""

    semantic_lines = ["Metadata:"]
    keyword_values: list[str] = []
    for key, value in metadata.items():
        if value is None or key == "permissions":
            continue

        values = value if isinstance(value, list) else [value]
        value_strings = [str(item).strip() for item in values if str(item).strip()]
        if not value_strings:
            continue

        keyword_values.extend(value_strings)
        semantic_lines.append(f"\t{key} - {', '.join(value_strings)}")

    if len(semantic_lines) == 1:
        return "", ""

    semantic = "\n".join(semantic_lines)
    keyword = " ".join(keyword_values)
    if include_separator:
        semantic = f"{RETURN_SEPARATOR}{semantic}"
        keyword = f"{RETURN_SEPARATOR}{keyword}" if keyword else ""
    return semantic, keyword


def _tokenize_for_chunking(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def _token_count(text: str) -> int:
    return len(_tokenize_for_chunking(text))


def _extract_blurb(text: str, blurb_size: int) -> str:
    chunks = _split_text(text.strip(), blurb_size, 0)
    return chunks[0] if chunks else ""


def _make_mini_chunk_texts(
    chunk_text: str,
    config: ChunkingConfig,
) -> list[str] | None:
    if not config.enable_mini_chunks:
        return None
    mini_chunks = _split_text(chunk_text, config.mini_chunk_size, 0)
    if len(mini_chunks) <= 1 and (not mini_chunks or mini_chunks[0] == chunk_text):
        return None
    return mini_chunks


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into chunks, respecting paragraph and section boundaries."""
    return _split_text_paragraphs(text, chunk_size, chunk_overlap)


def _split_paragraphs(text: str) -> list[str]:
    """Split text on paragraph and section boundaries without destroying internal whitespace.

    Splits on two or more consecutive newlines (blank-line paragraph breaks) or a
    newline immediately followed by a markdown heading marker (#).
    """
    if not text or not text.strip():
        return []
    parts = re.split(r"\n{2,}|\n(?=#)", text)
    return [p.strip() for p in parts if p.strip()]


def _split_sentences_in_paragraph(para: str) -> list[str]:
    """Sentence-split within a single paragraph, preserving intra-paragraph whitespace."""
    normalized = re.sub(r"[ \t]+", " ", para).strip()
    if not normalized:
        return []
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?。！？])\s+", normalized)
        if part.strip()
    ]


def _split_text_paragraphs(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Paragraph-aware chunking: flush at section boundaries when chunk is ≥ 50% full."""
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para_idx, para in enumerate(paragraphs):
        para_sentences = _split_sentences_in_paragraph(para)
        if not para_sentences:
            continue

        if current and current_tokens >= chunk_size // 2 and para_idx > 0:
            chunks.append(" ".join(current).strip())
            current = _overlap_tail(current, chunk_overlap)
            current_tokens = _token_count(" ".join(current))

        for sentence in para_sentences:
            sentence_tokens = _token_count(sentence)

            if sentence_tokens > chunk_size:
                if current:
                    chunks.append(" ".join(current).strip())
                    current = []
                    current_tokens = 0
                chunks.extend(_split_token_window(sentence, chunk_size, chunk_overlap))
                continue

            would_exceed = current and current_tokens + sentence_tokens > chunk_size
            if would_exceed:
                chunks.append(" ".join(current).strip())
                current = _overlap_tail(current, chunk_overlap)
                current_tokens = _token_count(" ".join(current))
                if current and current_tokens + sentence_tokens > chunk_size:
                    current = []
                    current_tokens = 0

            current.append(sentence)
            current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def _split_token_window(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    tokens = _tokenize_for_chunking(text)
    chunks: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(tokens):
        chunk = " ".join(tokens[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(tokens):
            break
        start += step
    return chunks


def _overlap_tail(sentences: Sequence[str], chunk_overlap: int) -> list[str]:
    if chunk_overlap <= 0:
        return []

    selected: list[str] = []
    selected_tokens = 0
    for sentence in reversed(sentences):
        sentence_tokens = _token_count(sentence)
        if selected and selected_tokens + sentence_tokens > chunk_overlap:
            break
        if sentence_tokens > chunk_overlap:
            tail = _tokenize_for_chunking(sentence)[-chunk_overlap:]
            return [" ".join(tail)]
        selected.insert(0, sentence)
        selected_tokens += sentence_tokens
    return selected


def _combine_index_chunks(
    chunks: Sequence[IndexChunk], large_chunk_id: int
) -> IndexChunk:
    combined_text = SECTION_SEPARATOR.join(chunk.text for chunk in chunks)
    reference_ids = [chunk.chunk_id for chunk in chunks]
    first = chunks[0]
    return IndexChunk(
        id=f"{first.document_id}::large-chunk-{large_chunk_id}",
        document_id=first.document_id,
        chunk_id=first.chunk_id,
        text=combined_text,
        title=first.title,
        url=first.url,
        metadata=dict(first.metadata or {}),
        blurb=first.blurb,
        large_chunk_reference_ids=reference_ids,
        large_chunk_id=large_chunk_id,
    )


def generate_large_chunks(
    chunks: Sequence[IndexChunk],
    large_chunk_ratio: int,
) -> list[IndexChunk]:
    """Generate grouped chunks for callers that want multi-pass indexing."""

    large_chunks: list[IndexChunk] = []
    for large_chunk_id, start in enumerate(range(0, len(chunks), large_chunk_ratio)):
        group = chunks[start : start + large_chunk_ratio]
        if len(group) > 1:
            large_chunks.append(_combine_index_chunks(group, large_chunk_id))
    return large_chunks


def _batched(items: list[IndexChunk], batch_size: int) -> Iterable[list[IndexChunk]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@dataclass(frozen=True)
class IndexBuilderConfig:
    retrieval_method: str
    corpus_path: str
    save_dir: str
    model_path: str | None = None
    max_length: int = 180
    batch_size: int = 512
    use_fp16: bool = False
    pooling_method: str = "mean"
    faiss_type: str = "Flat"
    hnsw_ef_construction: int | None = None
    hnsw_ef_search: int | None = None
    embedding_path: str | None = None
    save_embedding: bool = False
    faiss_gpu: bool = False
    save_vocabulary: bool = True
    keyword_limit: int = 10
    vocab_max_length: int = DEFAULT_VOCAB_MAX_LENGTH
    # 0 = auto-detect (uses all available CPUs)
    bm25_threads: int = 0

    def validate(self) -> None:
        retrieval_method = self.retrieval_method.strip().lower()
        if not retrieval_method:
            raise ValueError("retrieval_method is required.")
        if not self.corpus_path:
            raise ValueError("corpus_path is required.")
        if (
            retrieval_method != "bm25"
            and not self.model_path
            and not self.embedding_path
        ):
            raise ValueError(
                "model_path or embedding_path is required for dense indexing."
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.max_length < 1:
            raise ValueError("max_length must be at least 1.")
        if self.keyword_limit < 1:
            raise ValueError("keyword_limit must be at least 1.")
        if self.vocab_max_length < 1:
            raise ValueError("vocab_max_length must be at least 1.")
        if self.hnsw_ef_construction is not None and self.hnsw_ef_construction < 1:
            raise ValueError("hnsw_ef_construction must be at least 1.")
        if self.hnsw_ef_search is not None and self.hnsw_ef_search < 1:
            raise ValueError("hnsw_ef_search must be at least 1.")


class IndexBuilder:
    """Builds dense or BM25 indexes for retrieval."""

    def __init__(self, config: IndexBuilderConfig):
        config.validate()
        self.config = config
        self.retrieval_method = config.retrieval_method.lower()
        self.model_path = config.model_path
        self.corpus_path = config.corpus_path
        self.save_dir = Path(config.save_dir)
        self.max_length = config.max_length
        self.batch_size = config.batch_size
        self.use_fp16 = config.use_fp16
        self.pooling_method = config.pooling_method
        self.faiss_type = config.faiss_type
        self.hnsw_ef_construction = config.hnsw_ef_construction
        self.hnsw_ef_search = config.hnsw_ef_search
        self.embedding_path = config.embedding_path
        self.save_embedding = config.save_embedding
        self.faiss_gpu = config.faiss_gpu
        self.save_vocabulary = config.save_vocabulary
        self.keyword_limit = config.keyword_limit
        self.vocab_max_length = config.vocab_max_length
        self.bm25_threads = config.bm25_threads or cpu_count()

        torch = _require_torch()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_num = torch.cuda.device_count() if torch.cuda.is_available() else 0

        self._prepare_save_dir()
        self.index_save_path = (
            self.save_dir / f"{self.retrieval_method}_{self.faiss_type}.index"
        )
        self.embedding_save_path = self.save_dir / f"emb_{self.retrieval_method}.memmap"
        self.vocab_save_path = self.save_dir / "vocabulary_corpus.json"
        self.corpus = load_corpus(self.corpus_path)

    def _prepare_save_dir(self) -> None:
        if self.save_dir.exists():
            if any(self.save_dir.iterdir()):
                warnings.warn(
                    f"Some files already exist in {self.save_dir} and may be overwritten.",
                    UserWarning,
                )
        else:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def build_index(self) -> None:
        if self.save_vocabulary:
            self.save_vocabulary_metadata()
        if self.retrieval_method == "bm25":
            self.build_bm25_index()
        else:
            self.build_dense_index()

    def save_vocabulary_metadata(self) -> None:
        vocabulary, corpus_entries = self._build_vocabulary_metadata()

        with self.vocab_save_path.open("w", encoding="utf-8") as vocab_file:
            json.dump(
                {
                    "corpus_path": self.corpus_path,
                    "retrieval_method": self.retrieval_method,
                    "keyword_limit": self.keyword_limit,
                    "vocab_max_length": self.vocab_max_length,
                    "vocabulary": {
                        "num_token": vocabulary.num_token,
                        "token2idx": vocabulary.token2idx,
                        "token2cnt": vocabulary.token2cnt,
                        "idx2token": {
                            str(key): value
                            for key, value in vocabulary.idx2token.items()
                        },
                    },
                    "corpus": corpus_entries,
                },
                vocab_file,
                ensure_ascii=False,
                indent=2,
            )

    def _build_vocabulary_metadata(self) -> tuple["Vocabulary", list[dict[str, Any]]]:
        vocabulary = Vocabulary()
        corpus_entries: list[dict[str, Any]] = []

        for index in range(len(self.corpus)):
            item = self.corpus[index]
            text = normalize_document(
                item,
                text_fields=("title", "contents"),
            )
            tokens = tokenize_text(
                text,
                max_length=self.vocab_max_length,
            )
            if tokens:
                vocabulary.add_token_sequence(tokens)
            token_counts = Counter(tokens)
            corpus_entries.append(
                {
                    "doc_id": index,
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "contents": text,
                    "tokens": tokens,
                    "keywords": [
                        token
                        for token, _ in token_counts.most_common(self.keyword_limit)
                    ],
                    "token_count": len(tokens),
                }
            )

        return vocabulary, corpus_entries

    def build_bm25_index(self) -> None:
        bm25_dir = self.save_dir / "bm25"
        bm25_dir.mkdir(parents=True, exist_ok=True)

        with TemporaryDirectory(dir=bm25_dir) as temp_dir:
            temp_dir_path = Path(temp_dir)
            temp_file_path = temp_dir_path / "temp.jsonl"
            shutil.copyfile(self.corpus_path, temp_file_path)

            pyserini_args = [
                "--collection",
                "JsonCollection",
                "--input",
                str(temp_dir_path),
                "--index",
                str(bm25_dir),
                "--generator",
                "DefaultLuceneDocumentGenerator",
                "--threads",
                str(self.bm25_threads),
            ]
            subprocess.run(
                [sys.executable, "-m", "pyserini.index.lucene", *pyserini_args],
                check=True,
            )

    def _load_embedding(self, embedding_path: str, corpus_size: int) -> np.memmap:
        if corpus_size < 1:
            raise ValueError("Cannot infer embedding dimensions for an empty corpus.")

        embedding_bytes = Path(embedding_path).stat().st_size
        row_bytes = np.dtype(np.float32).itemsize * corpus_size
        if embedding_bytes % row_bytes != 0:
            raise ValueError(
                "Embedding file size is not divisible by corpus size and float32 size."
            )

        hidden_size = embedding_bytes // row_bytes
        if hidden_size < 1:
            raise ValueError("Embedding file does not contain any float32 vectors.")

        return np.memmap(embedding_path, mode="r", dtype=np.float32).reshape(
            corpus_size, hidden_size
        )

    def _iter_encoded_batches(
        self, encoder: Any, tokenizer: Any, batch_size: int | None = None
    ):
        tqdm = _require_tqdm()

        effective_batch_size = batch_size if batch_size is not None else self.batch_size
        for start_idx in tqdm(
            range(0, len(self.corpus), effective_batch_size),
            desc="Inference embeddings",
        ):
            yield (
                start_idx,
                _encode_batch(
                    encoder,
                    tokenizer,
                    prepare_texts(
                        self.corpus[start_idx : start_idx + effective_batch_size][
                            "contents"
                        ],
                        self.retrieval_method,
                        is_query=False,
                    ),
                    self.retrieval_method,
                    self.max_length,
                    self.pooling_method,
                    self.device,
                ),
            )

    def encode_all(self, encoder: Any, tokenizer: Any) -> np.ndarray:
        torch = _require_torch()
        batch_size = self.batch_size
        if self.gpu_num > 1 and self.device.startswith("cuda"):
            encoder = torch.nn.DataParallel(encoder)
            batch_size *= self.gpu_num

        all_embeddings: np.ndarray | None = None
        write_index = 0
        for _, batch_embeddings in self._iter_encoded_batches(
            encoder, tokenizer, batch_size=batch_size
        ):
            if all_embeddings is None:
                all_embeddings = np.empty(
                    (len(self.corpus), batch_embeddings.shape[1]),
                    dtype=batch_embeddings.dtype,
                )
            stop_index = write_index + batch_embeddings.shape[0]
            all_embeddings[write_index:stop_index] = batch_embeddings
            write_index = stop_index

        if all_embeddings is None:
            return np.empty((0, 0), dtype=np.float32)
        return all_embeddings

    def encode_all_to_memmap(self, encoder: Any, tokenizer: Any) -> np.memmap:
        torch = _require_torch()
        batch_size = self.batch_size
        if self.gpu_num > 1 and self.device.startswith("cuda"):
            encoder = torch.nn.DataParallel(encoder)
            batch_size *= self.gpu_num

        memmap: np.memmap | None = None
        write_index = 0
        for _, batch_embeddings in self._iter_encoded_batches(
            encoder, tokenizer, batch_size=batch_size
        ):
            if memmap is None:
                memmap = np.memmap(
                    self.embedding_save_path,
                    shape=(len(self.corpus), batch_embeddings.shape[1]),
                    mode="w+",
                    dtype=batch_embeddings.dtype,
                )
            stop_index = write_index + batch_embeddings.shape[0]
            memmap[write_index:stop_index] = batch_embeddings
            write_index = stop_index

        if memmap is None:
            raise ValueError("Cannot build a dense index for an empty corpus.")
        memmap.flush()
        return memmap

    def build_dense_index(self) -> None:
        is_hnsw = "HNSW" in self.faiss_type.upper()
        if self.faiss_gpu and is_hnsw:
            raise ValueError(
                "faiss_gpu=True is not compatible with HNSW indexes. "
                "FAISS cannot GPU-shard HNSW indexes. "
                "Use faiss_type='Flat' or an IVF variant for GPU indexing."
            )

        torch = _require_torch()

        if self.index_save_path.exists():
            warnings.warn(
                f"{self.index_save_path} already exists and will be overwritten.",
                UserWarning,
            )

        if self.embedding_path is not None:
            corpus_size = len(self.corpus)
            all_embeddings = self._load_embedding(self.embedding_path, corpus_size)
        else:
            encoder, tokenizer = load_model(
                model_path=self.model_path or "",
                use_fp16=self.use_fp16,
                device=self.device,
            )
            with torch.no_grad():
                if self.save_embedding:
                    all_embeddings = self.encode_all_to_memmap(encoder, tokenizer)
                else:
                    all_embeddings = self.encode_all(encoder, tokenizer)
            del self.corpus

        write_dense_faiss_index(
            all_embeddings,
            self.index_save_path,
            faiss_type=self.faiss_type,
            hnsw_ef_construction=self.hnsw_ef_construction,
            hnsw_ef_search=self.hnsw_ef_search,
            faiss_gpu=self.faiss_gpu,
            gpu_num=self.gpu_num,
        )


def resolve_pooling_method(retrieval_method: str, pooling_method: str | None) -> str:
    if pooling_method is None:
        for model_key, default_pooling in MODEL2POOLING.items():
            if model_key in retrieval_method.lower():
                return default_pooling
        return "mean"
    if pooling_method not in {"mean", "cls", "pooler"}:
        raise NotImplementedError("pooling_method must be one of: mean, cls, pooler")
    return pooling_method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a retrieval index.")
    parser.add_argument("--retrieval_method", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="indexes/")
    parser.add_argument("--max_length", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--use_fp16", default=False, action="store_true")
    parser.add_argument("--pooling_method", type=str, default=None)
    parser.add_argument("--faiss_type", type=str, default="Flat")
    parser.add_argument(
        "--hnsw_ef_construction",
        type=int,
        default=None,
        help=(
            "HNSW efConstruction: controls index build quality. "
            "Higher values improve recall but slow down indexing. "
            "Use with --faiss_type HNSW64."
        ),
    )
    parser.add_argument(
        "--hnsw_ef_search",
        type=int,
        default=None,
        help=(
            "HNSW efSearch: controls recall vs. latency at query time. "
            "Higher values improve recall but slow down search. "
            "Stored in the index file and used by the retrieval server."
        ),
    )
    parser.add_argument("--embedding_path", type=str, default=None)
    parser.add_argument("--save_embedding", action="store_true", default=False)
    parser.add_argument("--faiss_gpu", default=False, action="store_true")
    parser.add_argument(
        "--save_vocabulary", dest="save_vocabulary", action="store_true", default=True
    )
    parser.add_argument(
        "--no_save_vocabulary", dest="save_vocabulary", action="store_false"
    )
    parser.add_argument("--keyword_limit", type=int, default=10)
    parser.add_argument(
        "--vocab_max_length", type=int, default=DEFAULT_VOCAB_MAX_LENGTH
    )
    parser.add_argument(
        "--bm25_threads",
        type=int,
        default=0,
        help="BM25 indexing threads (0 = auto-detect CPUs)",
    )
    return parser.parse_args()


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(override=True)
    args = parse_args()
    config = IndexBuilderConfig(
        retrieval_method=args.retrieval_method,
        model_path=args.model_path,
        corpus_path=args.corpus_path,
        save_dir=args.save_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        use_fp16=args.use_fp16,
        pooling_method=resolve_pooling_method(
            args.retrieval_method, args.pooling_method
        ),
        faiss_type=args.faiss_type,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search,
        embedding_path=args.embedding_path,
        save_embedding=args.save_embedding,
        faiss_gpu=args.faiss_gpu,
        save_vocabulary=args.save_vocabulary,
        keyword_limit=args.keyword_limit,
        vocab_max_length=args.vocab_max_length,
        bm25_threads=args.bm25_threads,
    )
    IndexBuilder(config).build_index()


if __name__ == "__main__":
    main()
