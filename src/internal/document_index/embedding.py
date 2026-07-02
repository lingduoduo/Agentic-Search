"""Embedding: model loading, pooling, batch encoding, and chunk embedding with
document-level failure isolation.

Split out of ``index_builder`` to isolate the (optionally torch-backed)
embedding concerns from chunking and FAISS I/O. Heavy imports stay lazy via the
``_require_*`` guards in ``_common``.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from src.internal.connectors.models import ConnectorFailure
from src.internal.document_index._common import (
    IndexingHeartbeatInterface,
    _raise_if_indexing_stopped,
    _report_indexing_progress,
    _require_torch,
    _require_transformers,
)
from src.internal.document_index.models import (
    EmbeddedChunk,
    EmbeddingConfig,
    IndexChunk,
)

logger = logging.getLogger(__name__)

MODEL2POOLING = {
    "e5": "mean",
    "bge": "cls",
    "contriever": "mean",
    "jina": "mean",
}

EmbeddingFn = Callable[[list[str]], np.ndarray]


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


def _batched(items: list[IndexChunk], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


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


def resolve_pooling_method(retrieval_method: str, pooling_method: str | None) -> str:
    if pooling_method is None:
        for model_key, default_pooling in MODEL2POOLING.items():
            if model_key in retrieval_method.lower():
                return default_pooling
        return "mean"
    if pooling_method not in {"mean", "cls", "pooler"}:
        raise NotImplementedError("pooling_method must be one of: mean, cls, pooler")
    return pooling_method
