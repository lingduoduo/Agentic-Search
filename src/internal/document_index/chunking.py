"""Document chunking: split documents into token-budgeted chunks ready for
indexing, plus the paragraph/sentence-aware text splitter that powers it.

Split out of ``index_builder`` so the chunking subsystem stands on its own,
independent of embedding and FAISS concerns.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from src.internal.connectors.models import ConnectorFailure, Document
from src.internal.document_index._common import (
    IndexingHeartbeatInterface,
    _raise_if_indexing_stopped,
    _report_indexing_progress,
)
from src.internal.document_index.models import ChunkingConfig, IndexChunk

logger = logging.getLogger(__name__)

RETURN_SEPARATOR = "\n\n"
SECTION_SEPARATOR = "\n\n---\n\n"


def chunk_document(
    document: Document,
    config: ChunkingConfig,
    *,
    embedding_fn=None,
) -> list[IndexChunk]:
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

    if config.semantic_chunking and embedding_fn is not None:
        chunk_texts = _split_text_semantic(
            text,
            content_token_limit,
            config.chunk_overlap,
            embedding_fn,
            config.semantic_breakpoint_percentile,
            config.semantic_buffer_size,
        )
    else:
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
    embedding_fn=None,
) -> list[IndexChunk]:
    """Chunk all non-empty documents."""

    chunks: list[IndexChunk] = []
    for document in documents:
        _raise_if_indexing_stopped(callback, "chunk_documents")
        document_chunks = chunk_document(document, config, embedding_fn=embedding_fn)
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


def _document_sentences(text: str) -> list[str]:
    """Flatten a document into an ordered sentence list (paragraph then sentence)."""
    sentences: list[str] = []
    for para in _split_paragraphs(text):
        sentences.extend(_split_sentences_in_paragraph(para))
    return sentences


def _buffered_sentences(sentences: list[str], buffer_size: int) -> list[str]:
    """Combine each sentence with its neighbors to denoise the similarity signal."""
    if buffer_size <= 1:
        return sentences
    half = buffer_size - 1
    combined: list[str] = []
    for i in range(len(sentences)):
        lo = max(0, i - half)
        hi = min(len(sentences), i + half + 1)
        combined.append(" ".join(sentences[lo:hi]))
    return combined


def _adjacent_distances(vectors: np.ndarray) -> np.ndarray:
    """Cosine distance (1 - cos) between each adjacent pair of row vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    unit = vectors / safe
    sims = np.sum(unit[:-1] * unit[1:], axis=1)
    return 1.0 - sims


def _split_text_semantic(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_fn,
    percentile: float,
    buffer_size: int,
) -> list[str]:
    """Chunk at points where adjacent-sentence cosine distance spikes.

    Falls back to the paragraph splitter when embeddings are unavailable, the
    document has fewer than two sentences, or embedding fails.
    """
    if embedding_fn is None:
        return _split_text_paragraphs(text, chunk_size, chunk_overlap)

    sentences = _document_sentences(text)
    if len(sentences) < 2:
        return _split_text_paragraphs(text, chunk_size, chunk_overlap)

    try:
        vectors = np.asarray(embedding_fn(_buffered_sentences(sentences, buffer_size)))
        if vectors.ndim != 2 or vectors.shape[0] != len(sentences):
            raise ValueError("embedding output shape does not match sentence count")
    except Exception as exc:  # noqa: BLE001 — degrade to structural chunking
        logger.warning(
            "Semantic chunking embedding failed (%s); using paragraphs.", exc
        )
        return _split_text_paragraphs(text, chunk_size, chunk_overlap)

    distances = _adjacent_distances(vectors)
    threshold = float(np.percentile(distances, percentile))

    groups: list[list[str]] = []
    current: list[str] = [sentences[0]]
    for i, dist in enumerate(distances):
        if dist > threshold:
            groups.append(current)
            current = []
        current.append(sentences[i + 1])
    if current:
        groups.append(current)

    chunks: list[str] = []
    for group in groups:
        group_text = " ".join(group).strip()
        if not group_text:
            continue
        if _token_count(group_text) > chunk_size:
            chunks.extend(_split_text_paragraphs(group_text, chunk_size, chunk_overlap))
        else:
            chunks.append(group_text)
    return [c for c in chunks if c]


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
