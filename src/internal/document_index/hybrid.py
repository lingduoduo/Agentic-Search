"""HybridDocumentIndex — OpenSearch as keyword/KV store, Weaviate as vector store.

Routing strategy
----------------
| Operation            | Backend          | Rationale                           |
|----------------------|------------------|-------------------------------------|
| index / delete       | Both             | Both stores must stay in sync       |
| update               | Both             | Metadata must match in both stores  |
| keyword_retrieval    | OpenSearch only  | BM25 is OpenSearch's strength       |
| semantic_retrieval   | Weaviate only    | HNSW vector search                  |
| hybrid_retrieval     | Both + RRF merge | Best of both recall strategies      |
| id_based_retrieval   | OpenSearch only  | Key-value lookup by doc ID          |
| random_retrieval     | OpenSearch only  | Simpler, no need for Weaviate here  |
| verify/create schema | Both             | Both indices must exist             |
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from src.internal.document_index.interfaces import (
    DocumentIndex,
    DocumentInsertionRecord,
    DocumentSectionRequest,
    IndexingMetadata,
    MetadataUpdateRequest,
)
from src.internal.document_index.models import (
    DocMetadataAwareIndexChunk,
    Embedding,
    EmbeddingPrecision,
    IndexFilters,
    InferenceChunk,
    QueryType,
)
from src.internal.retrieval.fusion import rrf_rank

logger = logging.getLogger(__name__)

_RRF_K = 60  # standard constant from the original RRF paper


class HybridDocumentIndex(DocumentIndex):
    """DocumentIndex that uses OpenSearch for keyword/KV operations and Weaviate
    for vector/semantic operations.

    Both backends are kept in sync on every write.  Reads are routed to the
    backend best suited for the query type.  ``hybrid_retrieval`` queries both
    and merges results using Reciprocal Rank Fusion (RRF).

    Args:
        opensearch_index: A fully configured OpenSearchDocumentIndex instance.
        weaviate_index:   A fully configured WeaviateDocumentIndex instance.
    """

    def __init__(
        self,
        opensearch_index: DocumentIndex,
        weaviate_index: DocumentIndex,
    ) -> None:
        self._os = opensearch_index
        self._wv = weaviate_index

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def verify_and_create_index_if_necessary(
        self,
        embedding_dim: int,
        embedding_precision: EmbeddingPrecision,
    ) -> None:
        self._os.verify_and_create_index_if_necessary(
            embedding_dim, embedding_precision
        )
        self._wv.verify_and_create_index_if_necessary(
            embedding_dim, embedding_precision
        )

    # ------------------------------------------------------------------
    # Writes — both stores must stay in sync
    # ------------------------------------------------------------------

    def index(
        self,
        chunks: Iterable[DocMetadataAwareIndexChunk],
        indexing_metadata: IndexingMetadata,
    ) -> list[DocumentInsertionRecord]:
        # Materialise once so both backends see the same data.
        chunk_list = list(chunks)
        os_records = self._os.index(iter(chunk_list), indexing_metadata)
        self._wv.index(iter(chunk_list), indexing_metadata)
        return os_records

    def delete(
        self,
        document_id: str,
        chunk_count: int | None = None,
    ) -> int:
        deleted = self._os.delete(document_id, chunk_count)
        self._wv.delete(document_id, chunk_count)
        return deleted

    def update(self, update_requests: list[MetadataUpdateRequest]) -> None:
        self._os.update(update_requests)
        self._wv.update(update_requests)

    # ------------------------------------------------------------------
    # Keyword retrieval — OpenSearch only (BM25)
    # ------------------------------------------------------------------

    def keyword_retrieval(
        self,
        query: str,
        filters: IndexFilters,
        num_to_retrieve: int,
        include_hidden: bool = False,
    ) -> list[InferenceChunk]:
        return self._os.keyword_retrieval(
            query,
            filters,
            num_to_retrieve,
            include_hidden=include_hidden,
        )

    # ------------------------------------------------------------------
    # Semantic retrieval — Weaviate only (HNSW)
    # ------------------------------------------------------------------

    def semantic_retrieval(
        self,
        query_embedding: Embedding,
        filters: IndexFilters,
        num_to_retrieve: int,
    ) -> list[InferenceChunk]:
        return self._wv.semantic_retrieval(query_embedding, filters, num_to_retrieve)

    # ------------------------------------------------------------------
    # Hybrid retrieval — keyword (OS) + semantic (Weaviate) fused via RRF
    # ------------------------------------------------------------------

    def hybrid_retrieval(
        self,
        query: str,
        query_embedding: Embedding,
        final_keywords: list[str] | None,
        query_type: QueryType,
        filters: IndexFilters,
        num_to_retrieve: int,
    ) -> list[InferenceChunk]:
        # Fetch more candidates than needed so RRF has a larger pool to work with.
        fetch_k = min(num_to_retrieve * 2, num_to_retrieve + 20)

        keyword_chunks = self._os.keyword_retrieval(query, filters, fetch_k)
        semantic_chunks = self._wv.semantic_retrieval(query_embedding, filters, fetch_k)

        return _rrf_merge(keyword_chunks, semantic_chunks, top_k=num_to_retrieve)

    # ------------------------------------------------------------------
    # ID-based retrieval — OpenSearch as KV store
    # ------------------------------------------------------------------

    def id_based_retrieval(
        self,
        chunk_requests: list[DocumentSectionRequest],
        filters: IndexFilters,
        batch_retrieval: bool = False,
    ) -> list[InferenceChunk]:
        return self._os.id_based_retrieval(chunk_requests, filters, batch_retrieval)

    # ------------------------------------------------------------------
    # Random retrieval — OpenSearch
    # ------------------------------------------------------------------

    def random_retrieval(
        self,
        filters: IndexFilters,
        num_to_retrieve: int = 10,
        dirty: bool | None = None,
    ) -> list[InferenceChunk]:
        return self._os.random_retrieval(filters, num_to_retrieve, dirty)


# ---------------------------------------------------------------------------
# RRF merge helper
# ---------------------------------------------------------------------------


def _rrf_merge(
    keyword_chunks: list[InferenceChunk],
    semantic_chunks: list[InferenceChunk],
    top_k: int,
    k: int = _RRF_K,
) -> list[InferenceChunk]:
    """Reciprocal Rank Fusion over two ranked lists.

    For each document chunk that appears in either list the RRF score is:
        score = sum(1 / (k + rank_i))

    where rank_i is the 1-based rank in each list that includes the chunk.
    Chunks that appear in both lists get a bonus from both terms.

    The returned list is ordered by descending RRF score and truncated to
    ``top_k`` entries.  OpenSearch chunks (keyword results) are preferred as
    the canonical object when a chunk appears in both lists, because they
    carry richer metadata (highlights, doc_summary, etc.).
    """

    def _key(chunk: InferenceChunk) -> tuple[str, int]:
        return (chunk.document_id, chunk.chunk_ind)

    # OpenSearch (keyword) results preferred as the canonical object per key.
    chunk_map: dict[tuple[str, int], InferenceChunk] = {}
    for chunk in keyword_chunks:
        chunk_map[_key(chunk)] = chunk
    for chunk in semantic_chunks:
        chunk_map.setdefault(_key(chunk), chunk)  # don't overwrite OS result

    ranked = rrf_rank([keyword_chunks, semantic_chunks], _key, rrf_k=k)
    return [
        chunk_map[key].model_copy(update={"score": score})
        for key, score in ranked[:top_k]
    ]
