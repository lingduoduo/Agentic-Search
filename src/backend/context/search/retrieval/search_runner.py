import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from src.backend.context.search.enums import QueryType
from src.backend.context.search.models import ChunkIndexRequest
from src.backend.context.search.models import IndexFilters
from src.backend.context.search.models import InferenceChunk
from src.backend.context.search.models import InferenceSection
from src.backend.context.search.utils import get_query_embedding
from src.backend.context.search.utils import inference_section_from_chunks
from src.backend.document_index.interfaces import DocumentIndex
from src.backend.document_index.interfaces import DocumentSectionRequest
from src.backend.natural_language_processing.search_nlp_models import EmbeddingModel
from src.backend.chat.chat_utils import run_functions_tuples_in_parallel

logger = logging.getLogger(__name__)

HYBRID_ALPHA: float = float(os.environ.get("HYBRID_ALPHA", "0.5"))
NUM_RETURNED_HITS: int = int(os.environ.get("NUM_RETURNED_HITS", "10"))


@dataclass
class FederatedRetrievalInfo:
    """Stub — federated connector retrieval is not wired in this repo."""

    source: object
    retrieval_function: Callable


def get_federated_retrieval_functions(
    db_session: Session,
    user_id: UUID | None,
    source_types: list | None,
    document_set_names: list[str] | None,
) -> list[FederatedRetrievalInfo]:
    return []


def combine_retrieval_results(
    chunk_sets: list[list[InferenceChunk]],
) -> list[InferenceChunk]:
    all_chunks = [chunk for chunk_set in chunk_sets for chunk in chunk_set]

    unique_chunks: dict[tuple[str, int], InferenceChunk] = {}
    for chunk in all_chunks:
        key = (chunk.document_id, chunk.chunk_id)
        if key not in unique_chunks:
            unique_chunks[key] = chunk
            continue

        stored_chunk_score = unique_chunks[key].score or 0
        this_chunk_score = chunk.score or 0
        if stored_chunk_score < this_chunk_score:
            unique_chunks[key] = chunk

    sorted_chunks = sorted(
        unique_chunks.values(), key=lambda x: x.score or 0, reverse=True
    )

    return sorted_chunks


def _embed_and_hybrid_search(
    query_request: ChunkIndexRequest,
    document_index: DocumentIndex,
    db_session: Session | None = None,
    embedding_model: EmbeddingModel | None = None,
) -> list[InferenceChunk]:
    query_embedding = get_query_embedding(
        query_request.query,
        db_session=db_session,
        embedding_model=embedding_model,
    )

    hybrid_alpha = query_request.hybrid_alpha or HYBRID_ALPHA

    query_type = QueryType.KEYWORD if hybrid_alpha <= 0.2 else QueryType.SEMANTIC
    top_chunks = document_index.hybrid_retrieval(
        query=query_request.query,
        query_embedding=query_embedding,
        final_keywords=query_request.query_keywords,
        query_type=query_type,
        filters=query_request.filters,
        num_to_retrieve=query_request.limit or NUM_RETURNED_HITS,
    )

    return top_chunks


def _keyword_search(
    query_request: ChunkIndexRequest,
    document_index: DocumentIndex,
) -> list[InferenceChunk]:
    return document_index.keyword_retrieval(
        query=query_request.query,
        filters=query_request.filters,
        num_to_retrieve=query_request.limit or NUM_RETURNED_HITS,
    )


def search_chunks(
    query_request: ChunkIndexRequest,
    user_id: UUID | None,
    document_index: DocumentIndex,
    db_session: Session | None = None,
    embedding_model: EmbeddingModel | None = None,
    prefetched_federated_retrieval_infos: list[FederatedRetrievalInfo] | None = None,
) -> list[InferenceChunk]:
    run_queries: list[tuple[Callable, tuple]] = []

    source_filters = (
        set(query_request.filters.source_type)
        if query_request.filters.source_type
        else None
    )

    if prefetched_federated_retrieval_infos is not None:
        federated_retrieval_infos = prefetched_federated_retrieval_infos
    else:
        if db_session is None:
            raise ValueError(
                "Either db_session or prefetched_federated_retrieval_infos must be provided"
            )
        federated_retrieval_infos = get_federated_retrieval_functions(
            db_session=db_session,
            user_id=user_id,
            source_types=list(source_filters) if source_filters else None,
            document_set_names=query_request.filters.document_set,
        )

    federated_sources = set(
        federated_retrieval_info.source.to_non_federated_source()
        for federated_retrieval_info in federated_retrieval_infos
    )
    for federated_retrieval_info in federated_retrieval_infos:
        run_queries.append(
            (federated_retrieval_info.retrieval_function, (query_request,))
        )

    normal_search_enabled = (source_filters is None) or (
        len(set(source_filters) - federated_sources) > 0
    )

    if normal_search_enabled:
        if query_request.hybrid_alpha is not None and query_request.hybrid_alpha == 0.0:
            run_queries.append(
                (
                    lambda: _keyword_search(query_request, document_index),
                    (),
                )
            )
        else:
            run_queries.append(
                (
                    _embed_and_hybrid_search,
                    (query_request, document_index, db_session, embedding_model),
                )
            )

    parallel_search_results = run_functions_tuples_in_parallel(run_queries)
    top_chunks = combine_retrieval_results(parallel_search_results)

    if not top_chunks:
        logger.debug(
            "Search returned no results for query: %s with filters: %s.",
            query_request.query,
            query_request.filters,
        )

    return top_chunks


def inference_sections_from_ids(
    doc_identifiers: list[tuple[str, int]],
    document_index: DocumentIndex,
) -> list[InferenceSection]:
    doc_ids_set = set(doc_id for doc_id, _ in doc_identifiers)

    chunk_requests: list[DocumentSectionRequest] = [
        DocumentSectionRequest(document_id=doc_id) for doc_id in doc_ids_set
    ]

    filters = IndexFilters(access_control_list=None)

    retrieved_chunks = document_index.id_based_retrieval(
        chunk_requests=chunk_requests,
        filters=filters,
    )

    if not retrieved_chunks:
        return []

    chunks_by_doc_id: dict[str, list[InferenceChunk]] = {}
    for chunk in retrieved_chunks:
        chunks_by_doc_id.setdefault(chunk.document_id, []).append(chunk)

    inference_sections = [
        section  # ty: ignore[possibly-unresolved-reference]
        for chunks in chunks_by_doc_id.values()
        if chunks
        and (
            section := inference_section_from_chunks(
                center_chunk=chunks[0],
                chunks=chunks,
            )
        )
    ]

    return inference_sections
