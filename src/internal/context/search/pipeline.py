import logging
import os
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Any
from typing import Callable

from sqlalchemy.orm import Session

from src.internal.context.search.models import BaseFilters
from src.internal.context.search.models import ChunkIndexRequest
from src.internal.context.search.models import ChunkSearchRequest
from src.internal.context.search.models import IndexFilters
from src.internal.context.search.models import InferenceChunk
from src.internal.context.search.models import InferenceSection
from src.internal.context.search.models import PersonaSearchInfo
from src.internal.context.search.preprocessing.access_filters import (
    build_access_filters_for_user,
)
from src.internal.context.search.preprocessing.access_filters import User
from src.internal.context.search.retrieval.search_runner import FederatedRetrievalInfo
from src.internal.context.search.retrieval.search_runner import search_chunks
from src.internal.context.search.utils import inference_section_from_chunks
from src.internal.document_index.interfaces import DocumentIndex
from src.internal.llm.interfaces import LLM
from src.internal.natural_language_processing.english_stopwords import strip_stopwords
from src.internal.natural_language_processing.search_nlp_models import EmbeddingModel
from src.internal.servers.error_handling import AgenticSearchError
from src.internal.servers.error_handling import ErrorCode
from src.internal.servers.secondary_llm_flows import extract_source_filter
from src.internal.servers.secondary_llm_flows import extract_time_filter
from shared_configs.contextvars import get_current_tenant_id

logger = logging.getLogger(__name__)

MULTI_TENANT: bool = os.environ.get("MULTI_TENANT", "false").lower() in (
    "1",
    "true",
    "yes",
)


@dataclass
class FunctionCall:
    """Wrap a callable + args for parallel execution via run_functions_in_parallel."""

    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))


def run_functions_in_parallel(function_calls: list[FunctionCall]) -> dict[str, Any]:
    if not function_calls:
        return {}
    with ThreadPoolExecutor(max_workers=len(function_calls)) as executor:
        futures = {
            fc.result_id: executor.submit(fc.func, *fc.args, **fc.kwargs)
            for fc in function_calls
        }
    return {rid: fut.result() for rid, fut in futures.items()}


def fetch_ee_implementation_or_noop(
    _module: str, _func: str, noop_return_value: Any
) -> Callable:
    def noop(*_args: Any, **_kwargs: Any) -> Any:
        return noop_return_value

    return noop


def _filter_document_set_names_by_user_access(
    db_session: Session,
    document_set_names: list[str],
    user: User,
) -> list[str]:
    raise NotImplementedError(
        "filter_document_set_names_by_user_access requires a DB session"
    )


def _build_index_filters(
    user_provided_filters: BaseFilters | None,
    user: User,
    project_id_filter: int | None,
    persona_id_filter: int | None,
    persona_document_sets: list[str] | None,
    persona_time_cutoff: datetime | None,
    db_session: Session | None = None,
    auto_detect_filters: bool = False,
    query: str | None = None,
    llm: LLM | None = None,
    bypass_acl: bool = False,
    attached_document_ids: list[str] | None = None,
    hierarchy_node_ids: list[int] | None = None,
    acl_filters: list[str] | None = None,
) -> IndexFilters:
    if auto_detect_filters and (llm is None or query is None):
        raise RuntimeError("LLM and query are required for auto detect filters")

    base_filters = user_provided_filters or BaseFilters()

    if (
        base_filters.document_set is not None
        and not bypass_acl
        and not user.is_anonymous
        and db_session is not None
    ):
        accessible_names = _filter_document_set_names_by_user_access(
            db_session=db_session,
            document_set_names=base_filters.document_set,
            user=user,
        )
        unauthorized = sorted(
            name for name in base_filters.document_set if name not in accessible_names
        )
        if unauthorized:
            raise AgenticSearchError(
                ErrorCode.INSUFFICIENT_PERMISSIONS,
                f"User does not have access to document sets: {unauthorized}",
            )

    document_set_filter = (
        base_filters.document_set
        if base_filters.document_set is not None
        else persona_document_sets
    )

    time_filter = base_filters.time_cutoff or persona_time_cutoff
    source_filter = base_filters.source_type

    detected_time_filter = None
    detected_source_filter = None
    if auto_detect_filters:
        time_filter_fnc = FunctionCall(extract_time_filter, (query, llm), {})
        if not source_filter:
            source_filter_fnc: FunctionCall | None = FunctionCall(
                extract_source_filter, (query, llm, db_session), {}
            )
        else:
            source_filter_fnc = None

        functions_to_run = [fn for fn in [time_filter_fnc, source_filter_fnc] if fn]
        parallel_results = run_functions_in_parallel(functions_to_run)
        detected_time_filter, _detected_favor_recent = parallel_results[
            time_filter_fnc.result_id
        ]
        if source_filter_fnc:
            detected_source_filter = parallel_results[source_filter_fnc.result_id]

    if time_filter and detected_time_filter and detected_time_filter > time_filter:
        time_filter = detected_time_filter

    if not source_filter and detected_source_filter:
        source_filter = detected_source_filter

    if bypass_acl:
        user_acl_filters = None
    elif acl_filters is not None:
        user_acl_filters = acl_filters
    else:
        if db_session is None:
            raise ValueError("Either db_session or acl_filters must be provided")
        user_acl_filters = build_access_filters_for_user(user, db_session)

    final_filters = IndexFilters(
        project_id_filter=project_id_filter,
        persona_id_filter=persona_id_filter,
        source_type=source_filter,
        document_set=document_set_filter,
        time_cutoff=time_filter,
        tags=base_filters.tags,
        access_control_list=user_acl_filters,
        tenant_id=get_current_tenant_id() if MULTI_TENANT else None,
        attached_document_ids=attached_document_ids,
        hierarchy_node_ids=hierarchy_node_ids,
    )

    return final_filters


def merge_individual_chunks(
    chunks: list[InferenceChunk],
) -> list[InferenceSection]:
    """Merge adjacent chunks from the same document into sections."""
    if not chunks:
        return []

    chunk_to_original_index: dict[tuple[str, int], int] = {}
    for idx, chunk in enumerate(chunks):
        chunk_to_original_index[(chunk.document_id, chunk.chunk_id)] = idx

    doc_chunks: dict[str, list[InferenceChunk]] = defaultdict(list)
    for chunk in chunks:
        doc_chunks[chunk.document_id].append(chunk)

    for doc_id in doc_chunks:
        doc_chunks[doc_id].sort(key=lambda c: c.chunk_id)

    chunk_to_section: dict[tuple[str, int], InferenceSection] = {}

    for doc_id, doc_chunk_list in doc_chunks.items():
        if not doc_chunk_list:
            continue

        current_section_chunks = [doc_chunk_list[0]]

        for i in range(1, len(doc_chunk_list)):
            prev_chunk = doc_chunk_list[i - 1]
            curr_chunk = doc_chunk_list[i]

            if curr_chunk.chunk_id == prev_chunk.chunk_id + 1:
                current_section_chunks.append(curr_chunk)
            else:
                center_chunk = min(
                    current_section_chunks,
                    key=lambda c: chunk_to_original_index.get(
                        (c.document_id, c.chunk_id), float("inf")
                    ),
                )
                section = inference_section_from_chunks(
                    center_chunk=center_chunk,
                    chunks=current_section_chunks.copy(),
                )
                if section:
                    for chunk in current_section_chunks:
                        chunk_to_section[(chunk.document_id, chunk.chunk_id)] = section

                current_section_chunks = [curr_chunk]

        if current_section_chunks:
            center_chunk = min(
                current_section_chunks,
                key=lambda c: chunk_to_original_index.get(
                    (c.document_id, c.chunk_id), float("inf")
                ),
            )
            section = inference_section_from_chunks(
                center_chunk=center_chunk,
                chunks=current_section_chunks.copy(),
            )
            if section:
                for chunk in current_section_chunks:
                    chunk_to_section[(chunk.document_id, chunk.chunk_id)] = section

    seen_section_ids: set[tuple[str, int]] = set()
    result: list[InferenceSection] = []

    for chunk in chunks:
        section = chunk_to_section.get((chunk.document_id, chunk.chunk_id))
        if section:
            section_id = (
                section.center_chunk.document_id,
                section.center_chunk.chunk_id,
            )
            if section_id not in seen_section_ids:
                seen_section_ids.add(section_id)
                result.append(section)
        else:
            single_section = inference_section_from_chunks(
                center_chunk=chunk,
                chunks=[chunk],
            )
            if single_section:
                single_section_id = (
                    single_section.center_chunk.document_id,
                    single_section.center_chunk.chunk_id,
                )
                if single_section_id not in seen_section_ids:
                    seen_section_ids.add(single_section_id)
                    result.append(single_section)

    return result


def search_pipeline(
    chunk_search_request: ChunkSearchRequest,
    document_index: DocumentIndex,
    user: User,
    persona_search_info: PersonaSearchInfo | None,
    db_session: Session | None = None,
    auto_detect_filters: bool = False,
    llm: LLM | None = None,
    project_id_filter: int | None = None,
    persona_id_filter: int | None = None,
    acl_filters: list[str] | None = None,
    embedding_model: EmbeddingModel | None = None,
    prefetched_federated_retrieval_infos: list[FederatedRetrievalInfo] | None = None,
) -> list[InferenceChunk]:
    persona_document_sets: list[str] | None = (
        persona_search_info.document_set_names if persona_search_info else None
    )
    persona_time_cutoff: datetime | None = (
        persona_search_info.search_start_date if persona_search_info else None
    )
    attached_document_ids: list[str] | None = (
        persona_search_info.attached_document_ids or None
        if persona_search_info
        else None
    )
    hierarchy_node_ids: list[int] | None = (
        persona_search_info.hierarchy_node_ids or None if persona_search_info else None
    )

    filters = _build_index_filters(
        user_provided_filters=chunk_search_request.user_selected_filters,
        user=user,
        project_id_filter=project_id_filter,
        persona_id_filter=persona_id_filter,
        persona_document_sets=persona_document_sets,
        persona_time_cutoff=persona_time_cutoff,
        db_session=db_session,
        auto_detect_filters=auto_detect_filters,
        query=chunk_search_request.query,
        llm=llm,
        bypass_acl=chunk_search_request.bypass_acl,
        attached_document_ids=attached_document_ids,
        hierarchy_node_ids=hierarchy_node_ids,
        acl_filters=acl_filters,
    )

    query_keywords = strip_stopwords(chunk_search_request.query)

    query_request = ChunkIndexRequest(
        query=chunk_search_request.query,
        hybrid_alpha=chunk_search_request.hybrid_alpha,
        recency_bias_multiplier=chunk_search_request.recency_bias_multiplier,
        query_keywords=query_keywords,
        filters=filters,
        limit=chunk_search_request.limit,
    )

    retrieved_chunks = search_chunks(
        query_request=query_request,
        user_id=user.id if user else None,
        document_index=document_index,
        db_session=db_session,
        embedding_model=embedding_model,
        prefetched_federated_retrieval_infos=prefetched_federated_retrieval_infos,
    )

    censored_chunks: list[InferenceChunk] = fetch_ee_implementation_or_noop(
        "onyx.external_permissions.post_query_censoring",
        "_post_query_chunk_censoring",
        retrieved_chunks,
    )(
        chunks=retrieved_chunks,
        user=user,
    )

    return censored_chunks
