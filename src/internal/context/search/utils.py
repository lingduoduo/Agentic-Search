import logging
import os
import re
from typing import Any, Callable, TypeVar

from sqlalchemy.orm import Session

from src.internal.context.search.models import InferenceChunk
from src.internal.context.search.models import InferenceSection
from src.internal.context.search.models import SavedSearchDoc
from src.internal.context.search.models import SavedSearchDocWithContent
from src.internal.context.search.models import SearchDoc
from src.internal.natural_language_processing.query_embedding_cache import (
    cache_query_embeddings,
)
from src.internal.natural_language_processing.query_embedding_cache import (
    get_cached_query_embeddings,
)
from src.internal.natural_language_processing.query_embedding_cache import (
    record_cache_skipped,
)
from src.internal.natural_language_processing.search_nlp_models import EmbeddingModel
from shared_configs.configs import MODEL_SERVER_HOST
from shared_configs.configs import MODEL_SERVER_PORT
from shared_configs.enums import EmbedTextType
from shared_configs.model_server_models import Embedding

logger = logging.getLogger(__name__)

QUERY_EMBEDDING_CACHE_ENABLED: bool = os.environ.get(
    "QUERY_EMBEDDING_CACHE_ENABLED", "false"
).lower() in ("1", "true", "yes")
QUERY_EMBEDDING_CACHE_TTL_S: int = int(
    os.environ.get("QUERY_EMBEDDING_CACHE_TTL_S", "600")
)


def log_function_time(*_args: Any, **_kwargs: Any) -> Callable:
    def decorator(func: Callable) -> Callable:
        return func

    return decorator


T = TypeVar(
    "T",
    InferenceSection,
    InferenceChunk,
    SearchDoc,
    SavedSearchDoc,
    SavedSearchDocWithContent,
)

TSection = TypeVar(
    "TSection",
    InferenceSection,
    SearchDoc,
    SavedSearchDoc,
    SavedSearchDocWithContent,
)

_UNSAFE_CHARS_RE = re.compile(r"[\x00-\x1f/\\:\*\?\"<>\|]+")
_SANDBOX_FILENAME_MAX_LENGTH = 200


def inference_section_from_chunks(
    center_chunk: InferenceChunk,
    chunks: list[InferenceChunk],
) -> InferenceSection | None:
    if not chunks:
        return None

    combined_content = "\n".join([chunk.content for chunk in chunks])

    return InferenceSection(
        center_chunk=center_chunk,
        chunks=chunks,
        combined_content=combined_content,
    )


def inference_section_from_single_chunk(
    chunk: InferenceChunk,
) -> InferenceSection:
    return InferenceSection(
        center_chunk=chunk,
        chunks=[chunk],
        combined_content=chunk.content,
    )


def _get_current_search_settings(db_session: Session) -> Any:
    raise NotImplementedError(
        "get_current_search_settings requires a DB session — not yet wired to src.internal.db"
    )


def get_query_embeddings(
    queries: list[str],
    db_session: Session | None = None,
    embedding_model: EmbeddingModel | None = None,
) -> list[Embedding]:
    search_settings: Any = None
    if embedding_model is None:
        if db_session is None:
            raise ValueError("Either db_session or embedding_model must be provided")
        search_settings = _get_current_search_settings(db_session)
        embedding_model = EmbeddingModel.from_db_model(
            search_settings=search_settings,
            server_host=MODEL_SERVER_HOST,
            server_port=MODEL_SERVER_PORT,
        )
    elif db_session is not None:
        search_settings = _get_current_search_settings(db_session)

    result: list[Embedding] = []
    cache_usable: bool = (
        QUERY_EMBEDDING_CACHE_ENABLED and bool(queries) and search_settings is not None
    )
    if not cache_usable:
        if queries:
            record_cache_skipped(embedding_model.provider_type, count=len(queries))
        result = embedding_model.encode(queries, text_type=EmbedTextType.QUERY)
        assert len(result) == len(queries), (
            "Bug: The length of embeddings does not match the length of queries."
        )
        return result
    assert search_settings is not None, "Bug: search_settings is None."

    cached = get_cached_query_embeddings(
        queries=queries,
        search_settings_id=search_settings.id,
        provider_type=embedding_model.provider_type,
        ttl_seconds=QUERY_EMBEDDING_CACHE_TTL_S,
    )

    miss_indices = [i for i, value in enumerate(cached) if value is None]
    if not miss_indices:
        result = [emb for emb in cached if emb is not None]
        assert len(result) == len(queries), (
            "Bug: The length of embeddings does not match the length of queries."
        )
        return result

    miss_queries = [queries[i] for i in miss_indices]
    fresh_embeddings = embedding_model.encode(
        miss_queries, text_type=EmbedTextType.QUERY
    )

    cache_query_embeddings(
        queries=miss_queries,
        embeddings=fresh_embeddings,
        search_settings_id=search_settings.id,
        provider_type=embedding_model.provider_type,
        ttl_seconds=QUERY_EMBEDDING_CACHE_TTL_S,
    )

    fresh_iter = iter(fresh_embeddings)
    for value in cached:
        if value is None:
            result.append(next(fresh_iter))
        else:
            result.append(value)
    assert len(result) == len(queries), (
        "Bug: The length of embeddings does not match the length of queries."
    )
    return result


@log_function_time(print_only=True, debug_only=True)
def get_query_embedding(
    query: str,
    db_session: Session | None = None,
    embedding_model: EmbeddingModel | None = None,
) -> Embedding:
    return get_query_embeddings(
        [query], db_session=db_session, embedding_model=embedding_model
    )[0]


def convert_inference_sections_to_search_docs(
    inference_sections: list[InferenceSection],
    is_internet: bool = False,
) -> list[SearchDoc]:
    search_docs = SearchDoc.from_chunks_or_sections(inference_sections)
    for search_doc in search_docs:
        search_doc.is_internet = is_internet
    return search_docs


def sandbox_filename_for_document(title: str, file_id: str) -> str:
    """Sanitize a document title and append its file_id to produce a globally
    unique sandbox filename. Extensions on the title are preserved verbatim."""
    sanitized = _UNSAFE_CHARS_RE.sub("_", title).strip().strip(".")
    base, ext = os.path.splitext(sanitized)
    if not base:
        base = "document"
    suffix = f"_{file_id}{ext}"
    max_base_len = max(1, _SANDBOX_FILENAME_MAX_LENGTH - len(suffix))
    return f"{base[:max_base_len]}{suffix}"


def populate_file_ids_on_sections(
    sections: list[InferenceSection],
    db_session: Session,
) -> None:
    raise NotImplementedError(
        "populate_file_ids_on_sections requires a DB session — not yet wired to src.internal.db"
    )
