import math
import os
import uuid
from typing import Protocol
from uuid import UUID

from src.retrieval.models import DocMetadataAwareIndexChunk
from src.retrieval.models import MultipassConfig
from src.backend.document_index.vespa.internal_types import EnrichedDocumentIndexingInfo

MULTI_TENANT: bool = os.environ.get("MULTI_TENANT", "").lower() in {"1", "true", "yes"}
ENABLE_MULTIPASS_INDEXING: bool = os.environ.get(
    "ENABLE_MULTIPASS_INDEXING", ""
).lower() in {"1", "true", "yes"}

DEFAULT_BATCH_SIZE = 30
DEFAULT_INDEX_NAME = "danswer_chunk"


class IndexSearchSettings(Protocol):
    """Protocol for objects that carry search index configuration."""

    multipass_indexing: bool
    large_chunks_enabled: bool
    index_name: str


def should_use_multipass(search_settings: "IndexSearchSettings | None") -> bool:
    """Determine multipass usage from settings or env default."""
    if search_settings is not None:
        return getattr(search_settings, "multipass_indexing", False)
    return ENABLE_MULTIPASS_INDEXING


def get_multipass_config(search_settings: "IndexSearchSettings") -> MultipassConfig:
    """Build a MultipassConfig from search settings."""
    multipass = should_use_multipass(search_settings)
    enable_large_chunks = getattr(search_settings, "large_chunks_enabled", False)
    return MultipassConfig(
        multipass_indexing=multipass, enable_large_chunks=enable_large_chunks
    )


def get_both_index_properties(
    primary_index_name: str,
    secondary_index_name: str | None = None,
    primary_multipass: bool = False,
    secondary_multipass: bool = False,
) -> tuple[str, str | None, bool, bool | None]:
    """Return index names and multipass flags (no DB session required)."""
    return (
        primary_index_name,
        secondary_index_name,
        primary_multipass,
        secondary_multipass if secondary_index_name else None,
    )


def translate_boost_count_to_multiplier(boost: int) -> float:
    """Mapping boost integer values to a multiplier according to a sigmoid curve
    Piecewise such that at many downvotes, its 0.5x the score and with many upvotes
    it is 2x the score. This should be in line with the Vespa calculation."""
    # 3 in the equation below stretches it out to hit asymptotes slower
    if boost < 0:
        # 0.5 + sigmoid -> range of 0.5 to 1
        return 0.5 + (1 / (1 + math.exp(-1 * boost / 3)))

    # 2 x sigmoid -> range of 1 to 2
    return 2 / (1 + math.exp(-1 * boost / 3))


# Assembles a list of Vespa chunk IDs for a document
# given the required context. This can be used to directly query
# Vespa's Document API.
def get_document_chunk_ids(
    enriched_document_info_list: list[EnrichedDocumentIndexingInfo],
    tenant_id: str,
    large_chunks_enabled: bool,
) -> list[UUID]:
    doc_chunk_ids = []

    for enriched_document_info in enriched_document_info_list:
        for chunk_index in range(
            enriched_document_info.chunk_start_index,
            enriched_document_info.chunk_end_index,
        ):
            if not enriched_document_info.old_version:
                doc_chunk_ids.append(
                    get_uuid_from_chunk_info(
                        document_id=enriched_document_info.doc_id,
                        chunk_id=chunk_index,
                        tenant_id=tenant_id,
                    )
                )
            else:
                doc_chunk_ids.append(
                    get_uuid_from_chunk_info_old(
                        document_id=enriched_document_info.doc_id,
                        chunk_id=chunk_index,
                    )
                )

            if large_chunks_enabled and chunk_index % 4 == 0:
                large_chunk_id = int(chunk_index / 4)
                large_chunk_reference_ids = [
                    large_chunk_id + i
                    for i in range(4)
                    if large_chunk_id + i < enriched_document_info.chunk_end_index
                ]
                if enriched_document_info.old_version:
                    doc_chunk_ids.append(
                        get_uuid_from_chunk_info_old(
                            document_id=enriched_document_info.doc_id,
                            chunk_id=large_chunk_id,
                            large_chunk_reference_ids=large_chunk_reference_ids,
                        )
                    )
                else:
                    doc_chunk_ids.append(
                        get_uuid_from_chunk_info(
                            document_id=enriched_document_info.doc_id,
                            chunk_id=large_chunk_id,
                            tenant_id=tenant_id,
                            large_chunk_id=large_chunk_id,
                        )
                    )

    return doc_chunk_ids


def get_uuid_from_chunk_info(
    *,
    document_id: str,
    chunk_id: int,
    tenant_id: str,
    large_chunk_id: int | None = None,
) -> UUID:
    """NOTE: be VERY carefuly about changing this function. If changed without a migration,
    this can cause deletion/update/insertion to function incorrectly."""
    doc_str = document_id

    # Web parsing URL duplicate catching
    if doc_str and doc_str[-1] == "/":
        doc_str = doc_str[:-1]

    chunk_index = (
        "large_" + str(large_chunk_id) if large_chunk_id is not None else str(chunk_id)
    )
    unique_identifier_string = "_".join([doc_str, chunk_index])
    if MULTI_TENANT:
        unique_identifier_string += "_" + tenant_id

    uuid_value = uuid.uuid5(uuid.NAMESPACE_X500, unique_identifier_string)
    return uuid_value


def get_uuid_from_chunk_info_old(
    *, document_id: str, chunk_id: int, large_chunk_reference_ids: list[int] = []
) -> UUID:
    doc_str = document_id

    # Web parsing URL duplicate catching
    if doc_str and doc_str[-1] == "/":
        doc_str = doc_str[:-1]
    unique_identifier_string = "_".join([doc_str, str(chunk_id), "0"])
    if large_chunk_reference_ids:
        unique_identifier_string += "_large" + "_".join(
            [
                str(referenced_chunk_id)
                for referenced_chunk_id in large_chunk_reference_ids
            ]
        )
    return uuid.uuid5(uuid.NAMESPACE_X500, unique_identifier_string)


def get_uuid_from_chunk(chunk: DocMetadataAwareIndexChunk) -> uuid.UUID:
    return get_uuid_from_chunk_info(
        document_id=chunk.embedded_chunk.chunk.document_id,
        chunk_id=chunk.embedded_chunk.chunk.chunk_id,
        tenant_id=chunk.tenant_id,
        large_chunk_id=chunk.embedded_chunk.chunk.large_chunk_id,
    )


def get_uuid_from_chunk_old(
    chunk: DocMetadataAwareIndexChunk, large_chunk_reference_ids: list[int] = []
) -> UUID:
    return get_uuid_from_chunk_info_old(
        document_id=chunk.embedded_chunk.chunk.document_id,
        chunk_id=chunk.embedded_chunk.chunk.chunk_id,
        large_chunk_reference_ids=large_chunk_reference_ids,
    )
