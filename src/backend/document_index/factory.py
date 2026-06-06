"""Factory for creating DocumentIndex instances.

Selects the appropriate backend (Vespa, OpenSearch, or Disabled) based on
environment variables. No DB session required.
"""

import os

from src.backend.document_index.disabled import DisabledDocumentIndex
from src.backend.document_index.interfaces_new import DocumentIndex

_DISABLE_VECTOR_DB: bool = os.environ.get("DISABLE_VECTOR_DB", "").lower() in {
    "1",
    "true",
    "yes",
}
_ONYX_DISABLE_VESPA: bool = os.environ.get("ONYX_DISABLE_VESPA", "").lower() in {
    "1",
    "true",
    "yes",
}
_ENABLE_OPENSEARCH: bool = os.environ.get(
    "ENABLE_OPENSEARCH_INDEXING_FOR_ONYX", ""
).lower() in {"1", "true", "yes"}


def _build_tenant_state():
    from src.backend.document_index.interfaces_new import TenantState

    tenant_id = os.environ.get("CURRENT_TENANT_ID", "default")
    multi_tenant = os.environ.get("MULTI_TENANT", "").lower() in {"1", "true", "yes"}
    return TenantState(tenant_id=tenant_id, multitenant=multi_tenant)


def _get_embedding_precision():
    from src.retrieval.models import EmbeddingPrecision

    raw = os.environ.get("EMBEDDING_PRECISION", "float")
    return (
        EmbeddingPrecision(raw)
        if raw in ("float", "bfloat16")
        else EmbeddingPrecision.FLOAT
    )


def get_default_document_index(
    primary_index_name: str,
    secondary_index_name: str | None,
    large_chunks_enabled: bool = False,
    secondary_large_chunks_enabled: bool = False,
    embedding_dim: int = 768,
    secondary_embedding_dim: int | None = None,
) -> DocumentIndex:
    """Get the primary document index for retrieval.

    Returns DisabledDocumentIndex when DISABLE_VECTOR_DB=true.
    Returns OpenSearchDocumentIndex when ENABLE_OPENSEARCH_INDEXING_FOR_ONYX=true.
    Otherwise returns VespaDocumentIndex.
    """
    if _DISABLE_VECTOR_DB:
        return DisabledDocumentIndex()

    if _ENABLE_OPENSEARCH:
        from src.backend.document_index.opensearch.opensearch_document_index import (
            OpenSearchDocumentIndex,
            OpenSearchIndexPair,
        )

        tenant_state = _build_tenant_state()
        primary = OpenSearchDocumentIndex(
            tenant_state=tenant_state,
            index_name=primary_index_name,
            embedding_dim=embedding_dim,
            embedding_precision=_get_embedding_precision(),
        )
        if secondary_index_name is None:
            return OpenSearchIndexPair(primary=primary, secondary=None)
        secondary = OpenSearchDocumentIndex(
            tenant_state=tenant_state,
            index_name=secondary_index_name,
            embedding_dim=secondary_embedding_dim or embedding_dim,
            embedding_precision=_get_embedding_precision(),
        )
        return OpenSearchIndexPair(
            primary=primary,
            secondary=secondary,
            secondary_embedding_dim=secondary_embedding_dim or embedding_dim,
            secondary_embedding_precision=_get_embedding_precision(),
        )

    from src.backend.document_index.vespa.vespa_document_index import (
        VespaDocumentIndex,
        VespaIndexPair,
    )

    tenant_state = _build_tenant_state()
    primary = VespaDocumentIndex(
        index_name=primary_index_name,
        tenant_state=tenant_state,
        large_chunks_enabled=large_chunks_enabled,
    )
    if secondary_index_name is None:
        return VespaIndexPair(
            primary=primary,
            secondary=None,
            secondary_index_name=None,
            secondary_embedding_dim=None,
            secondary_embedding_precision=None,
        )
    secondary = VespaDocumentIndex(
        index_name=secondary_index_name,
        tenant_state=tenant_state,
        large_chunks_enabled=secondary_large_chunks_enabled,
    )
    return VespaIndexPair(
        primary=primary,
        secondary=secondary,
        secondary_index_name=secondary_index_name,
        secondary_embedding_dim=secondary_embedding_dim,
        secondary_embedding_precision=_get_embedding_precision(),
    )


def get_all_document_indices(
    primary_index_name: str,
    secondary_index_name: str | None,
    large_chunks_enabled: bool = False,
    secondary_large_chunks_enabled: bool = False,
    embedding_dim: int = 768,
    secondary_embedding_dim: int | None = None,
) -> list[DocumentIndex]:
    """Get every document index that should be written to during indexing."""
    if _DISABLE_VECTOR_DB:
        return [DisabledDocumentIndex()]

    if _ONYX_DISABLE_VESPA and not _ENABLE_OPENSEARCH:
        raise ValueError(
            "ONYX_DISABLE_VESPA is set but ENABLE_OPENSEARCH_INDEXING_FOR_ONYX is not."
        )

    result: list[DocumentIndex] = []
    if not _ONYX_DISABLE_VESPA:
        result.append(
            get_default_document_index(
                primary_index_name=primary_index_name,
                secondary_index_name=secondary_index_name,
                large_chunks_enabled=large_chunks_enabled,
                secondary_large_chunks_enabled=secondary_large_chunks_enabled,
                embedding_dim=embedding_dim,
                secondary_embedding_dim=secondary_embedding_dim,
            )
        )
    if _ENABLE_OPENSEARCH:
        from src.backend.document_index.opensearch.opensearch_document_index import (
            OpenSearchDocumentIndex,
            OpenSearchIndexPair,
        )

        tenant_state = _build_tenant_state()
        primary = OpenSearchDocumentIndex(
            tenant_state=tenant_state,
            index_name=primary_index_name,
            embedding_dim=embedding_dim,
            embedding_precision=_get_embedding_precision(),
        )
        secondary = None
        if secondary_index_name:
            secondary = OpenSearchDocumentIndex(
                tenant_state=tenant_state,
                index_name=secondary_index_name,
                embedding_dim=secondary_embedding_dim or embedding_dim,
                embedding_precision=_get_embedding_precision(),
            )
        result.append(OpenSearchIndexPair(primary=primary, secondary=secondary))
    return result
