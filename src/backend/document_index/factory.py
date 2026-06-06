"""Factory for creating DocumentIndex instances.

Selects the appropriate backend (Vespa, OpenSearch, or Disabled) based on
environment variables. No DB session required.
"""

import os

from src.backend.document_index.disabled import DisabledDocumentIndex
from src.backend.document_index.interfaces_new import DocumentIndex

_TRUTHY = {"1", "true", "yes"}


def _is_vector_db_disabled() -> bool:
    return os.environ.get("DISABLE_VECTOR_DB", "").lower() in _TRUTHY


def _is_vespa_disabled() -> bool:
    return os.environ.get("ONYX_DISABLE_VESPA", "").lower() in _TRUTHY


def _is_opensearch_enabled() -> bool:
    return os.environ.get("ENABLE_OPENSEARCH_INDEXING_FOR_ONYX", "").lower() in _TRUTHY


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
    if _is_vector_db_disabled():
        return DisabledDocumentIndex()

    if _is_opensearch_enabled():
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
    if _is_vector_db_disabled():
        return [DisabledDocumentIndex()]

    if _is_vespa_disabled() and not _is_opensearch_enabled():
        raise ValueError(
            "ONYX_DISABLE_VESPA is set but ENABLE_OPENSEARCH_INDEXING_FOR_ONYX is not."
        )

    result: list[DocumentIndex] = []
    if not _is_vespa_disabled():
        from src.backend.document_index.vespa.vespa_document_index import (
            VespaDocumentIndex,
            VespaIndexPair,
        )

        tenant_state = _build_tenant_state()
        vespa_primary = VespaDocumentIndex(
            index_name=primary_index_name,
            tenant_state=tenant_state,
            large_chunks_enabled=large_chunks_enabled,
        )
        vespa_secondary = None
        if secondary_index_name:
            vespa_secondary = VespaDocumentIndex(
                index_name=secondary_index_name,
                tenant_state=tenant_state,
                large_chunks_enabled=secondary_large_chunks_enabled,
            )
        result.append(
            VespaIndexPair(
                primary=vespa_primary,
                secondary=vespa_secondary,
                secondary_index_name=secondary_index_name,
                secondary_embedding_dim=secondary_embedding_dim,
                secondary_embedding_precision=_get_embedding_precision()
                if vespa_secondary
                else None,
            )
        )
    if _is_opensearch_enabled():
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
