"""Factory for creating DocumentIndex instances.

Selects the appropriate backend (OpenSearch or Disabled) based on
environment variables. No DB session required.
"""

import os

from src.internal.document_index.disabled import DisabledDocumentIndex
from src.internal.document_index.interfaces import DocumentIndex

_TRUTHY = {"1", "true", "yes"}


def _is_vector_db_disabled() -> bool:
    return os.environ.get("DISABLE_VECTOR_DB", "").lower() in _TRUTHY


def _is_opensearch_enabled() -> bool:
    return os.environ.get("ENABLE_OPENSEARCH_INDEXING", "").lower() in _TRUTHY


def _build_tenant_state():
    from src.internal.document_index.interfaces import TenantState

    tenant_id = os.environ.get("CURRENT_TENANT_ID", "default")
    multi_tenant = os.environ.get("MULTI_TENANT", "").lower() in {"1", "true", "yes"}
    return TenantState(tenant_id=tenant_id, multitenant=multi_tenant)


def _get_embedding_precision():
    from src.internal.document_index.models import EmbeddingPrecision

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
    Returns OpenSearchDocumentIndex when ENABLE_OPENSEARCH_INDEXING=true.
    Returns DisabledDocumentIndex as fallback.
    """
    if _is_vector_db_disabled():
        return DisabledDocumentIndex()

    if _is_opensearch_enabled():
        from src.internal.document_index.opensearch.opensearch_document_index import (
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

    return DisabledDocumentIndex()


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

    if not _is_opensearch_enabled():
        return [DisabledDocumentIndex()]

    from src.internal.document_index.opensearch.opensearch_document_index import (
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
    return [OpenSearchIndexPair(primary=primary, secondary=secondary)]
