"""Factory for creating DocumentIndex instances.

Selects the appropriate backend based on environment variables:
  DISABLE_VECTOR_DB=true         → DisabledDocumentIndex (no-op)
  ENABLE_OPENSEARCH_INDEXING=true → OpenSearchDocumentIndex
  default                        → WeaviateDocumentIndex

Weaviate connection is configured via WEAVIATE_HOST (default: localhost)
and WEAVIATE_PORT (default: 8080).
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


def _build_weaviate_index(
    index_name: str,
    embedding_dim: int,
) -> DocumentIndex:
    from src.internal.document_index.weaviate.weaviate_document_index import (
        WeaviateDocumentIndex,
    )

    tenant_state = _build_tenant_state()
    return WeaviateDocumentIndex(
        tenant_state=tenant_state,
        index_name=index_name,
        embedding_dim=embedding_dim,
        embedding_precision=_get_embedding_precision(),
        weaviate_host=os.environ.get("WEAVIATE_HOST", "localhost"),
        weaviate_port=int(os.environ.get("WEAVIATE_PORT", "8080")),
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
    Returns WeaviateDocumentIndex by default.
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

    return _build_weaviate_index(primary_index_name, embedding_dim)


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
        secondary = None
        if secondary_index_name:
            secondary = OpenSearchDocumentIndex(
                tenant_state=tenant_state,
                index_name=secondary_index_name,
                embedding_dim=secondary_embedding_dim or embedding_dim,
                embedding_precision=_get_embedding_precision(),
            )
        return [OpenSearchIndexPair(primary=primary, secondary=secondary)]

    return [_build_weaviate_index(primary_index_name, embedding_dim)]
