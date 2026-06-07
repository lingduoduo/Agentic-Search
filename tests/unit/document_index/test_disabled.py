import pytest
from src.backend.document_index.disabled import DisabledDocumentIndex
from src.retrieval.models import (
    EmbeddingPrecision,
    IndexFilters,
    QueryType,
)
from src.backend.document_index.interfaces import (
    IndexingMetadata,
)


def test_disabled_index_importable():
    """DisabledDocumentIndex imports without error."""
    from src.backend.document_index.disabled import DisabledDocumentIndex  # noqa: F401


def test_verify_and_create_is_noop():
    """verify_and_create_index_if_necessary is a no-op that does not raise."""
    idx = DisabledDocumentIndex()
    # Should complete without raising
    idx.verify_and_create_index_if_necessary(
        embedding_dim=768, embedding_precision=EmbeddingPrecision.FLOAT
    )


def test_index_raises():
    """index() raises RuntimeError."""
    idx = DisabledDocumentIndex()
    with pytest.raises(RuntimeError, match="Vector DB is disabled"):
        idx.index(
            chunks=[],
            indexing_metadata=IndexingMetadata(doc_id_to_chunk_cnt_diff={}),
        )


def test_delete_raises():
    """delete() raises RuntimeError."""
    idx = DisabledDocumentIndex()
    with pytest.raises(RuntimeError, match="Vector DB is disabled"):
        idx.delete(document_id="doc1")


def test_update_raises():
    """update() raises RuntimeError."""
    idx = DisabledDocumentIndex()
    with pytest.raises(RuntimeError, match="Vector DB is disabled"):
        idx.update(update_requests=[])


def test_id_based_retrieval_raises():
    """id_based_retrieval() raises RuntimeError."""
    idx = DisabledDocumentIndex()
    with pytest.raises(RuntimeError, match="Vector DB is disabled"):
        idx.id_based_retrieval(
            chunk_requests=[],
            filters=IndexFilters(),
        )


def test_hybrid_retrieval_raises():
    """hybrid_retrieval() raises RuntimeError."""
    idx = DisabledDocumentIndex()
    with pytest.raises(RuntimeError, match="Vector DB is disabled"):
        idx.hybrid_retrieval(
            query="test",
            query_embedding=[0.1] * 4,
            final_keywords=None,
            query_type=QueryType.HYBRID,
            filters=IndexFilters(),
            num_to_retrieve=10,
        )


def test_keyword_retrieval_raises():
    """keyword_retrieval() raises RuntimeError."""
    idx = DisabledDocumentIndex()
    with pytest.raises(RuntimeError, match="Vector DB is disabled"):
        idx.keyword_retrieval(
            query="test",
            filters=IndexFilters(),
            num_to_retrieve=10,
        )


def test_semantic_retrieval_raises():
    """semantic_retrieval() raises RuntimeError."""
    idx = DisabledDocumentIndex()
    with pytest.raises(RuntimeError, match="Vector DB is disabled"):
        idx.semantic_retrieval(
            query_embedding=[0.1] * 4,
            filters=IndexFilters(),
            num_to_retrieve=10,
        )


def test_random_retrieval_raises():
    """random_retrieval() raises RuntimeError."""
    idx = DisabledDocumentIndex()
    with pytest.raises(RuntimeError, match="Vector DB is disabled"):
        idx.random_retrieval(
            filters=IndexFilters(),
            num_to_retrieve=10,
        )
