"""Unit tests for HybridDocumentIndex and _rrf_merge."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.internal.document_index.hybrid import HybridDocumentIndex, _rrf_merge
from src.internal.document_index.interfaces import (
    DocumentInsertionRecord,
    IndexingMetadata,
    MetadataUpdateRequest,
)
from src.internal.document_index.models import (
    DocMetadataAwareIndexChunk,
    DocumentAccess,
    EmbeddedChunk,
    EmbeddingPrecision,
    IndexChunk,
    IndexFilters,
    InferenceChunk,
    QueryType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(
    document_id: str = "doc-1", chunk_ind: int = 0, score: float = 0.5
) -> InferenceChunk:
    return InferenceChunk(
        document_id=document_id,
        chunk_ind=chunk_ind,
        content=f"content of {document_id}:{chunk_ind}",
        score=score,
    )


def _make_hybrid(
    os_index: MagicMock | None = None,
    wv_index: MagicMock | None = None,
) -> HybridDocumentIndex:
    return HybridDocumentIndex(
        opensearch_index=os_index or MagicMock(),
        weaviate_index=wv_index or MagicMock(),
    )


def _make_doc_chunk(document_id: str = "doc-1") -> DocMetadataAwareIndexChunk:
    ic = IndexChunk(
        id=f"{document_id}_0",
        document_id=document_id,
        chunk_id=0,
        text="hello",
        blurb="hello",
    )
    ec = EmbeddedChunk(chunk=ic, embedding=np.zeros(4, dtype=np.float32))
    return DocMetadataAwareIndexChunk(
        embedded_chunk=ec,
        tenant_id="default",
        access=DocumentAccess(is_public=True),
    )


def _indexing_meta() -> IndexingMetadata:
    return IndexingMetadata(doc_id_to_chunk_cnt_diff={})


# ---------------------------------------------------------------------------
# verify_and_create_index_if_necessary
# ---------------------------------------------------------------------------


def test_verify_calls_both_backends():
    os_idx = MagicMock()
    wv_idx = MagicMock()
    hybrid = _make_hybrid(os_idx, wv_idx)

    hybrid.verify_and_create_index_if_necessary(4, EmbeddingPrecision.FLOAT)

    os_idx.verify_and_create_index_if_necessary.assert_called_once_with(
        4, EmbeddingPrecision.FLOAT
    )
    wv_idx.verify_and_create_index_if_necessary.assert_called_once_with(
        4, EmbeddingPrecision.FLOAT
    )


# ---------------------------------------------------------------------------
# index()
# ---------------------------------------------------------------------------


def test_index_writes_to_both_backends():
    os_idx = MagicMock()
    wv_idx = MagicMock()
    os_idx.index.return_value = [
        DocumentInsertionRecord(document_id="doc-1", already_existed=False)
    ]
    wv_idx.index.return_value = []
    hybrid = _make_hybrid(os_idx, wv_idx)

    chunk = _make_doc_chunk("doc-1")
    records = hybrid.index([chunk], _indexing_meta())

    os_idx.index.assert_called_once()
    wv_idx.index.assert_called_once()
    assert len(records) == 1
    assert records[0].document_id == "doc-1"


def test_index_returns_opensearch_records():
    """OpenSearch insertion records are returned (Weaviate records discarded)."""
    os_idx = MagicMock()
    wv_idx = MagicMock()
    os_records = [
        DocumentInsertionRecord(document_id="doc-1", already_existed=True),
        DocumentInsertionRecord(document_id="doc-2", already_existed=False),
    ]
    os_idx.index.return_value = os_records
    wv_idx.index.return_value = []
    hybrid = _make_hybrid(os_idx, wv_idx)

    records = hybrid.index([_make_doc_chunk("doc-1")], _indexing_meta())
    assert records == os_records


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


def test_delete_removes_from_both_backends():
    os_idx = MagicMock()
    wv_idx = MagicMock()
    os_idx.delete.return_value = 3
    hybrid = _make_hybrid(os_idx, wv_idx)

    count = hybrid.delete("doc-1", chunk_count=3)

    os_idx.delete.assert_called_once_with("doc-1", 3)
    wv_idx.delete.assert_called_once_with("doc-1", 3)
    assert count == 3


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------


def test_update_applies_to_both_backends():
    os_idx = MagicMock()
    wv_idx = MagicMock()
    hybrid = _make_hybrid(os_idx, wv_idx)

    req = MetadataUpdateRequest(
        document_ids=["doc-1"],
        doc_id_to_chunk_cnt={"doc-1": 2},
        hidden=True,
    )
    hybrid.update([req])

    os_idx.update.assert_called_once_with([req])
    wv_idx.update.assert_called_once_with([req])


# ---------------------------------------------------------------------------
# Keyword retrieval → OpenSearch only
# ---------------------------------------------------------------------------


def test_keyword_retrieval_uses_opensearch_only():
    os_idx = MagicMock()
    wv_idx = MagicMock()
    expected = [_chunk("doc-1")]
    os_idx.keyword_retrieval.return_value = expected
    hybrid = _make_hybrid(os_idx, wv_idx)

    result = hybrid.keyword_retrieval("test", IndexFilters(), num_to_retrieve=5)

    os_idx.keyword_retrieval.assert_called_once()
    wv_idx.keyword_retrieval.assert_not_called()
    assert result == expected


def test_keyword_retrieval_passes_include_hidden():
    os_idx = MagicMock()
    os_idx.keyword_retrieval.return_value = []
    hybrid = _make_hybrid(os_idx)

    hybrid.keyword_retrieval(
        "q", IndexFilters(), num_to_retrieve=3, include_hidden=True
    )

    call_kwargs = os_idx.keyword_retrieval.call_args
    assert call_kwargs.kwargs.get("include_hidden") is True


# ---------------------------------------------------------------------------
# Semantic retrieval → Weaviate only
# ---------------------------------------------------------------------------


def test_semantic_retrieval_uses_weaviate_only():
    os_idx = MagicMock()
    wv_idx = MagicMock()
    expected = [_chunk("doc-2")]
    wv_idx.semantic_retrieval.return_value = expected
    hybrid = _make_hybrid(os_idx, wv_idx)

    result = hybrid.semantic_retrieval([0.0] * 4, IndexFilters(), num_to_retrieve=5)

    wv_idx.semantic_retrieval.assert_called_once()
    os_idx.semantic_retrieval.assert_not_called()
    assert result == expected


# ---------------------------------------------------------------------------
# Hybrid retrieval → RRF merge of both
# ---------------------------------------------------------------------------


def test_hybrid_retrieval_queries_both_backends():
    os_idx = MagicMock()
    wv_idx = MagicMock()
    os_idx.keyword_retrieval.return_value = [_chunk("doc-1")]
    wv_idx.semantic_retrieval.return_value = [_chunk("doc-2")]
    hybrid = _make_hybrid(os_idx, wv_idx)

    result = hybrid.hybrid_retrieval(
        "q",
        [0.0] * 4,
        final_keywords=None,
        query_type=QueryType.HYBRID,
        filters=IndexFilters(),
        num_to_retrieve=5,
    )

    os_idx.keyword_retrieval.assert_called_once()
    wv_idx.semantic_retrieval.assert_called_once()
    assert len(result) == 2


def test_hybrid_retrieval_returns_at_most_num_to_retrieve():
    os_idx = MagicMock()
    wv_idx = MagicMock()
    os_idx.keyword_retrieval.return_value = [_chunk(f"doc-{i}") for i in range(10)]
    wv_idx.semantic_retrieval.return_value = [_chunk(f"doc-{i}") for i in range(10)]
    hybrid = _make_hybrid(os_idx, wv_idx)

    result = hybrid.hybrid_retrieval(
        "q",
        [0.0] * 4,
        final_keywords=None,
        query_type=QueryType.HYBRID,
        filters=IndexFilters(),
        num_to_retrieve=3,
    )

    assert len(result) <= 3


def test_hybrid_retrieval_merges_rrf_scores():
    """Chunks that appear in both lists get a higher merged RRF score."""
    os_idx = MagicMock()
    wv_idx = MagicMock()
    # doc-shared appears in both; doc-os only in keyword; doc-wv only in semantic
    os_idx.keyword_retrieval.return_value = [
        _chunk("doc-shared", 0, score=0.9),
        _chunk("doc-os", 0, score=0.8),
    ]
    wv_idx.semantic_retrieval.return_value = [
        _chunk("doc-shared", 0, score=0.85),
        _chunk("doc-wv", 0, score=0.7),
    ]
    hybrid = _make_hybrid(os_idx, wv_idx)

    result = hybrid.hybrid_retrieval(
        "q",
        [0.0] * 4,
        final_keywords=None,
        query_type=QueryType.HYBRID,
        filters=IndexFilters(),
        num_to_retrieve=5,
    )

    doc_ids = [c.document_id for c in result]
    # doc-shared should rank highest (appears in both lists)
    assert doc_ids[0] == "doc-shared"


# ---------------------------------------------------------------------------
# id_based_retrieval → OpenSearch only
# ---------------------------------------------------------------------------


def test_id_based_retrieval_uses_opensearch_only():
    os_idx = MagicMock()
    wv_idx = MagicMock()
    expected = [_chunk("doc-1")]
    os_idx.id_based_retrieval.return_value = expected
    hybrid = _make_hybrid(os_idx, wv_idx)

    from src.internal.document_index.interfaces import DocumentSectionRequest

    result = hybrid.id_based_retrieval(
        [DocumentSectionRequest(document_id="doc-1")],
        IndexFilters(),
    )

    os_idx.id_based_retrieval.assert_called_once()
    wv_idx.id_based_retrieval.assert_not_called()
    assert result == expected


# ---------------------------------------------------------------------------
# random_retrieval → OpenSearch only
# ---------------------------------------------------------------------------


def test_random_retrieval_uses_opensearch_only():
    os_idx = MagicMock()
    wv_idx = MagicMock()
    expected = [_chunk("doc-1")]
    os_idx.random_retrieval.return_value = expected
    hybrid = _make_hybrid(os_idx, wv_idx)

    result = hybrid.random_retrieval(IndexFilters(), num_to_retrieve=5)

    os_idx.random_retrieval.assert_called_once()
    wv_idx.random_retrieval.assert_not_called()
    assert result == expected


# ---------------------------------------------------------------------------
# _rrf_merge unit tests
# ---------------------------------------------------------------------------


def test_rrf_merge_empty_lists():
    assert _rrf_merge([], [], top_k=5) == []


def test_rrf_merge_keyword_only():
    chunks = [_chunk(f"doc-{i}", 0, score=float(i)) for i in range(3)]
    result = _rrf_merge(chunks, [], top_k=5)
    assert len(result) == 3


def test_rrf_merge_semantic_only():
    chunks = [_chunk(f"doc-{i}", 0, score=float(i)) for i in range(3)]
    result = _rrf_merge([], chunks, top_k=5)
    assert len(result) == 3


def test_rrf_merge_chunk_in_both_lists_ranks_highest():
    keyword = [_chunk("doc-shared", 0), _chunk("doc-k", 0)]
    semantic = [_chunk("doc-shared", 0), _chunk("doc-s", 0)]
    result = _rrf_merge(keyword, semantic, top_k=5)
    assert result[0].document_id == "doc-shared"


def test_rrf_merge_opensearch_result_preferred_for_shared_chunk():
    """When a chunk appears in both lists, the OpenSearch (keyword) version is kept."""
    os_chunk = InferenceChunk(
        document_id="doc-1",
        chunk_ind=0,
        content="rich OS content",
        score=0.9,
    )
    wv_chunk = _chunk("doc-1", 0, score=0.3)
    result = _rrf_merge([os_chunk], [wv_chunk], top_k=1)
    assert result[0].content == "rich OS content"


def test_rrf_merge_stamps_rrf_score_not_original():
    keyword = [_chunk("doc-1", 0, score=0.99)]
    result = _rrf_merge(keyword, [], top_k=1, k=60)
    # RRF score for rank 1 with k=60: 1/(60+1) ≈ 0.01639
    assert result[0].score == pytest.approx(1.0 / 61, rel=1e-4)
    assert result[0].score != 0.99


def test_rrf_merge_top_k_truncation():
    keyword = [_chunk(f"doc-{i}", 0) for i in range(10)]
    semantic = [_chunk(f"doc-{i}", 0) for i in range(10)]
    result = _rrf_merge(keyword, semantic, top_k=4)
    assert len(result) == 4


def test_rrf_merge_deduplicates_same_chunk():
    shared = _chunk("doc-1", 0)
    result = _rrf_merge([shared], [shared], top_k=5)
    # Same (doc_id, chunk_ind) should appear only once
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


def test_factory_hybrid_mode_builds_hybrid_index(monkeypatch):
    monkeypatch.setenv("ENABLE_HYBRID_INDEXING", "true")
    monkeypatch.delenv("DISABLE_VECTOR_DB", raising=False)
    monkeypatch.delenv("ENABLE_OPENSEARCH_INDEXING", raising=False)

    with (
        patch("src.internal.document_index.factory._build_opensearch_index") as mock_os,
        patch("src.internal.document_index.factory._build_weaviate_index") as mock_wv,
    ):
        mock_os.return_value = MagicMock()
        mock_wv.return_value = MagicMock()

        from src.internal.document_index import factory

        result = factory.get_default_document_index("primary", None)

    mock_os.assert_called_once_with("primary", 768)
    mock_wv.assert_called_once_with("primary", 768)
    assert isinstance(result, HybridDocumentIndex)


def test_factory_hybrid_takes_priority_over_opensearch(monkeypatch):
    monkeypatch.setenv("ENABLE_HYBRID_INDEXING", "true")
    monkeypatch.setenv("ENABLE_OPENSEARCH_INDEXING", "true")
    monkeypatch.delenv("DISABLE_VECTOR_DB", raising=False)

    with (
        patch("src.internal.document_index.factory._build_opensearch_index") as mock_os,
        patch("src.internal.document_index.factory._build_weaviate_index") as mock_wv,
    ):
        mock_os.return_value = MagicMock()
        mock_wv.return_value = MagicMock()

        from src.internal.document_index import factory

        result = factory.get_default_document_index("primary", None)

    # Result is HybridDocumentIndex, not a bare OpenSearch index
    assert isinstance(result, HybridDocumentIndex)
