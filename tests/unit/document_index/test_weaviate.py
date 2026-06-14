"""Unit tests for WeaviateDocumentIndex.

All tests use a mock Weaviate client — no running Weaviate needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import json
import numpy as np
import pytest

from src.internal.document_index.interfaces import (
    IndexingMetadata,
    MetadataUpdateRequest,
    TenantState,
)
from src.internal.document_index.models import (
    DocumentAccess,
    DocMetadataAwareIndexChunk,
    EmbeddedChunk,
    EmbeddingPrecision,
    IndexChunk,
    IndexFilters,
    QueryType,
)
from src.internal.document_index.weaviate.weaviate_document_index import (
    WeaviateDocumentIndex,
    _objects_to_inference_chunks,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tenant(multitenant: bool = False) -> TenantState:
    return TenantState(
        tenant_id="test-tenant" if multitenant else "default",
        multitenant=multitenant,
    )


def _mock_client(collection: MagicMock | None = None) -> MagicMock:
    client = MagicMock()
    client.collections.exists.return_value = False
    client.collections.create.return_value = None
    if collection is None:
        collection = MagicMock()
    client.collections.get.return_value = collection
    return client


def _make_chunk(
    document_id: str = "doc-1",
    chunk_id: int = 0,
    text: str = "hello world",
    title: str = "Test Doc",
    embedding_dim: int = 4,
    is_public: bool = True,
) -> DocMetadataAwareIndexChunk:
    index_chunk = IndexChunk(
        id=f"{document_id}_{chunk_id}",
        document_id=document_id,
        chunk_id=chunk_id,
        text=text,
        title=title,
        blurb=text[:48],
    )
    embedded = EmbeddedChunk(
        chunk=index_chunk,
        embedding=np.zeros(embedding_dim, dtype=np.float32),
    )
    return DocMetadataAwareIndexChunk(
        embedded_chunk=embedded,
        tenant_id="default",
        access=DocumentAccess(is_public=is_public),
    )


def _make_indexing_meta(
    doc_id: str,
    old_cnt: int,
    new_cnt: int,
) -> IndexingMetadata:
    return IndexingMetadata(
        doc_id_to_chunk_cnt_diff={
            doc_id: IndexingMetadata.ChunkCounts(
                old_chunk_cnt=old_cnt,
                new_chunk_cnt=new_cnt,
            )
        }
    )


def _weaviate_obj(
    document_id: str = "doc-1",
    chunk_ind: int = 0,
    content: str = "text",
    score: float = 0.8,
    use_distance: bool = False,
) -> SimpleNamespace:
    meta = SimpleNamespace(
        score=score if not use_distance else None,
        distance=1.0 - score if use_distance else None,
    )
    props = {
        "document_id": document_id,
        "chunk_ind": chunk_ind,
        "content": content,
        "blurb": content[:48],
        "title": "T",
        "semantic_identifier": document_id,
        "source_type": "",
        "source_links": "{}",
        "metadata": "{}",
        "boost": 0,
        "hidden": False,
        "is_public": True,
        "document_sets": [],
        "access_control_list": [],
        "large_chunk_id": None,
    }
    return SimpleNamespace(properties=props, metadata=meta, uuid="uuid-1")


def _make_index(
    client: MagicMock | None = None,
    multitenant: bool = False,
) -> WeaviateDocumentIndex:
    if client is None:
        client = _mock_client()
    return WeaviateDocumentIndex(
        tenant_state=_tenant(multitenant),
        index_name="test_index",
        embedding_dim=4,
        client=client,
    )


# ---------------------------------------------------------------------------
# verify_and_create_index_if_necessary
# ---------------------------------------------------------------------------


def test_creates_collection_when_absent():
    client = _mock_client()
    client.collections.exists.return_value = False
    idx = _make_index(client)
    idx.verify_and_create_index_if_necessary(4, EmbeddingPrecision.FLOAT)
    client.collections.create.assert_called_once()
    call_kwargs = client.collections.create.call_args[1]
    assert call_kwargs["name"] == "test_index"


def test_skips_creation_when_collection_exists():
    client = _mock_client()
    client.collections.exists.return_value = True
    idx = _make_index(client)
    idx.verify_and_create_index_if_necessary(4, EmbeddingPrecision.FLOAT)
    client.collections.create.assert_not_called()


def test_multitenant_schema_includes_multi_tenancy_config():
    client = _mock_client()
    client.collections.exists.return_value = False
    idx = _make_index(client, multitenant=True)
    idx.verify_and_create_index_if_necessary(4, EmbeddingPrecision.FLOAT)
    call_kwargs = client.collections.create.call_args[1]
    assert "multi_tenancy_config" in call_kwargs


# ---------------------------------------------------------------------------
# index()
# ---------------------------------------------------------------------------


def test_index_inserts_chunks_and_returns_records():
    collection = MagicMock()
    # No existing chunks
    collection.query.fetch_objects.return_value = SimpleNamespace(objects=[])
    # batch context manager
    batch_ctx = MagicMock()
    collection.batch.dynamic.return_value.__enter__ = MagicMock(return_value=batch_ctx)
    collection.batch.dynamic.return_value.__exit__ = MagicMock(return_value=False)

    client = _mock_client(collection)
    idx = _make_index(client)

    chunk = _make_chunk("doc-1", 0)
    meta = _make_indexing_meta("doc-1", 0, 1)

    records = idx.index([chunk], meta)

    assert len(records) == 1
    assert records[0].document_id == "doc-1"
    assert records[0].already_existed is False
    batch_ctx.add_object.assert_called_once()


def test_index_marks_existing_document():
    collection = MagicMock()
    # First call (existence check) returns an existing object
    existing_obj = _weaviate_obj("doc-1")
    collection.query.fetch_objects.return_value = SimpleNamespace(
        objects=[existing_obj]
    )
    batch_ctx = MagicMock()
    collection.batch.dynamic.return_value.__enter__ = MagicMock(return_value=batch_ctx)
    collection.batch.dynamic.return_value.__exit__ = MagicMock(return_value=False)

    client = _mock_client(collection)
    idx = _make_index(client)

    chunk = _make_chunk("doc-1", 0)
    meta = _make_indexing_meta("doc-1", 1, 1)

    records = idx.index([chunk], meta)

    assert records[0].already_existed is True
    # old doc chunks should be deleted before re-insert
    collection.data.delete_many.assert_called()


def test_index_deletes_tail_chunks_when_document_shrinks():
    collection = MagicMock()
    collection.query.fetch_objects.return_value = SimpleNamespace(objects=[])
    batch_ctx = MagicMock()
    collection.batch.dynamic.return_value.__enter__ = MagicMock(return_value=batch_ctx)
    collection.batch.dynamic.return_value.__exit__ = MagicMock(return_value=False)

    client = _mock_client(collection)
    idx = _make_index(client)

    chunk = _make_chunk("doc-1", 0)
    # old had 5 chunks, new has 1 → delete tail chunks [1..4]
    meta = _make_indexing_meta("doc-1", 5, 1)

    idx.index([chunk], meta)

    # delete_many called for tail chunks
    collection.data.delete_many.assert_called()


def test_index_empty_input_returns_empty():
    client = _mock_client()
    idx = _make_index(client)
    meta = IndexingMetadata(doc_id_to_chunk_cnt_diff={})
    records = idx.index([], meta)
    assert records == []


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


def test_delete_removes_chunks_and_returns_count():
    collection = MagicMock()
    existing = [_weaviate_obj("doc-1", i) for i in range(3)]
    collection.query.fetch_objects.return_value = SimpleNamespace(objects=existing)

    client = _mock_client(collection)
    idx = _make_index(client)

    count = idx.delete("doc-1")

    assert count == 3
    collection.data.delete_many.assert_called_once()


def test_delete_returns_zero_when_not_found():
    collection = MagicMock()
    collection.query.fetch_objects.return_value = SimpleNamespace(objects=[])

    client = _mock_client(collection)
    idx = _make_index(client)

    count = idx.delete("missing-doc")
    assert count == 0
    collection.data.delete_many.assert_not_called()


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------


def test_update_patches_hidden_field():
    collection = MagicMock()
    obj = _weaviate_obj("doc-1")
    collection.query.fetch_objects.return_value = SimpleNamespace(objects=[obj])

    client = _mock_client(collection)
    idx = _make_index(client)

    req = MetadataUpdateRequest(
        document_ids=["doc-1"],
        doc_id_to_chunk_cnt={"doc-1": 1},
        hidden=True,
    )
    idx.update([req])

    collection.data.update.assert_called_once()
    call_kwargs = collection.data.update.call_args[1]
    assert call_kwargs["properties"]["hidden"] is True


def test_update_skips_if_no_fields_change():
    collection = MagicMock()
    obj = _weaviate_obj("doc-1")
    collection.query.fetch_objects.return_value = SimpleNamespace(objects=[obj])

    client = _mock_client(collection)
    idx = _make_index(client)

    # No fields set → patch is empty → update should not be called
    req = MetadataUpdateRequest(
        document_ids=["doc-1"],
        doc_id_to_chunk_cnt={"doc-1": 1},
    )
    idx.update([req])

    collection.data.update.assert_not_called()


# ---------------------------------------------------------------------------
# Retrieval methods
# ---------------------------------------------------------------------------


def test_keyword_retrieval_calls_bm25():
    collection = MagicMock()
    result_obj = _weaviate_obj("doc-1", 0, "test content", score=0.9)
    collection.query.bm25.return_value = SimpleNamespace(objects=[result_obj])

    client = _mock_client(collection)
    idx = _make_index(client)

    chunks = idx.keyword_retrieval("test", IndexFilters(), num_to_retrieve=5)

    collection.query.bm25.assert_called_once()
    assert len(chunks) == 1
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].score == pytest.approx(0.9)


def test_semantic_retrieval_calls_near_vector():
    collection = MagicMock()
    result_obj = _weaviate_obj("doc-1", 0, "text", use_distance=True)
    result_obj.metadata.distance = 0.2
    collection.query.near_vector.return_value = SimpleNamespace(objects=[result_obj])

    client = _mock_client(collection)
    idx = _make_index(client)

    chunks = idx.semantic_retrieval(
        [0.0, 0.0, 0.0, 0.0], IndexFilters(), num_to_retrieve=5
    )

    collection.query.near_vector.assert_called_once()
    assert len(chunks) == 1
    # 1 - distance → 1 - 0.2 = 0.8
    assert chunks[0].score == pytest.approx(0.8)


def test_hybrid_retrieval_calls_hybrid():
    collection = MagicMock()
    result_obj = _weaviate_obj("doc-1", 0, "text", score=0.7)
    collection.query.hybrid.return_value = SimpleNamespace(objects=[result_obj])

    client = _mock_client(collection)
    idx = _make_index(client)

    chunks = idx.hybrid_retrieval(
        "query",
        [0.0, 0.0, 0.0, 0.0],
        final_keywords=None,
        query_type=QueryType.HYBRID,
        filters=IndexFilters(),
        num_to_retrieve=5,
    )

    collection.query.hybrid.assert_called_once()
    assert len(chunks) == 1


# ---------------------------------------------------------------------------
# id_based_retrieval()
# ---------------------------------------------------------------------------


def test_id_based_retrieval_fetches_by_doc_id():
    collection = MagicMock()
    objs = [_weaviate_obj("doc-1", i) for i in range(3)]
    collection.query.fetch_objects.return_value = SimpleNamespace(objects=objs)

    client = _mock_client(collection)
    idx = _make_index(client)

    from src.internal.document_index.interfaces import DocumentSectionRequest

    chunks = idx.id_based_retrieval(
        [DocumentSectionRequest(document_id="doc-1")],
        IndexFilters(),
    )

    assert len(chunks) == 3
    # Should be sorted by chunk_ind
    assert [c.chunk_ind for c in chunks] == [0, 1, 2]


def test_id_based_retrieval_respects_chunk_range():
    collection = MagicMock()
    objs = [_weaviate_obj("doc-1", i) for i in range(2)]
    collection.query.fetch_objects.return_value = SimpleNamespace(objects=objs)

    client = _mock_client(collection)
    idx = _make_index(client)

    from src.internal.document_index.interfaces import DocumentSectionRequest

    idx.id_based_retrieval(
        [DocumentSectionRequest(document_id="doc-1", min_chunk_ind=1, max_chunk_ind=3)],
        IndexFilters(),
    )

    call_kwargs = collection.query.fetch_objects.call_args[1]
    # filter should include min and max conditions
    assert call_kwargs.get("filters") is not None


# ---------------------------------------------------------------------------
# random_retrieval()
# ---------------------------------------------------------------------------


def test_random_retrieval_returns_chunks():
    collection = MagicMock()
    objs = [_weaviate_obj("doc-1", i) for i in range(5)]
    collection.query.fetch_objects.return_value = SimpleNamespace(objects=objs)

    client = _mock_client(collection)
    idx = _make_index(client)

    chunks = idx.random_retrieval(IndexFilters(), num_to_retrieve=5)
    assert len(chunks) == 5


# ---------------------------------------------------------------------------
# _objects_to_inference_chunks helper
# ---------------------------------------------------------------------------


def test_objects_to_inference_chunks_parses_source_links():
    links = {0: "https://example.com"}
    obj = _weaviate_obj("doc-1")
    obj.properties["source_links"] = json.dumps(links)
    chunks = _objects_to_inference_chunks([obj])
    # JSON always serialises int keys as strings; Pydantic coerces them back to int
    assert chunks[0].source_links == {0: "https://example.com"}


def test_objects_to_inference_chunks_handles_bad_json_gracefully():
    obj = _weaviate_obj("doc-1")
    obj.properties["source_links"] = "not-json"
    obj.properties["metadata"] = "also-bad"
    # Should not raise
    chunks = _objects_to_inference_chunks([obj])
    assert chunks[0].source_links is None
    assert chunks[0].metadata == {}


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


def test_factory_returns_weaviate_by_default(monkeypatch):
    """Without ENABLE_OPENSEARCH_INDEXING or DISABLE_VECTOR_DB, Weaviate is used."""
    monkeypatch.delenv("ENABLE_OPENSEARCH_INDEXING", raising=False)
    monkeypatch.delenv("DISABLE_VECTOR_DB", raising=False)

    from src.internal.document_index.weaviate.weaviate_document_index import (
        WeaviateDocumentIndex,
    )

    with patch(
        "src.internal.document_index.factory._build_weaviate_index"
    ) as mock_build:
        mock_index = MagicMock(spec=WeaviateDocumentIndex)
        mock_build.return_value = mock_index

        from src.internal.document_index import factory

        result = factory.get_default_document_index("primary", None)

    mock_build.assert_called_once_with("primary", 768)
    assert result is mock_index


def test_factory_returns_disabled_when_disable_env_set(monkeypatch):
    monkeypatch.setenv("DISABLE_VECTOR_DB", "true")
    monkeypatch.delenv("ENABLE_OPENSEARCH_INDEXING", raising=False)

    from src.internal.document_index.disabled import DisabledDocumentIndex
    from src.internal.document_index import factory

    result = factory.get_default_document_index("primary", None)
    assert isinstance(result, DisabledDocumentIndex)


def test_factory_all_indices_returns_weaviate_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_OPENSEARCH_INDEXING", raising=False)
    monkeypatch.delenv("DISABLE_VECTOR_DB", raising=False)

    from src.internal.document_index.weaviate.weaviate_document_index import (
        WeaviateDocumentIndex,
    )

    with patch(
        "src.internal.document_index.factory._build_weaviate_index"
    ) as mock_build:
        mock_index = MagicMock(spec=WeaviateDocumentIndex)
        mock_build.return_value = mock_index

        from src.internal.document_index import factory

        results = factory.get_all_document_indices("primary", None)

    assert results == [mock_index]
