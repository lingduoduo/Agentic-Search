"""Unit tests for retrieval indexing models."""

from __future__ import annotations

import numpy as np

from src.retrieval.models import ChunkEmbedding
from src.retrieval.models import DocMetadataAwareIndexChunk
from src.retrieval.models import DocumentAccess
from src.retrieval.models import EmbeddedChunk
from src.retrieval.models import EmbeddingModelDetail
from src.retrieval.models import IndexChunk
from src.retrieval.models import IndexingSetting


def test_embedded_chunk_exposes_chunk_embedding_view():
    chunk = IndexChunk(
        id="doc::chunk-0",
        document_id="doc",
        chunk_id=0,
        text="alpha beta",
    )
    embedding = np.array([1.0, 0.0], dtype=np.float32)
    mini_embedding = np.array([0.0, 1.0], dtype=np.float32)

    embedded = EmbeddedChunk(
        chunk=chunk,
        embedding=embedding,
        mini_chunk_embeddings=[mini_embedding],
    )

    assert isinstance(embedded.embeddings, ChunkEmbedding)
    np.testing.assert_array_equal(embedded.embeddings.full_embedding, embedding)
    np.testing.assert_array_equal(
        embedded.embeddings.mini_chunk_embeddings[0],
        mini_embedding,
    )


def test_indexing_setting_final_embedding_dim_prefers_reduced_dimension():
    setting = IndexingSetting(
        model_name="local-model",
        model_dim=768,
        normalize=True,
        reduced_dimension=256,
    )

    assert setting.final_embedding_dim == 256


def test_embedding_model_detail_from_mapping():
    detail = EmbeddingModelDetail.from_mapping(
        {
            "id": 7,
            "model_name": "bge-small",
            "normalize": True,
            "query_prefix": "query:",
            "passage_prefix": "passage:",
            "api_url": "https://example.test",
        }
    )

    assert detail.id == 7
    assert detail.model_name == "bge-small"
    assert detail.normalize is True
    assert detail.query_prefix == "query:"


def test_doc_metadata_aware_index_chunk_from_embedded_chunk():
    embedded = EmbeddedChunk(
        chunk=IndexChunk(
            id="doc::chunk-0",
            document_id="doc",
            chunk_id=0,
            text="alpha beta",
        ),
        embedding=np.ones(2, dtype=np.float32),
    )
    access = DocumentAccess(is_public=False, user_ids={"user-1"})

    enriched = DocMetadataAwareIndexChunk.from_embedded_chunk(
        embedded,
        access=access,
        tenant_id="tenant",
        document_sets={"set-a"},
        boost=3,
        ancestor_hierarchy_node_ids=[1, 2],
    )

    assert enriched.embedded_chunk is embedded
    assert enriched.access.user_ids == {"user-1"}
    assert enriched.document_sets == {"set-a"}
    assert enriched.boost == 3
    assert enriched.ancestor_hierarchy_node_ids == [1, 2]
