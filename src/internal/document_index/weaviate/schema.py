"""Weaviate collection schema constants and builder for the document index."""

from __future__ import annotations

# Collection property names
PROP_DOCUMENT_ID = "document_id"
PROP_CHUNK_IND = "chunk_ind"
PROP_CONTENT = "content"
PROP_BLURB = "blurb"
PROP_TITLE = "title"
PROP_SEMANTIC_IDENTIFIER = "semantic_identifier"
PROP_SOURCE_TYPE = "source_type"
PROP_SOURCE_LINKS = "source_links"
PROP_METADATA = "metadata"
PROP_BOOST = "boost"
PROP_HIDDEN = "hidden"
PROP_IS_PUBLIC = "is_public"
PROP_DOCUMENT_SETS = "document_sets"
PROP_ACCESS_CONTROL_LIST = "access_control_list"
PROP_LARGE_CHUNK_ID = "large_chunk_id"

# All properties returned on retrieval
RETRIEVAL_PROPS = [
    PROP_DOCUMENT_ID,
    PROP_CHUNK_IND,
    PROP_CONTENT,
    PROP_BLURB,
    PROP_TITLE,
    PROP_SEMANTIC_IDENTIFIER,
    PROP_SOURCE_TYPE,
    PROP_SOURCE_LINKS,
    PROP_METADATA,
    PROP_BOOST,
    PROP_HIDDEN,
    PROP_IS_PUBLIC,
    PROP_DOCUMENT_SETS,
    PROP_ACCESS_CONTROL_LIST,
    PROP_LARGE_CHUNK_ID,
]


def build_collection_config(
    collection_name: str,
    multitenant: bool,
) -> dict:
    """Return keyword arguments for client.collections.create().

    The caller is responsible for unpacking these into the create call.
    Raises ImportError if weaviate-client is not installed.
    """
    from weaviate.classes.config import (
        Configure,
        DataType,
        Property,
        Tokenization,
        VectorDistances,
    )

    properties = [
        Property(
            name=PROP_DOCUMENT_ID,
            data_type=DataType.TEXT,
            tokenization=Tokenization.FIELD,
        ),
        Property(name=PROP_CHUNK_IND, data_type=DataType.INT),
        Property(name=PROP_CONTENT, data_type=DataType.TEXT),
        Property(name=PROP_BLURB, data_type=DataType.TEXT),
        Property(name=PROP_TITLE, data_type=DataType.TEXT),
        Property(name=PROP_SEMANTIC_IDENTIFIER, data_type=DataType.TEXT),
        Property(
            name=PROP_SOURCE_TYPE,
            data_type=DataType.TEXT,
            tokenization=Tokenization.FIELD,
        ),
        Property(name=PROP_SOURCE_LINKS, data_type=DataType.TEXT),
        Property(name=PROP_METADATA, data_type=DataType.TEXT),
        Property(name=PROP_BOOST, data_type=DataType.INT),
        Property(name=PROP_HIDDEN, data_type=DataType.BOOL),
        Property(name=PROP_IS_PUBLIC, data_type=DataType.BOOL),
        Property(
            name=PROP_DOCUMENT_SETS,
            data_type=DataType.TEXT_ARRAY,
            tokenization=Tokenization.FIELD,
        ),
        Property(
            name=PROP_ACCESS_CONTROL_LIST,
            data_type=DataType.TEXT_ARRAY,
            tokenization=Tokenization.FIELD,
        ),
        Property(name=PROP_LARGE_CHUNK_ID, data_type=DataType.INT),
    ]

    config: dict = dict(
        name=collection_name,
        properties=properties,
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
        ),
    )
    if multitenant:
        config["multi_tenancy_config"] = Configure.multi_tenancy(enabled=True)

    return config
