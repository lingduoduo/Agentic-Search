"""Verify every document_index file can be imported without error."""


def test_vespa_constants_importable():
    import src.backend.document_index.vespa_constants  # noqa: F401


def test_disabled_importable():
    import src.backend.document_index.disabled  # noqa: F401


def test_interfaces_new_importable():
    import src.backend.document_index.interfaces_new  # noqa: F401


def test_document_metadata_importable():
    import src.backend.document_index.document_metadata  # noqa: F401


def test_document_index_utils_importable():
    import src.backend.document_index.document_index_utils  # noqa: F401


def test_chunk_content_enrichment_importable():
    import src.backend.document_index.chunk_content_enrichment  # noqa: F401


def test_opensearch_constants_importable():
    import src.backend.document_index.opensearch.constants  # noqa: F401


def test_opensearch_schema_importable():
    import src.backend.document_index.opensearch.schema  # noqa: F401


def test_opensearch_search_importable():
    import src.backend.document_index.opensearch.search  # noqa: F401


def test_opensearch_client_importable():
    import src.backend.document_index.opensearch.client  # noqa: F401


def test_opensearch_document_index_importable():
    import src.backend.document_index.opensearch.opensearch_document_index  # noqa: F401


def test_vespa_internal_types_importable():
    import src.backend.document_index.vespa.internal_types  # noqa: F401


def test_vespa_shared_utils_importable():
    import src.backend.document_index.vespa.shared_utils.utils  # noqa: F401
    import src.backend.document_index.vespa.shared_utils.vespa_request_builders  # noqa: F401


def test_vespa_chunk_retrieval_importable():
    import src.backend.document_index.vespa.chunk_retrieval  # noqa: F401


def test_vespa_deletion_importable():
    import src.backend.document_index.vespa.deletion  # noqa: F401


def test_vespa_indexing_utils_importable():
    import src.backend.document_index.vespa.indexing_utils  # noqa: F401


def test_vespa_document_index_importable():
    import src.backend.document_index.vespa.vespa_document_index  # noqa: F401


def test_vespa_kg_interactions_importable():
    import src.backend.document_index.vespa.kg_interactions  # noqa: F401
